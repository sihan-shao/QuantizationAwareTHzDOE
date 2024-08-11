import sys

from utils.units import mm
sys.path.append('./')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC
import pathlib
import math
import matplotlib
import matplotlib.pyplot as plt
from copy import copy
from utils.units import *
from utils.Visualization_Helper import float_to_unit_identifier, add_colorbar
from DataType.ElectricField import ElectricField
import torch.nn.functional as F
from utils.Helper_Functions import UniformNoise


BASE_PLANE_THICKNESS = 2 * mm

LIGHT_SPEED = 2.998e8

def _copy_quad_to_full(quad_map):
    height_map_half_left = torch.cat([torch.flip(quad_map, dims=[0]), quad_map], dim=0)
    height_map_full = torch.cat([torch.flip(height_map_half_left, dims=[1]), height_map_half_left], dim=1)
    return height_map_full


class HologramLayer(ABC, nn.Module):
    
    @staticmethod
    def phase_to_height_with_material_func(phase      : torch.Tensor,
                                           wavlengths : torch.Tensor,
                                           epsilon    : torch.Tensor):
        """
        Calculates the height map from the quantized phase with certain refractive index and wavelengths
        # we use minimal wavelength to compute the height map based on the quantized phase map to warp all phase shift in the range of [0, 2*pi]
        Args:
            phase (torch.Tensor): optimized quantized phase map
            wavlengths (torch.Tensor): wavelngths
            epsilon (torch.Tensor): Material parameters including relative permittivity and loss tangent
            tand (torch.Tensor): Material parameters including relative permittivity and loss tangent

        Returns:
            The quantized height map 
        """
        #wavelengths = torch.tensor([0.5 * mm])
        #print(wavelengths)
        design_wavelength = torch.min(wavlengths)
        #design_wavelength = LIGHT_SPEED = 2.998e8 / 120e9
        
        return phase / (2 * torch.pi / design_wavelength) / (torch.sqrt(epsilon) - 1)
    
    @staticmethod
    def phase_shift_according_to_height(height_map  : torch.Tensor, 
                                        wavelengths : torch.Tensor, 
                                        epsilon     : torch.Tensor,
                                        tand        : torch.Tensor) -> torch.Tensor:
        
        """
        Calculates the phase shifts created by a height map with certain refractive index for light with specific wave lengths.

        Args:
            input_field     :   Input field.
            height_map      :   Hologram height map.
            wave_lengths    :   Wavelengths.
            materials_func  :   Material parameters including relative permittivity and loss tangent

        Returns: Modulated wave field.
        """
        height_map = height_map[None, :, :]
        # ensure wavelengths has one dimension, resulting shape: [C]
        wavelengths = wavelengths.view(-1)
        
        # add dimensions to wavelengths for broadcasting, resulting shape: [C, 1, 1]
        wavelengths = wavelengths[:, None, None]
        wave_numbers = 2 * torch.pi / wavelengths
        dev = wave_numbers.device

        # calculate loss and phase delay
        loss = torch.exp(-0.5 * wave_numbers * (height_map + torch.tensor(BASE_PLANE_THICKNESS, device=dev)) * tand * torch.sqrt(epsilon))
        phase_delay = torch.exp(-1j * wave_numbers * (height_map + torch.tensor(BASE_PLANE_THICKNESS, device=dev)) * (torch.sqrt(epsilon) - 1))
        
        # The hologram is not a thin element, we need to consider the air thickness 
        #air_phase = torch.exp(-1j * wave_numbers * torch.max(height_map))
        
        # calculate final phase shift combined with loss
        phase_shift = loss * phase_delay #* air_phase
        
        return phase_shift
    
    @staticmethod
    def add_height_map_noise(height_map, tolerance=None):
        dev = height_map.device
        if tolerance is not None:
            height_map = height_map + (torch.rand_like(height_map, device=dev) - 0.5) * 2 * tolerance
            
        return height_map
    
    @staticmethod
    def add_circ_aperture_to_field(input_field    : ElectricField, 
                                   circ_aperture  : bool = True, 
                                   radius=7.5*cm)-> torch.Tensor: 
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if circ_aperture is True:
            dx, dy = input_field.spacing[0], input_field.spacing[1]
            height, width = input_field.height, input_field.width
            radius = torch.tensor(radius)
            if radius==None:
                r = min([dx * height, dy * width]) / 2.0
            
            if radius!=None and radius < min([dx * height, dy * width]) / 2.0:
                r = radius
            else:
                ValueError('The radius should not larger than the physical length of E-field ')
            
            
            x = torch.linspace(-dx * height / 2, dx * height / 2, height, dtype=dx.dtype)
            
            y = torch.linspace(-dy * width / 2, dy * width / 2, width, dtype=dy.dtype)
                
            X, Y = torch.meshgrid(x, y)
            
            R = torch.sqrt(X**2 + Y**2)
            
            # Create a mask that is 1 inside the circle and 0 outside
            Mask = torch.where(R <= r, 1, 0)
            Mask = Mask[None, None, :, :]
        else:
            Mask = torch.ones_like(input_field.data)
        
        return Mask.to(device)
    
    @staticmethod
    def add_rect_aperture_to_field(input_field, 
                                   rect_aperture: bool = True, 
                                   rect_width=None, 
                                   rect_height=None)-> torch.Tensor:
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if rect_aperture is True:
            dx, dy = input_field.spacing[0], input_field.spacing[1]
            height, width = input_field.height, input_field.width
            
            if rect_width is None:
                rect_width = dx * width / 2
            if rect_height is None:
                rect_height = dy * height / 2
            
            # Ensure the rectangle dimensions are within the field's dimensions
            rect_width = min(rect_width, dx * width)
            rect_height = min(rect_height, dy * height)
            
            x = torch.linspace(-dx * width / 2, dx * width / 2, width, dtype=dx.dtype)
            y = torch.linspace(-dy * height / 2, dy * height / 2, height, dtype=dy.dtype)

            X, Y = torch.meshgrid(x, y, indexing='xy')

            # Create a mask that is 1 inside the rectangle and 0 outside
            Mask = torch.where((torch.abs(X) <= rect_width / 2) & (torch.abs(Y) <= rect_height / 2), 1, 0)
            Mask = Mask[None, None, :, :]
        
        else:
            Mask = torch.ones_like(input_field.data)
        
        return Mask.to(device)
    
    def build_height_map(self):
        return NotImplemented
    
    def modulate(self, 
                 input_field, 
                 preprocessed_height_map, 
                 height_tolerance, 
                 epsilon, 
                 tand,
                 circ_aperture)-> ElectricField:
        
        preprocessed_height_map = self.add_height_map_noise(preprocessed_height_map, tolerance=height_tolerance)
        
        # Update: 13.11.2023
        # The supsample functionality to align the size of input field and height map of hologram.
        if input_field.height != preprocessed_height_map.shape[0] or input_field.width != preprocessed_height_map.shape[1]:
            upsample_height = input_field.height
            upsample_width = input_field.width
            preprocessed_height_map = preprocessed_height_map[None, None, :, :]
            preprocessed_height_map = nn.functional.interpolate(preprocessed_height_map, size=[upsample_height, upsample_width], mode='nearest')
            self._height_map_ = torch.squeeze(preprocessed_height_map, (0, 1))
        # --------------------------------------- #
        else:
            self._height_map_ = torch.squeeze(preprocessed_height_map, (0, 1))

        
        phase_shift = self.phase_shift_according_to_height(height_map=self._height_map_,
                                                           wavelengths=input_field.wavelengths,
                                                           epsilon=epsilon, 
                                                           tand=tand)
        
        """
        self.Mask = self.add_circ_aperture_to_field(input_field=input_field, 
                                                    circ_aperture=circ_aperture)
                                                    
        Further step: 
        1. add a parameter for rect/circ aperture with defined dimensions
        """
        
        self.Mask = self.add_rect_aperture_to_field(input_field=input_field, 
                                                    rect_aperture=circ_aperture, 
                                                    rect_height=80*mm, 
                                                    rect_width=80*mm)
        
        

        modulate_field = self.Mask * input_field.data * phase_shift[None, :, :, :]   
        
        E_out = ElectricField(
            data = modulate_field,
            wavelengths=input_field.wavelengths,
            spacing = input_field.spacing
        )
        
        return E_out

    
    
class PhaseHologramElement(HologramLayer):
    def __init__(self, 
                 phase_map      : torch.Tensor,
                 holo_level     : int = 6,
                 look_up_table  : list = None,
                 tolerance      : float = 0.1*mm,
                 material       : list = None,
                 circ_aperture  : bool = True,
                 device         : torch.device = None) -> None:
        super(PhaseHologramElement, self).__init__()
        
        if device is None:
            self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device     = device
        
        self.phase_map      = torch.tensor(phase_map, device=self.device)
        self.holo_level     = holo_level
        self.tolerance      = torch.tensor(tolerance, device=self.device)
        self.material       = torch.tensor(material, device=self.device)
        self.epsilon        = self.material[0]   # relative permittivity of hologram
        self.tand           = self.material[1]   # loss tangent of hologram
        self.circ_aperture  = circ_aperture
        
        self.c_s = 300
        self.tau_max = 5.5
        self.tau_min = 5.5
        
        self.look_up_table(look_up_table)
    
    
    def look_up_table(self, look_up_table):
        # Phase look-up-table
        if look_up_table == None: 
            # linear phase look-up table
            lut = torch.linspace(-torch.pi, torch.pi, self.holo_level+1).to(self.device)
            self.lut = lut[:-1]
    
        if look_up_table != None:
            # non-linear or pre-defined phase look-up table 
            self.lut = torch.tensor(look_up_table, dtype=torch.float32).to(self.device)
            self.holo_level = len(self.lut)
            
    def score_phase(self, phase, lut, s=5.0, func='sigmoid'):
        
        # Here s is kinda representing the steepness
        
        wrapped_phase = (phase + torch.pi) % (2 * torch.pi) - torch.pi
        lut = lut[None, :, None, None]
        
        diff = wrapped_phase - lut
        diff = (diff + torch.pi) % (2*torch.pi) - torch.pi  # signed angular difference
        diff /= torch.pi  # normalize
        
        if func == 'sigmoid':
            z = s * diff
            scores = torch.sigmoid(z) * (1 - torch.sigmoid(z)) * 4
        elif func == 'log':
            scores = -torch.log(diff.abs() + 1e-20) * s
        elif func == 'poly':
            scores = (1-torch.abs(diff)**s)
        elif func == 'sine':
            scores = torch.cos(torch.pi * (s * diff).clamp(-1., 1.))
        elif func == 'chirp':
            scores = 1 - torch.cos(torch.pi * (1-diff.abs())**s)

        return scores


    def preprocessed_height_map(self, wavelengths):
        height, width = self.phase_map.shape[2], self.phase_map.shape[3]
        # Sample soft categorical using reparameterization trick
        
        scores = self.score_phase(self.phase_map, self.lut.to(self.phase_map.device), (self.tau_max / self.tau_min) **1) *  self.c_s * (self.tau_max / self.tau_min) **1
        
        softmax_scores = F.softmax(scores, dim=1)
        # Convert softmax probabilities to one-hot vectors
        index = softmax_scores.max(1, keepdim=True)[1]
        one_hot = torch.zeros_like(scores, memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)
        
        q_phase = (one_hot * self.lut.reshape(1, len(self.lut), 1, 1))
        q_phase = q_phase.sum(1, keepdim=True)
        
        # convert phase to positive to calculate the height map
        self.q_phase = (q_phase + 2 * torch.pi) % (2 * torch.pi)
        
        self.height_map = self.phase_to_height_with_material_func(phase=self.q_phase, 
                                                                  wavlengths=wavelengths,
                                                                  epsilon=self.epsilon)
        
        self.height_map = self.height_map.squeeze(0,1).to(self.device)
        #print(self.height_map.shape)
        return self.height_map
            
        
    def visualize(self,
                  cmap                    ='viridis',
                  figsize                 = (4,4)):
        """  visualize the thickness of the hologram
        """
        
        
        if self.circ_aperture == True:
            mask = self.Mask.squeeze(0,1).detach().cpu().numpy()  
            thickness = self._height_map_.squeeze(0,1).detach().cpu().numpy() * mask
        else:
            thickness = self._height_map_.squeeze(0,1).detach().cpu().numpy()
        
        if figsize is not None:
            fig = plt.figure(figsize=figsize)
            
        # First subplot: 2D plot
        plt.subplot(1, 1, 1)
        _im1 = plt.imshow(thickness, cmap=cmap)  # use colormap 'viridis'
        plt.title('2D Height Map of Hologram')
        plt.xlabel('X')
        plt.ylabel('Y')

        # Show the plots
        add_colorbar(_im1)
        plt.tight_layout()
        plt.show()
    
        """
    visualize the phase of the hologram
    def visualize_phase(self,
                        flag_axis : str         = True,
                        cmap                    ='viridis',
                        figsize                 = (4,4)):
                        
                        
        if self.Mask.shape[2] != self.q_phase.shape[2] or self.Mask.shape[3] != self.q_phase.shape[3]:
            upsample_height = self.Mask.shape[2]
            upsample_width = self.Mask.shape[3]
            upsample_phase_map = nn.functional.interpolate(self.q_phase, size=[upsample_height, upsample_width], mode='nearest')
            
        if self.circ_aperture == True:
            mask = self.Mask.squeeze(0,1).detach().cpu().numpy()  
            phase = upsample_phase_map.squeeze(0,1).detach().cpu().numpy() * mask
        else:
            phase = upsample_phase_map.squeeze(0,1).detach().cpu().numpy()
        
        if figsize is not None:
            fig = plt.figure(figsize=figsize)    
        """        
    
    
    def forward(self, field: ElectricField)->ElectricField:
        
        return self.modulate(input_field=field, 
                             preprocessed_height_map=self.preprocessed_height_map(wavelengths=field.wavelengths),
                             height_tolerance=self.tolerance, 
                             epsilon=self.epsilon,
                             tand=self.tand, 
                             circ_aperture=self.circ_aperture)



class SoftGumbelQuantizedHologramLayer(HologramLayer):
    def __init__(self, 
                 holo_size      : list = None,
                 holo_level     : int = 6,
                 look_up_table  : list = None,
                 num_unit       : int = None,
                 height_constraint_max: float = 4*mm,
                 tolerance      : float = 0.1*mm,
                 material       : list = None,
                 circ_aperture  : bool = True,
                 device         : torch.device = None) -> None:
        super(SoftGumbelQuantizedHologramLayer, self).__init__()
        
        if device is None:
            self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device     = device
        
        self.holo_size = holo_size
        self.holo_level = holo_level
        self.num_unit = num_unit
        
        # Coefficient mutliplied to score value - considering Gumbel noise scale
        self.c_s = 300
        self.tau_max = 5.5
        self.tau_min = 2.0
        
        self.height_constraint_max = torch.tensor(height_constraint_max, device=self.device)
        self.tolerance      = tolerance
        
        self.material       = torch.tensor(material, device=self.device)
        self.epsilon        = self.material[0] # relative permittivity of hologram
        self.tand           = self.material[1]   # loss tangent of hologram
        
        self.circ_aperture  = circ_aperture
        #self.weight_height_map = None
        #self.height_map = None
        self.look_up_table(look_up_table)
        self.build_init_phase()
        
        
    def visualize(self,
                  cmap                    ='viridis',
                  figsize                 = (4,4)):
        """  visualize the thickness of the hologram
        """
        
        
        if self.circ_aperture == True:
            mask = self.Mask.squeeze(0,1).detach().cpu().numpy()  
            thickness = self._height_map_.squeeze(0,1).detach().cpu().numpy() * mask
        else:
            thickness = self._height_map_.squeeze(0,1).detach().cpu().numpy()
        
        if figsize is not None:
            fig = plt.figure(figsize=figsize)
            
        # First subplot: 2D plot
        plt.subplot(1, 1, 1)
        _im1 = plt.imshow(thickness, cmap=cmap)  # use colormap 'viridis'
        plt.title('2D Height Map of Hologram')
        plt.xlabel('X')
        plt.ylabel('Y')

        # Show the plots
        add_colorbar(_im1)
        plt.tight_layout()
        plt.show()
        
    
    def look_up_table(self, look_up_table):
        # Phase look-up-table
        if look_up_table == None: 
            # linear phase look-up table
            lut = torch.linspace(-torch.pi, torch.pi, self.holo_level+1).to(self.device)
            self.lut = lut[:-1]
    
        if look_up_table != None:
            # non-linear or pre-defined phase look-up table 
            self.lut = torch.tensor(look_up_table, dtype=torch.float32).to(self.device)
            self.holo_level = len(self.lut)


    def build_init_phase(self):
        height, width = self.holo_size[0], self.holo_size[1]
        
        if self.num_unit is None:
            self.init_phase = nn.parameter.Parameter(
                -torch.pi + 2 * torch.pi * torch.rand(1, 1, height, width, device=self.device), 
                requires_grad=True
                )
        else:
            unit_size = [int(height / self.num_unit), int(width / self.num_unit)]
            self.init_phase = nn.parameter.Parameter(
                -torch.pi + 2 * torch.pi * torch.rand(1, 1, unit_size[0], unit_size[1], device=self.device), 
                requires_grad=True
                )
    
    
    def score_phase(self, phase, lut, s=5.0, func='sigmoid'):
        
        # Here s is kinda representing the steepness
        
        wrapped_phase = (phase + torch.pi) % (2 * torch.pi) - torch.pi
        lut = lut[None, :, None, None]
        
        diff = wrapped_phase - lut
        diff = (diff + torch.pi) % (2*torch.pi) - torch.pi  # signed angular difference
        diff /= torch.pi  # normalize
        
        if func == 'sigmoid':
            z = s * diff
            scores = torch.sigmoid(z) * (1 - torch.sigmoid(z)) * 4
        elif func == 'log':
            scores = -torch.log(diff.abs() + 1e-20) * s
        elif func == 'poly':
            scores = (1-torch.abs(diff)**s)
        elif func == 'sine':
            scores = torch.cos(torch.pi * (s * diff).clamp(-1., 1.))
        elif func == 'chirp':
            scores = 1 - torch.cos(torch.pi * (1-diff.abs())**s)

        return scores


    def preprocessed_height_map(self, wavelengths, tau):
        height, width = self.holo_size[0], self.holo_size[1]
        
        # Sample soft categorical using reparameterization trick
        
        scores = self.score_phase(self.init_phase, self.lut.to(self.init_phase.device), (self.tau_max / tau) **1) *  self.c_s * (self.tau_max / tau) **1
        
        one_hot = F.gumbel_softmax(scores, tau=tau, hard=True, dim=1)
        
        q_phase = (one_hot * self.lut.reshape(1, len(self.lut), 1, 1))
        q_phase = q_phase.sum(1, keepdim=True)
        
        
        
        # convert phase to positive to calculate the height map
        self.q_phase = (q_phase + 2 * torch.pi) % (2 * torch.pi)
        
        self.height_map = self.phase_to_height_with_material_func(phase=self.q_phase, 
                                                                  wavlengths=wavelengths,
                                                                  epsilon=self.epsilon)
        
        if self.num_unit is None:
            self.height_map = self.height_map.squeeze(0,1).to(self.device)
        #self.height_map = torch.round(height_map, decimals=3).squeeze(0,1).to(self.device)
        else:
            unit_height_map = _copy_quad_to_full(self.height_map)
            #print(unit_height_map.shape)
            unit_height, unit_width = unit_height_map.shape[0], unit_height_map.shape[1]
            #print(unit_height)
            self.height_map = unit_height_map.repeat(1, 1, int(height/unit_height), int(width/unit_width)).squeeze(0,1).to(self.device)
        #print(self.height_map.shape)
        return self.height_map
  
    
    def forward(self, field: ElectricField, iter_frac=None)->ElectricField:

        def tau_iter(iter_frac, tau_min=0.5, tau_max=50, r=None):
            if r is None:
                r = math.log(tau_max / tau_min)
            tau = max(tau_min, tau_max * math.exp(-r * iter_frac))
            return tau
        
        if iter_frac != None:
            tau = tau_iter(iter_frac=iter_frac, 
                           tau_min=self.tau_min, 
                           tau_max=self.tau_max)
        elif iter_frac == None:
            tau = None
        #print(tau)
        
        return self.modulate(input_field=field, 
                             preprocessed_height_map=self.preprocessed_height_map(wavelengths=field.wavelengths, tau=tau),
                             height_tolerance=self.tolerance, 
                             epsilon=self.epsilon,
                             tand=self.tand, 
                             circ_aperture=self.circ_aperture)           



class SoftGumbelQuantizedMaskLayer(HologramLayer):
    def __init__(self, 
                 holo_size      : list = None,
                 num_unit       : int = None,
                 circ_aperture  : bool = True,
                 device         : torch.device = None) -> None:
        super(SoftGumbelQuantizedMaskLayer, self).__init__()
        
        if device is None:
            self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device     = device
        
        self.holo_size = holo_size
        self.num_unit = num_unit
        
        # Coefficient mutliplied to score value - considering Gumbel noise scale
        self.c_s = 300
        self.tau_max = 5.5
        self.tau_min = 2.0

        self.circ_aperture  = circ_aperture
        #self.weight_height_map = None
        #self.height_map = None
        self.look_up_table()
        self.build_init_mask()
        
        
    def visualize(self,
                  cmap                    ='viridis',
                  figsize                 = (4,4)):
        """  visualize the thickness of the hologram
        """
        
        def circle_mask(input_tensor, radius=None):
            H, W = input_tensor.shape
            # Set default radius if not provided
            if radius is None:
                radius = min(H, W) / 2
            # Create a meshgrid
            y, x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
            
            # Compute distance to center
            center_y, center_x = H / 2, W / 2
            dist = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            # Create the mask
            mask = (dist <= radius).float()
            return mask.detach().cpu().numpy()  
        
        if self.circ_aperture == True:
            mask = circle_mask(self.mask)
            thickness = self.mask.detach().cpu().numpy() * mask
        else:
            thickness = self.mask.detach().cpu().numpy()
        
        if figsize is not None:
            fig = plt.figure(figsize=figsize)
            
        # First subplot: 2D plot
        plt.subplot(1, 1, 1)
        _im1 = plt.imshow(thickness, cmap=cmap)  # use colormap 'viridis'
        plt.title('2D Height Map of Hologram')
        plt.xlabel('X')
        plt.ylabel('Y')

        # Show the plots
        add_colorbar(_im1)
        plt.tight_layout()
        plt.show()
        
    
    def look_up_table(self):
        # amplitude look-up-table [-1, 1] --> [0, 1]
        self.lut = torch.tensor([-1, 1], dtype=torch.float32).to(self.device)


    def build_init_mask(self):
        height, width = self.holo_size[0], self.holo_size[1]
        
        if self.num_unit is None:
            self.init_mask = nn.parameter.Parameter(
                -2 + 4 * torch.rand(1, 1, height, width, device=self.device), 
                requires_grad=True
                )
        else:
            unit_size = [int(height / self.num_unit), int(width / self.num_unit)]
            self.init_mask = nn.parameter.Parameter(
                -2 + 4 * torch.rand(1, 1, unit_size[0], unit_size[1], device=self.device), 
                requires_grad=True
                )
    
    
    def score_phase(self, mask, lut, s=5.0, func='sigmoid'):
        
        # Here s is kinda representing the steepness
        
        wrapped_mask = (mask + 2) % (2 * 2) - 2
        lut = lut[None, :, None, None]
        
        diff = wrapped_mask - lut
        diff = (diff + 2) % (2*2) - 2  # signed angular difference
        diff /= 2  # normalize
        
        if func == 'sigmoid':
            z = s * diff
            scores = torch.sigmoid(z) * (1 - torch.sigmoid(z)) * 4
        elif func == 'log':
            scores = -torch.log(diff.abs() + 1e-20) * s
        elif func == 'poly':
            scores = (1-torch.abs(diff)**s)
        elif func == 'sine':
            scores = torch.cos(torch.pi * (s * diff).clamp(-1., 1.))
        elif func == 'chirp':
            scores = 1 - torch.cos(torch.pi * (1-diff.abs())**s)

        return scores


    def preprocessed_mask(self, tau):
        height, width = self.holo_size[0], self.holo_size[1]
        # Sample soft categorical using reparameterization trick
        
        scores = self.score_phase(self.init_mask, self.lut.to(self.init_mask.device), (self.tau_max / tau) **1) *  self.c_s * (self.tau_max / tau) **1
        
        one_hot = F.gumbel_softmax(scores, tau=tau, hard=True, dim=1)
        
        q_mask = (one_hot * self.lut.reshape(1, len(self.lut), 1, 1))
        q_mask = q_mask.sum(1, keepdim=True)
        
        # convert phase to positive to calculate the height map
        self.mask = (q_mask + 1) * 0.5
        
        if self.num_unit is None:
            self.mask = self.mask.squeeze(0,1).to(self.device)
        #self.height_map = torch.round(height_map, decimals=3).squeeze(0,1).to(self.device)
        else:
            unit_height_map = _copy_quad_to_full(self.mask)
            #print(unit_height_map.shape)
            unit_height, unit_width = unit_height_map.shape[0], unit_height_map.shape[1]
            #print(unit_height)
            self.mask = unit_height_map.repeat(1, 1, int(height/unit_height), int(width/unit_width)).squeeze(0,1).to(self.device)
        #print(self.height_map.shape)
        return self.mask
    
    def modulate(self, 
                 input_field: ElectricField, 
                 preprocessed_mask, 
                 circ_aperture: bool)->ElectricField:
        
        Mask = self.add_circ_aperture_to_field(input_field=input_field, 
                                               circ_aperture=circ_aperture)
        
        # Update: 13.11.2023
        # The supsample functionality to align the size of input field and height map of hologram.
        if input_field.height != preprocessed_mask.shape[0] or input_field.width != preprocessed_mask.shape[1]:
            upsample_height = input_field.height
            upsample_width = input_field.width
            preprocessed_mask = preprocessed_mask[None, None, :, :]
            preprocessed_mask = nn.functional.interpolate(preprocessed_mask, size=[upsample_height, upsample_width], mode='nearest')
            preprocessed_mask = torch.squeeze(preprocessed_mask, (0, 1))
        
        
        modulate_field = Mask * input_field.data * preprocessed_mask[None, None, :, :]   
        
        E_out = ElectricField(
            data = modulate_field,
            wavelengths=input_field.wavelengths,
            spacing = input_field.spacing
        )
        
        return E_out

    
    def forward(self, field: ElectricField, iter_frac=None)->ElectricField:

        def tau_iter(iter_frac, tau_min=0.5, tau_max=50, r=None):
            if r is None:
                r = math.log(tau_max / tau_min)
            tau = max(tau_min, tau_max * math.exp(-r * iter_frac))
            return tau
        
        if iter_frac != None:
            tau = tau_iter(iter_frac=iter_frac, 
                           tau_min=self.tau_min, 
                           tau_max=self.tau_max)
        elif iter_frac == None:
            tau = None
        #print(tau)
        
        return self.modulate(input_field=field, 
                             preprocessed_mask=self.preprocessed_mask(tau=tau),
                             circ_aperture=self.circ_aperture)      


if __name__ == '__main__':
    
    holo = SoftGumbelQuantizedHologramLayer(holo_size=10, holo_level=4, material=[2.6, 0.01], circ_aperture=False)