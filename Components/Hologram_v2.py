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


BASE_PLANE_THICKNESS = 2 * 1e-3


def _copy_quad_to_full(quad_map):
    height_map_half_left = torch.cat([torch.flip(quad_map, dims=[0]), quad_map], dim=0)
    height_map_full = torch.cat([torch.flip(height_map_half_left, dims=[1]), height_map_half_left], dim=1)
    return height_map_full


class HologramLayer(ABC, nn.Module):
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
        
        # calculate loss and phase delay
        loss = torch.exp(-0.5 * wave_numbers * height_map * tand * torch.sqrt(epsilon))
        phase_delay = torch.exp(-1j * wave_numbers * height_map * (torch.sqrt(epsilon) - 1))
        
        # The hologram is not a thin element, we need to consider the air thickness 
        air_phase = torch.exp(-1j * wave_numbers * torch.max(height_map))
        
        # calculate final phase shift combined with loss
        phase_shift = loss * phase_delay * air_phase
        
        return phase_shift
    
    @staticmethod
    def add_height_map_noise(height_map, tolerance=None):
        
        if tolerance is not None:
            height_map = height_map + (torch.rand_like(height_map) - 0.5) * 2 * tolerance
            
        return height_map
    
    @staticmethod
    def add_circ_aperture_to_field(input_field    : ElectricField, 
                                   circ_aperture  : bool = True)-> torch.Tensor: 
        if circ_aperture is True:
            dx, dy = input_field.spacing[0], input_field.spacing[1]
            height, width = input_field.height, input_field.width
            
            r = max([dx * height, dy * width]) / 2.0
            
            x = torch.linspace(-dx * height / 2, dx * height / 2, height, dtype=dx.dtype)
            
            y = torch.linspace(-dy * width / 2, dy * width / 2, width, dtype=dy.dtype)
                
            X, Y = torch.meshgrid(x, y)
            
            R = torch.sqrt(X**2 + Y**2)
            
            # Create a mask that is 1 inside the circle and 0 outside
            Mask = torch.where(R <= r, 1, 0)
            Mask = Mask[None, None, :, :]
        else:
            Mask = torch.ones_like(input_field.data)
        
        return Mask
    
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
        
        phase_shift = self.phase_shift_according_to_height(height_map=preprocessed_height_map,
                                                           wavelengths=input_field.wavelengths,
                                                           epsilon=epsilon, 
                                                           tand=tand)
        
        Mask = self.add_circ_aperture_to_field(input_field=input_field, 
                                               circ_aperture=circ_aperture)
        

        modulate_field = Mask * input_field.data * phase_shift[None, :, :, :]   
        
        E_out = ElectricField(
            data = modulate_field,
            wavelengths=input_field.wavelengths,
            spacing = input_field.spacing
        )
        
        return E_out


class HologramElement(HologramLayer):
    def __init__(self, 
                 height_map     : torch.Tensor,
                 tolerance      : float = 1e-4,
                 material       : list = None,
                 circ_aperture  : bool = True,
                 device         : torch.device = None) -> None:
        super(HologramElement, self).__init__()
        
        if device is None:
            self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device     = device
            
        self.height_map     = torch.tensor(height_map, device=device, dtype=torch.float32)
        self.tolerance      = tolerance
        self.material       = torch.tensor(material, device=self.device)
        self.epsilon        = self.material[0] # relative permittivity of hologram
        self.tand           = self.material[1]   # loss tangent of hologram
        
        self.circ_aperture  = circ_aperture
        
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
            mask = circle_mask(self.height_map)
            thickness = self.height_map.detach().cpu().numpy() * mask
        else:
            thickness = self.height_map.detach().cpu().numpy()
        
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
    
    
    def forward(self, field: ElectricField)->ElectricField:
        
        return self.modulate(input_field=field, 
                             preprocessed_height_map=self.height_map,
                             height_tolerance=self.tolerance, 
                             epsilon=self.epsilon,
                             tand=self.tand, 
                             circ_aperture=self.circ_aperture)



##### annealling factor ###############################
def tau_iter(iter_frac, tau_min=0.5, tau_max=3.0, r=None):
    if r is None:
        r = math.log(tau_max / tau_min)
    tau = max(tau_min, tau_max * math.exp(-r * iter_frac))
    return tau

#######################################################


class QuantizedDOELayer(HologramLayer):
    def __init__(self, 
                 holo_size      : list = None,
                 num_level     : int = 6,
                 look_up_table  : list = None,
                 num_unit       : int = None,
                 height_constraint_max: float = None,
                 tolerance      : float = 0.1*mm,
                 material       : list = None,
                 circ_aperture  : bool = True,
                 device         : torch.device = None) -> None:
        super(QuantizedDOELayer, self).__init__()
        
        if device is None:
            self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device     = device
            
        self.holo_size = holo_size
        self.num_level = num_level
        self.num_unit = num_unit
        
        self.height_constraint_max = torch.tensor(height_constraint_max, device=self.device)
        self.tolerance      = tolerance
        
        self.material       = torch.tensor(material, device=self.device)
        self.epsilon        = self.material[0] # relative permittivity of hologram
        self.tand           = self.material[1]   # loss tangent of hologram
        
        self.circ_aperture  = circ_aperture
        #self.weight_height_map = None
        #self.height_map = None
        self.look_up_table(look_up_table)
        self.build_height_map()
        
        
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
            mask = circle_mask(self.height_map)
            thickness = self.height_map.detach().cpu().numpy() * mask
        else:
            thickness = self.height_map.detach().cpu().numpy()
        
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
   
        if look_up_table == None: 
            # linear look-up table
            lut = torch.linspace(0, self.height_constraint_max, self.num_level+1).to(self.device)
            self.lut = lut[:-1]
    
        if look_up_table != None:
            # non-linear or pre-defined look-up table 
            self.lut = torch.tensor(self.lut, dtype=torch.float32).to(self.device)
            self.num_level = len(self.lut)
            self.height_constraint_max = torch.max(self.lut)


    def build_height_map(self):
        height, width = self.holo_size[0], self.holo_size[1]
        if self.num_unit is None:
            self.height_map = nn.parameter.Parameter(
                torch.rand(height, width, device=self.device) * self.height_constraint_max, requires_grad=True)
        else:
            unit_size = [int(height / self.num_unit), int(width / self.num_unit)]
            self.height_map = nn.parameter.Parameter(
                torch.rand(unit_size[0], unit_size[1], device=self.device) * self.height_constraint_max, requires_grad=True
            )


    def preprocessed_quantize_height_map(self, tau=1.0, max_tau=3.0, c=300., hard=True):
        height, width = self.holo_size[0], self.holo_size[1]
        lut = self.lut.reshape(1, self.num_level, 1, 1)
        
        # height to score
        diff = self.height_map - lut
        # normalization  TODO: consider a better normalization way
        diff = diff / (torch.max(torch.abs(diff))+1e-7)
        # score function
        s = (max_tau / tau)**1
        z = s * diff
        scores = torch.sigmoid(z) * (1 - torch.sigmoid(z)) * 4
        scores = c * s * scores
    
        # scores to one-hot encoding 
        one_hot = F.gumbel_softmax(scores, tau=tau, hard=hard, dim=1)
        # one-hot encoding to height value
        q_height_map = one_hot * lut
        self.q_height_map = q_height_map.sum(1, keepdims=True)
        
        if self.num_unit is None:
            self.q_height_map = self.q_height_map.squeeze(0,1).to(self.device)
        #self.height_map = torch.round(height_map, decimals=3).squeeze(0,1).to(self.device)
        else:
            unit_height_map = _copy_quad_to_full(self.q_height_map)
            #print(unit_height_map.shape)
            unit_height, unit_width = unit_height_map.shape[0], unit_height_map.shape[1]
            #print(unit_height)
            self.q_height_map = unit_height_map.repeat(1, 1, int(height/unit_height), int(width/unit_width)).squeeze(0,1).to(self.device)
        #print(self.height_map.shape)
        return self.q_height_map
    
    def forward(self, field: ElectricField, iter_frac=None)->ElectricField:
        #self.height_map = self.build_height_map()
        
        tau = tau_iter(iter_frac=iter_frac)
        #print(tau)
        return self.modulate(input_field=field, 
                             preprocessed_height_map=self.preprocessed_quantize_height_map(tau=tau),
                             height_tolerance=self.tolerance, 
                             epsilon=self.epsilon,
                             tand=self.tand, 
                             circ_aperture=self.circ_aperture)           
    


"""
Methods in Quantized DOE Layer

def look_up_table(self):
    input: self.lut
           self.num_level
           self.height_constraint_max
    
    if self.lut == None: 
        # linear look-up table
        lut = torch.linspace(0, self.height_constraint_max, self.num_level+1).to(self.device)
        self.lut = lut[:-1]
    
    if self.lut != None:
        # non-linear or pre-defined look-up table 
        self.lut = torch.tensor(self.lut, dtype=torch.float32).to(self.device)
        self.num_level = len(self.lut)
        self.height_constraint_max = torch.max(self.lut)
    
    # TODO: implement a differentiable look-up-table that is trainable
    

def build_height_map(self):
    height, width = self.holo_size[0], self.holo_size[1]
    if self.num_unit is None:
            self.height_map = nn.parameter.Parameter(
                torch.rand(height, width, device=self.device) * self.height_constraint_max, requires_grad=True)
        else:
            unit_size = [int(height / self.num_unit), int(width / self.num_unit)]
            self.height_map = nn.parameter.Parameter(
                torch.rand(unit_size[0], unit_size[1], device=self.device) * self.height_constraint_max, requires_grad=True
            )
            
def preprocessed_quantize_height_map(self, height_map, tau=1.0, max_tau=3.0, c=300., hard=True):
    lut = self.lut.reshape(1, self.num_level, 1, 1)
    
    # height to score
    diff = self.height_map - lut
    # normalization  TODO: consider a better normalization way
    diff /= torch.max(torch.abs(diff))
    
    # score function
    s = (tau_max / tau)**1
    z = s * diff
    scores = torch.sigmoid(z) * (1 - torch.sigmoid(z)) * 4
    scores = c * s * scores
    
    # scores to one-hot encoding 
    one_hot = F.gumbel_softmax(scores, tau=tau, hard=hard, dim=1)
    # one-hot encoding to height value
    q_height_map = (one_hot * lut)
    self.q_height_map = q_height_map.sum(1, keepdims=True)
    
    return self.q_height_map

"""
# implement Spherical Quantized Flat DOELens 


# implement Spectral Splitter Hologram



