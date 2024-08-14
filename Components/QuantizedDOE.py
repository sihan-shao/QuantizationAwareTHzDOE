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
    if len(quad_map.shape) == 4:
        height_map_half_left = torch.cat([torch.flip(quad_map, dims=[2]), quad_map], dim=2)
        height_map_full = torch.cat([torch.flip(height_map_half_left, dims=[3]), height_map_half_left], dim=3)
    else:
        height_map_half_left = torch.cat([torch.flip(quad_map, dims=[0]), quad_map], dim=0)
        height_map_full = torch.cat([torch.flip(height_map_half_left, dims=[1]), height_map_half_left], dim=1)
    return height_map_full

def _phase_to_height_with_material_refractive_idx(_phase, _wavelength, _refractive_index):
    return _phase / (2 * torch.pi / _wavelength) / (_refractive_index - 1)

def _height_to_phase_with_material_refractive_idx(_height, _wavelength, _refractive_index):
    return 2 * torch.pi / _wavelength * (_refractive_index - 1) * _height


class DOELayer(ABC, nn.Module):
    
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
        
        # calculate final phase shift combined with loss
        phase_shift = loss * phase_delay #* air_phase
        
        return phase_shift
    
    @staticmethod
    def add_height_map_noise(height_map, tolerance=None):
        dev = height_map.device
        if tolerance is not None:
            height_map = height_map + (torch.rand_like(height_map, device=dev) - 0.5) * 2 * tolerance
            
        return height_map
    
    def build_height_map(self):
        return NotImplemented
    
    def modulate(self, 
                 input_field, 
                 preprocessed_height_map, 
                 height_tolerance, 
                 epsilon, 
                 tand)-> ElectricField:
        
        preprocessed_height_map = self.add_height_map_noise(preprocessed_height_map, tolerance=height_tolerance)
        
        # The supsample functionality to align the size of input field and height map of doe
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

        modulate_field = input_field.data * phase_shift[None, :, :, :]   
        
        E_out = ElectricField(
            data = modulate_field,
            wavelengths=input_field.wavelengths,
            spacing = input_field.spacing
        )
        
        return E_out


class FixDOEElement(DOELayer):
    def __init__(self, 
                 height_map     : torch.Tensor,
                 tolerance      : float = 0.1*mm,
                 material       : list = None,
                 device         : torch.device = None) -> None:
        super(FixDOEElement, self).__init__()
        
        if device is None:
            self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device     = device
        
        self.height_map     = torch.tensor(height_map, device=self.device)
        self.tolerance      = torch.tensor(tolerance, device=self.device)
        self.material       = torch.tensor(material, device=self.device)
        self.epsilon        = self.material[0] # relative permittivity of DOE
        self.tand           = self.material[1]   # loss tangent of DOE
        
    def visualize(self,
                  cmap                    ='viridis',
                  figsize                 = (4,4)):
        """  visualize the thickness of the hologram
        """
        
        thickness = self.height_map.detach().cpu().numpy()
        
        if figsize is not None:
            fig = plt.figure(figsize=figsize)
            
        # First subplot: 2D plot
        plt.subplot(1, 1, 1)
        _im1 = plt.imshow(thickness, cmap=cmap)  # use colormap 'viridis'
        plt.title('2D Height Map of DOE')
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
                             tand=self.tand)


class FullPrecisionDOELayer(DOELayer):
    def __init__(self, 
                 doe_params     : dict,
                 device         : torch.device = None) -> None:
        super(FullPrecisionDOELayer, self).__init__()
        
        if device is None:
            self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device     = device
        
        # get the design parameters of DOE
        self.doe_size = doe_params.get('doe_size', None)
        self.doe_dxy = doe_params.get('doe_dxy', None)

        self.num_unit = doe_params.get('num_unit', None)

        height_constraint_max       = doe_params.get('height_constraint_max', 2 * mm)
        self.height_constraint_max  = torch.tensor(height_constraint_max, device=self.device)
        self.tolerance = doe_params.get('tolerance', 0.05 * mm)  # assuming mm to meters

        material            = doe_params.get('material', None)
        self.material       = torch.tensor(material, device=self.device)
        self.epsilon        = self.material[0]   # relative permittivity of doe
        self.tand           = self.material[1]   # loss tangent of doe
        
        self.build_weight_height_map()
        
        
    def visualize(self,
                  cmap                    ='viridis',
                  figsize                 = (4,4)):
        """  visualize the thickness of the doe
        """
        

        thickness = self.height_map.squeeze(0,1).detach().cpu().numpy()

        size_x = np.array(self.doe_dxy * self.doe_size[0] / 2)
        size_y = np.array(self.doe_dxy * self.doe_size[1] / 2)
        unit_val, unit_axis = float_to_unit_identifier(max(size_x,size_y))
        size_x = size_x / unit_val
        size_y = size_y / unit_val

        extent = [-size_x, size_x, -size_y, size_y]

        if figsize is not None:
            fig = plt.figure(figsize=figsize)
            
        # First subplot: 2D plot
        plt.subplot(1, 1, 1)
        _im1 = plt.imshow(thickness, cmap=cmap, extent=extent)  # use colormap 'viridis'
        plt.title('2D Height Map of DOE')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlabel("Position (" + unit_axis + ")")
        plt.ylabel("Position (" + unit_axis + ")")

        # Show the plots
        add_colorbar(_im1)
        plt.tight_layout()
        plt.show()
    
    def build_weight_height_map(self):
        height, width = self.doe_size[0], self.doe_size[1]
        
        #self.weight_height_map = nn.parameter.Parameter(
        #        torch.randn(height, width, device=self.device), requires_grad=True)
        if self.num_unit is None:
            self.weight_height_map = nn.parameter.Parameter(
                    -torch.pi + 2 * torch.pi * torch.rand(1, 1, height, width, device=self.device), 
                    requires_grad=True
                    )
        else:
            unit_size = [int(height / self.num_unit), int(width / self.num_unit)]
            self.weight_height_map = nn.parameter.Parameter(
                -torch.pi + 2 * torch.pi * torch.rand(1, 1, unit_size[0], unit_size[1], device=self.device), 
                requires_grad=True
                )

    def preprocessed_height_map(self):
        height_map = self.height_constraint_max * torch.sigmoid(torch.clamp(self.weight_height_map, min=-8.0, max=8.0))
        if self.num_unit is None:
            self.height_map = height_map.squeeze(0,1).to(self.device)
        else:
            self.height_map = _copy_quad_to_full(height_map).squeeze(0,1).to(self.device)
        return self.height_map
    
    
    def forward(self, field: ElectricField, iter_frac=None)->ElectricField:
        return self.modulate(input_field=field, 
                             preprocessed_height_map=self.preprocessed_height_map(),
                             height_tolerance=self.tolerance, 
                             epsilon=self.epsilon,
                             tand=self.tand)    


class SoftGumbelQuantizedDOELayer(DOELayer):
    def __init__(self, 
                 doe_params   : dict, 
                 optim_params : dict, 
                 device       : torch.device = None):
        super(SoftGumbelQuantizedDOELayer, self).__init__()

        if device is None:
            self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device     = device
        
        # get the design parameters of DOE
        self.doe_size = doe_params.get('doe_size', None)
        self.doe_dxy = doe_params.get('doe_dxy', None)
        self.doe_level = doe_params.get('doe_level', 6)

        self.num_unit = doe_params.get('num_unit', None)

        height_constraint_max       = doe_params.get('height_constraint_max', 2 * mm)
        self.height_constraint_max  = torch.tensor(height_constraint_max, device=self.device)
        self.tolerance = doe_params.get('tolerance', 0.05 * mm)  # assuming mm to meters

        material            = doe_params.get('material', None)
        self.material       = torch.tensor(material, device=self.device)
        self.epsilon        = self.material[0]   # relative permittivity of doe
        self.tand           = self.material[1]   # loss tangent of doe

        # get the hyper-parameters of optimization
        self.c_s = optim_params.get('c_s', 300)
        self.tau_max = optim_params.get('tau_max', 5.5)
        self.tau_min = optim_params.get('tau_min', 2.0)

        look_up_table = doe_params.get('look_up_table', None)
        self.look_up_table(look_up_table)
        self.build_init_phase()

    def visualize(self,
                  cmap                    ='viridis',
                  figsize                 = (4,4)):
        """  visualize the thickness of the doe
        """
        

        thickness = self.height_map.squeeze(0,1).detach().cpu().numpy()

        size_x = np.array(self.doe_dxy * self.doe_size[0] / 2)
        size_y = np.array(self.doe_dxy * self.doe_size[1] / 2)
        unit_val, unit_axis = float_to_unit_identifier(max(size_x,size_y))
        size_x = size_x / unit_val
        size_y = size_y / unit_val

        extent = [-size_x, size_x, -size_y, size_y]

        if figsize is not None:
            fig = plt.figure(figsize=figsize)
            
        # First subplot: 2D plot
        plt.subplot(1, 1, 1)
        _im1 = plt.imshow(thickness, cmap=cmap, extent=extent)  # use colormap 'viridis'
        plt.title('2D Height Map of DOE')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlabel("Position (" + unit_axis + ")")
        plt.ylabel("Position (" + unit_axis + ")")

        # Show the plots
        add_colorbar(_im1)
        plt.tight_layout()
        plt.show()

    def look_up_table(self, look_up_table):
        """
        look-up-table with manufacturable height values

        look_up_table: 
            list of manufacturable height values
        if look_up_table is None:
            use height_constraint_max and doe_level to build a look_up_table
        """
        if look_up_table == None: 
            # linear height look-up table
            lut = torch.linspace(0, self.height_constraint_max, self.doe_level+1).to(self.device)
            self.lut = lut[:-1]
    
        if look_up_table != None:
            # non-linear or pre-defined height look-up table 
            self.lut = torch.tensor(look_up_table, dtype=torch.float32).to(self.device)
            self.doe_level = len(self.lut)

    def build_init_phase(self):
        height, width = self.doe_size[0], self.doe_size[1]
        
        if self.num_unit is None:
            #self.weight_init_phase = nn.parameter.Parameter(
            #    torch.randn(height, width, device=self.device), requires_grad=True
            #)
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
        lut = (lut[None, :, None, None] + torch.pi) % (2 * torch.pi) - torch.pi
        
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
        height, width = self.doe_size[0], self.doe_size[1]

        # convert look_up_table to phase lut with designed wavelength
        phase_lut = _height_to_phase_with_material_refractive_idx(self.lut, wavelengths.min(), torch.sqrt(self.epsilon))

        #self.init_phase = 2 * torch.pi * torch.sigmoid(torch.clamp(self.weight_init_phase, min=-12.0, max=12.0))
        # Sample soft categorical using reparameterization trick
        #scores = self.score_phase(self.init_phase, phase_lut.to(self.init_phase.device), (self.tau_max / tau) **1) *  self.c_s * (self.tau_max / tau) **1
        scores = self.score_phase(self.init_phase, phase_lut.to(self.init_phase.device), (self.tau_max / tau) **1) *  self.c_s * (self.tau_max / tau) **1
        #print(scores)
        one_hot = F.gumbel_softmax(scores, tau=tau, hard=True, dim=1)
        self.height_map = (self.lut.reshape(1, len(self.lut), 1, 1) * one_hot).sum(1, keepdim=True)

        if self.num_unit is None:
            self.height_map = self.height_map.squeeze(0,1).to(self.device)
        else:
            unit_height_map = _copy_quad_to_full(self.height_map)
            #print(unit_height_map.shape)
            unit_height, unit_width = unit_height_map.shape[0], unit_height_map.shape[1]
            #print(unit_height)
            self.height_map = unit_height_map.repeat(1, 1, int(height/unit_height), int(width/unit_width)).squeeze(0,1).to(self.device)
        #print(self.height_map.shape)
        return self.height_map

    def forward(self, field: ElectricField, iter_frac=None)->ElectricField:

        #def tau_iter(iter_frac, tau_min=0.5, tau_max=50, r=None):
        #    if r is None:
        #        r = math.log(tau_max / tau_min)
        #    tau = max(tau_min, tau_max * math.exp(-r * iter_frac))
        #    return tau
        def tau_iter(iter_frac, tau_min=0.5, tau_max=50):
            tau = tau_min + 0.5 * (tau_max - tau_min) * (1 + math.cos(iter_frac * math.pi))
            return tau
        #def tau_iter(iter_frac, tau_min=0.5, tau_max=50, r=None):
            # linearly increasing tenperature
        #    delta_tau = tau_max - tau_min
        #    tau = tau_min + delta_tau * iter_frac
        #    return tau
        
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
                             tand=self.tand)     


class SoftGumbelQuantizedDOELayerv2(DOELayer):
    def __init__(self, 
                 doe_params   : dict, 
                 optim_params : dict, 
                 device       : torch.device = None):
        super(SoftGumbelQuantizedDOELayerv2, self).__init__()

        if device is None:
            self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device     = device
        
        # get the design parameters of DOE
        self.doe_size = doe_params.get('doe_size', None)
        self.doe_dxy = doe_params.get('doe_dxy', None)
        self.doe_level = doe_params.get('doe_level', 6)

        self.num_unit = doe_params.get('num_unit', None)

        height_constraint_max       = doe_params.get('height_constraint_max', 2 * mm)
        self.height_constraint_max  = torch.tensor(height_constraint_max, device=self.device)
        self.tolerance = doe_params.get('tolerance', 0.05 * mm)  # assuming mm to meters

        material            = doe_params.get('material', None)
        self.material       = torch.tensor(material, device=self.device)
        self.epsilon        = self.material[0]   # relative permittivity of doe
        self.tand           = self.material[1]   # loss tangent of doe

        # get the hyper-parameters of optimization
        self.c_s = optim_params.get('c_s', 300)
        self.tau_max = optim_params.get('tau_max', 5.5)
        self.tau_min = optim_params.get('tau_min', 2.0)

        look_up_table = doe_params.get('look_up_table', None)
        self.look_up_table(look_up_table)
        self.build_init_phase()

    def visualize(self,
                  cmap                    ='viridis',
                  figsize                 = (4,4)):
        """  visualize the thickness of the doe
        """
        

        thickness = self.height_map.squeeze(0,1).detach().cpu().numpy()

        size_x = np.array(self.doe_dxy * self.doe_size[0] / 2)
        size_y = np.array(self.doe_dxy * self.doe_size[1] / 2)
        unit_val, unit_axis = float_to_unit_identifier(max(size_x,size_y))
        size_x = size_x / unit_val
        size_y = size_y / unit_val

        extent = [-size_x, size_x, -size_y, size_y]

        if figsize is not None:
            fig = plt.figure(figsize=figsize)
            
        # First subplot: 2D plot
        plt.subplot(1, 1, 1)
        _im1 = plt.imshow(thickness, cmap=cmap, extent=extent)  # use colormap 'viridis'
        plt.title('2D Height Map of DOE')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlabel("Position (" + unit_axis + ")")
        plt.ylabel("Position (" + unit_axis + ")")

        # Show the plots
        add_colorbar(_im1)
        plt.tight_layout()
        plt.show()

    def look_up_table(self, look_up_table):
        """
        look-up-table with manufacturable height values

        look_up_table: 
            list of manufacturable height values
        if look_up_table is None:
            use height_constraint_max and doe_level to build a look_up_table
        """
        if look_up_table == None: 
            # linear height look-up table
            lut = torch.linspace(0, self.height_constraint_max, self.doe_level+1).to(self.device)
            self.lut = lut[:-1]
    
        if look_up_table != None:
            # non-linear or pre-defined height look-up table 
            self.lut = torch.tensor(look_up_table, dtype=torch.float32).to(self.device)
            self.doe_level = len(self.lut)

    def build_init_phase(self):
        height, width = self.doe_size[0], self.doe_size[1]
        
        if self.num_unit is None:
            self.weight_init_phase = nn.parameter.Parameter(
                torch.randn(height, width, device=self.device), requires_grad=True
            )
            #self.init_phase = nn.parameter.Parameter(
            #    -torch.pi + 2 * torch.pi * torch.rand(1, 1, height, width, device=self.device), 
            #    requires_grad=True
            #    )
        else:
            unit_size = [int(height / self.num_unit), int(width / self.num_unit)]
            self.init_phase = nn.parameter.Parameter(
                -torch.pi + 2 * torch.pi * torch.rand(1, 1, unit_size[0], unit_size[1], device=self.device), 
                requires_grad=True
                )

    def score_phase(self, phase, lut, s=5.0, func='sigmoid'):
        
        # Here s is kinda representing the steepness
        
        wrapped_phase = (phase + torch.pi) % (2 * torch.pi) - torch.pi
        lut = (lut[None, :, None, None] + torch.pi) % (2 * torch.pi) - torch.pi
        
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
    
    def preprocessed_height_map(self, wavelengths, tau, iter_frac=None):
        height, width = self.doe_size[0], self.doe_size[1]

        #if iter_frac<= 0.5:
        height_map = self.height_constraint_max * torch.sigmoid(torch.clamp(self.weight_init_phase, min=-10.0, max=10.0))
        height_map = height_map[None, None, :, :]
        
        if iter_frac > 0.5:

            # convert look_up_table to phase lut with designed wavelength
            phase_lut = _height_to_phase_with_material_refractive_idx(self.lut, wavelengths.min(), torch.sqrt(self.epsilon))
            quantized_phase = _height_to_phase_with_material_refractive_idx(height_map, wavelengths.min(), torch.sqrt(self.epsilon))
            # Sample soft categorical using reparameterization trick
            #scores = self.score_phase(self.init_phase, phase_lut.to(self.init_phase.device), (self.tau_max / tau) **1) *  self.c_s * (self.tau_max / tau) **1
            scores = self.score_phase(quantized_phase, phase_lut.to(quantized_phase.device), (self.tau_max / tau) **1) *  self.c_s * (self.tau_max / tau) **1
            #print(scores)
            one_hot = F.gumbel_softmax(scores, tau=tau, hard=True, dim=1)
            height_map = (self.lut.reshape(1, len(self.lut), 1, 1) * one_hot).sum(1, keepdim=True)

        if self.num_unit is None:
            self.height_map = height_map.squeeze(0,1).to(self.device)
        else:
            unit_height_map = _copy_quad_to_full(self.height_map)
            #print(unit_height_map.shape)
            unit_height, unit_width = unit_height_map.shape[0], unit_height_map.shape[1]
            #print(unit_height)
            self.height_map = unit_height_map.repeat(1, 1, int(height/unit_height), int(width/unit_width)).squeeze(0,1).to(self.device)
        #print(self.height_map.shape)
        return self.height_map

    def forward(self, field: ElectricField, iter_frac=None)->ElectricField:

        #def tau_iter(iter_frac, tau_min=0.5, tau_max=50, r=None):
        #    if r is None:
        #        r = math.log(tau_max / tau_min)
        #    tau = max(tau_min, tau_max * math.exp(-r * iter_frac))
        #    return tau
        def tau_iter(iter_frac, tau_min=0.5, tau_max=50):
            tau = tau_min + 0.5 * (tau_max - tau_min) * (1 + math.cos(iter_frac * math.pi))
            return tau
        #def tau_iter(iter_frac, tau_min=0.5, tau_max=50, r=None):
            # linearly increasing tenperature
        #    delta_tau = tau_max - tau_min
        #    tau = tau_min + delta_tau * iter_frac
        #    return tau
        
        if iter_frac != None:
            tau = tau_iter(iter_frac=iter_frac, 
                           tau_min=self.tau_min, 
                           tau_max=self.tau_max)
        elif iter_frac == None:
            tau = None
        #print(tau)
        
        return self.modulate(input_field=field, 
                             preprocessed_height_map=self.preprocessed_height_map(wavelengths=field.wavelengths, tau=tau, iter_frac=iter_frac),
                             height_tolerance=self.tolerance, 
                             epsilon=self.epsilon,
                             tand=self.tand)    



class SoftGumbelQuantizedDOELayerv3(DOELayer):
    def __init__(self, 
                 doe_params   : dict, 
                 optim_params : dict, 
                 device       : torch.device = None):
        super(SoftGumbelQuantizedDOELayerv3, self).__init__()

        if device is None:
            self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device     = device
        
        # get the design parameters of DOE
        self.doe_size = doe_params.get('doe_size', None)
        self.doe_dxy = doe_params.get('doe_dxy', None)
        self.doe_level = doe_params.get('doe_level', 6)

        self.num_unit = doe_params.get('num_unit', None)

        height_constraint_max       = doe_params.get('height_constraint_max', 2 * mm)
        self.height_constraint_max  = torch.tensor(height_constraint_max, device=self.device)
        self.tolerance = doe_params.get('tolerance', 0.05 * mm)  # assuming mm to meters

        material            = doe_params.get('material', None)
        self.material       = torch.tensor(material, device=self.device)
        self.epsilon        = self.material[0]   # relative permittivity of doe
        self.tand           = self.material[1]   # loss tangent of doe

        # get the hyper-parameters of optimization
        self.c_s = optim_params.get('c_s', 300)
        self.tau_max = optim_params.get('tau_max', 5.5)
        self.tau_min = optim_params.get('tau_min', 2.0)

        look_up_table = doe_params.get('look_up_table', None)
        self.look_up_table(look_up_table)
        self.build_init_phase()

    def visualize(self,
                  cmap                    ='viridis',
                  figsize                 = (4,4)):
        """  visualize the thickness of the doe
        """
        

        thickness = self.height_map.squeeze(0,1).detach().cpu().numpy()

        size_x = np.array(self.doe_dxy * self.doe_size[0] / 2)
        size_y = np.array(self.doe_dxy * self.doe_size[1] / 2)
        unit_val, unit_axis = float_to_unit_identifier(max(size_x,size_y))
        size_x = size_x / unit_val
        size_y = size_y / unit_val

        extent = [-size_x, size_x, -size_y, size_y]

        if figsize is not None:
            fig = plt.figure(figsize=figsize)
            
        # First subplot: 2D plot
        plt.subplot(1, 1, 1)
        _im1 = plt.imshow(thickness, cmap=cmap, extent=extent)  # use colormap 'viridis'
        plt.title('2D Height Map of DOE')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlabel("Position (" + unit_axis + ")")
        plt.ylabel("Position (" + unit_axis + ")")

        # Show the plots
        add_colorbar(_im1)
        plt.tight_layout()
        plt.show()

    def look_up_table(self, look_up_table):
        """
        look-up-table with manufacturable height values

        look_up_table: 
            list of manufacturable height values
        if look_up_table is None:
            use height_constraint_max and doe_level to build a look_up_table
        """
        if look_up_table == None: 
            # linear height look-up table
            lut = torch.linspace(0, self.height_constraint_max, self.doe_level+1).to(self.device)
            self.lut = lut[:-1]
    
        if look_up_table != None:
            # non-linear or pre-defined height look-up table 
            self.lut = torch.tensor(look_up_table, dtype=torch.float32).to(self.device)
            self.doe_level = len(self.lut)

    def build_init_phase(self):
        height, width = self.doe_size[0], self.doe_size[1]
        
        if self.num_unit is None:
            self.weight_init_phase = nn.parameter.Parameter(
                torch.randn(height, width, device=self.device), requires_grad=True
            )
            #self.init_phase = nn.parameter.Parameter(
            #    -torch.pi + 2 * torch.pi * torch.rand(1, 1, height, width, device=self.device), 
            #    requires_grad=True
            #    )
        else:
            unit_size = [int(height / self.num_unit), int(width / self.num_unit)]
            self.weight_init_phase = nn.parameter.Parameter(
                torch.randn(unit_size[0], unit_size[1], device=self.device), 
                requires_grad=True
                )

    def score_phase(self, phase, lut, s=5.0, func='sigmoid'):
        
        # Here s is kinda representing the steepness
        
        wrapped_phase = (phase + torch.pi) % (2 * torch.pi) - torch.pi
        lut = (lut[None, :, None, None] + torch.pi) % (2 * torch.pi) - torch.pi
        
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
    
    def preprocessed_height_map(self, wavelengths, tau, iter_frac=None):
        height, width = self.doe_size[0], self.doe_size[1]

        #if iter_frac<= 0.5:
        height_map = self.height_constraint_max * torch.sigmoid(torch.clamp(self.weight_init_phase, min=-10.0, max=10.0))
        height_map = height_map[None, None, :, :]
        
        if iter_frac > 0.3 and iter_frac <= 0.8:
            beta = (iter_frac - 0.3) / (0.8 - 0.3)
            # convert look_up_table to phase lut with designed wavelength
            phase_lut = _height_to_phase_with_material_refractive_idx(self.lut, wavelengths.min(), torch.sqrt(self.epsilon))
            quantized_phase = _height_to_phase_with_material_refractive_idx(height_map, wavelengths.min(), torch.sqrt(self.epsilon))
            # Sample soft categorical using reparameterization trick
            #scores = self.score_phase(self.init_phase, phase_lut.to(self.init_phase.device), (self.tau_max / tau) **1) *  self.c_s * (self.tau_max / tau) **1
            scores = self.score_phase(quantized_phase, phase_lut.to(quantized_phase.device), (self.tau_max / tau) **1) *  self.c_s * (self.tau_max / tau) **1
            #print(scores)
            one_hot = F.gumbel_softmax(scores, tau=tau, hard=True, dim=1)
            q_height_map = (self.lut.reshape(1, len(self.lut), 1, 1) * one_hot).sum(1, keepdim=True)
            height_map = (1 - iter_frac) * height_map + iter_frac * q_height_map

        if iter_frac > 0.8:

            # convert look_up_table to phase lut with designed wavelength
            phase_lut = _height_to_phase_with_material_refractive_idx(self.lut, wavelengths.min(), torch.sqrt(self.epsilon))
            quantized_phase = _height_to_phase_with_material_refractive_idx(height_map, wavelengths.min(), torch.sqrt(self.epsilon))
            # Sample soft categorical using reparameterization trick
            #scores = self.score_phase(self.init_phase, phase_lut.to(self.init_phase.device), (self.tau_max / tau) **1) *  self.c_s * (self.tau_max / tau) **1
            scores = self.score_phase(quantized_phase, phase_lut.to(quantized_phase.device), (self.tau_max / tau) **1) *  self.c_s * (self.tau_max / tau) **1
            #print(scores)
            one_hot = F.gumbel_softmax(scores, tau=tau, hard=True, dim=1)
            height_map = (self.lut.reshape(1, len(self.lut), 1, 1) * one_hot).sum(1, keepdim=True)

        if self.num_unit is None:
            self.height_map = height_map.squeeze(0,1).to(self.device)
        else:
            self.height_map = _copy_quad_to_full(height_map).squeeze(0,1).to(self.device)
            #print(unit_height_map.shape)
            #unit_height, unit_width = unit_height_map.shape[0], unit_height_map.shape[1]
            #print(unit_height)
            #self.height_map = unit_height_map.repeat(1, 1, int(height/unit_height), int(width/unit_width)).squeeze(0,1).to(self.device)
        #print(self.height_map.shape)
        return self.height_map

    def forward(self, field: ElectricField, iter_frac=None)->ElectricField:

        #def tau_iter(iter_frac, tau_min=0.5, tau_max=50, r=None):
        #    if r is None:
        #        r = math.log(tau_max / tau_min)
        #    tau = max(tau_min, tau_max * math.exp(-r * iter_frac))
        #    return tau
        def tau_iter(iter_frac, tau_min=0.5, tau_max=50):
            tau = tau_min + 0.5 * (tau_max - tau_min) * (1 + math.cos(iter_frac * math.pi))
            return tau
        #def tau_iter(iter_frac, tau_min=0.5, tau_max=50, r=None):
            # linearly increasing tenperature
        #    delta_tau = tau_max - tau_min
        #    tau = tau_min + delta_tau * iter_frac
        #    return tau
        
        if iter_frac != None:
            tau = tau_iter(iter_frac=iter_frac, 
                           tau_min=self.tau_min, 
                           tau_max=self.tau_max)
        elif iter_frac == None:
            tau = None
        #print(tau)
        
        return self.modulate(input_field=field, 
                             preprocessed_height_map=self.preprocessed_height_map(wavelengths=field.wavelengths, tau=tau, iter_frac=iter_frac),
                             height_tolerance=self.tolerance, 
                             epsilon=self.epsilon,
                             tand=self.tand)    

class NaiveGumbelQuantizedDOELayer(DOELayer):
    def __init__(self, 
                 doe_params   : dict, 
                 optim_params : dict, 
                 device       : torch.device = None):
        super(NaiveGumbelQuantizedDOELayer, self).__init__()

        if device is None:
            self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device     = device
        
        # get the design parameters of DOE
        self.doe_size = doe_params.get('doe_size', None)
        self.doe_dxy = doe_params.get('doe_dxy', None)
        self.doe_level = doe_params.get('doe_level', 6)

        self.num_unit = doe_params.get('num_unit', None)

        height_constraint_max       = doe_params.get('height_constraint_max', 2 * mm)
        self.height_constraint_max  = torch.tensor(height_constraint_max, device=self.device)
        self.tolerance = doe_params.get('tolerance', 0.05 * mm)  # assuming mm to meters

        material            = doe_params.get('material', None)
        self.material       = torch.tensor(material, device=self.device)
        self.epsilon        = self.material[0]   # relative permittivity of doe
        self.tand           = self.material[1]   # loss tangent of doe

        # get the hyper-parameters of optimization
        self.c_s = optim_params.get('c_s', 300)
        self.tau_max = optim_params.get('tau_max', 5.5)
        self.tau_min = optim_params.get('tau_min', 2.0)

        look_up_table = doe_params.get('look_up_table', None)
        self.look_up_table(look_up_table)
        self.build_init_logits()
    

    def visualize(self,
                  cmap                    ='viridis',
                  figsize                 = (4,4)):
        """  visualize the thickness of the doe
        """
        

        thickness = self.height_map.squeeze(0,1).detach().cpu().numpy()

        size_x = np.array(self.doe_dxy * self.doe_size[0] / 2)
        size_y = np.array(self.doe_dxy * self.doe_size[1] / 2)
        unit_val, unit_axis = float_to_unit_identifier(max(size_x,size_y))
        size_x = size_x / unit_val
        size_y = size_y / unit_val

        extent = [-size_x, size_x, -size_y, size_y]

        if figsize is not None:
            fig = plt.figure(figsize=figsize)
            
        # First subplot: 2D plot
        plt.subplot(1, 1, 1)
        _im1 = plt.imshow(thickness, cmap=cmap, extent=extent)  # use colormap 'viridis'
        plt.title('2D Height Map of DOE')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlabel("Position (" + unit_axis + ")")
        plt.ylabel("Position (" + unit_axis + ")")

        # Show the plots
        add_colorbar(_im1)
        plt.tight_layout()
        plt.show()

    def look_up_table(self, look_up_table):
        """
        look-up-table with manufacturable height values

        look_up_table: 
            list of manufacturable height values
        if look_up_table is None:
            use height_constraint_max and doe_level to build a look_up_table
        """
        if look_up_table == None: 
            # linear height look-up table
            lut = torch.linspace(0, self.height_constraint_max, self.doe_level+1).to(self.device)
            self.lut = lut[:-1]
    
        if look_up_table != None:
            # non-linear or pre-defined height look-up table 
            self.lut = torch.tensor(look_up_table, dtype=torch.float32).to(self.device)
            self.doe_level = len(self.lut)

    def build_init_logits(self):
        height, width = self.doe_size[0], self.doe_size[1]
        
        if self.num_unit is None:
            self.weight_height_map = nn.parameter.Parameter(
                torch.rand(height, width, self.doe_level, device=self.device), requires_grad=True)
        else:
            unit_size = [int(height / self.num_unit), int(width / self.num_unit)]
            self.weight_height_map = nn.parameter.Parameter(
                torch.rand(unit_size[0], unit_size[1], self.doe_level, device=self.device), requires_grad=True
            )

    def preprocessed_height_map(self, tau):
        height, width = self.doe_size[0], self.doe_size[1]
        # Sample soft categorical using reparameterization trick
        if tau != None:
            sample_one_hot = F.gumbel_softmax(self.weight_height_map, tau=tau, hard=True)
        elif tau == None:
            sample_one_hot = F.gumbel_softmax(self.weight_height_map, tau=1, hard=True)

        height_map = (self.lut[None, None, :] * sample_one_hot).sum(dim=-1)
        height_map = height_map[None, None, :, :]
        if self.num_unit is None:
            self.height_map = height_map.squeeze(0,1).to(self.device)
        else:
            self.height_map = _copy_quad_to_full(height_map).squeeze(0,1).to(self.device)
            #print(unit_height_map.shape)
            #unit_height, unit_width = unit_height_map.shape[0], unit_height_map.shape[1]
            #print(unit_height)
            #self.height_map = unit_height_map.repeat(1, 1, int(height/unit_height), int(width/unit_width)).squeeze(0,1).to(self.device)
        #print(self.height_map.shape)
        return self.height_map

    def forward(self, field: ElectricField, iter_frac=None)->ElectricField:
        #def tau_iter(iter_frac, tau_min=0.5, tau_max=50, r=None):
        #    if r is None:
        #        r = math.log(tau_max / tau_min)
        #    tau = max(tau_min, tau_max * math.exp(-r * iter_frac))
        #    return tau

        def tau_iter(iter_frac, tau_min=0.5, tau_max=50):
            tau = tau_min + 0.5 * (tau_max - tau_min) * (1 + math.cos(iter_frac * math.pi))
            return tau
        
        if iter_frac != None:
            tau = tau_iter(iter_frac=iter_frac, 
                           tau_min=self.tau_min, 
                           tau_max=self.tau_max)
        elif iter_frac == None:
            tau = None

        return self.modulate(input_field=field, 
                             preprocessed_height_map=self.preprocessed_height_map(tau=tau),
                             height_tolerance=self.tolerance, 
                             epsilon=self.epsilon,
                             tand=self.tand)   


class PSQuantizedDOELayer(DOELayer):
    # Progressive Sigmoid Quantization (PSQ) method
    def __init__(self, 
                 doe_params   : dict, 
                 optim_params : dict, 
                 device       : torch.device = None):
        super(PSQuantizedDOELayer, self).__init__()

        if device is None:
            self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device     = device
        
        # get the design parameters of DOE
        self.doe_size = doe_params.get('doe_size', None)
        self.doe_dxy = doe_params.get('doe_dxy', None)
        self.doe_level = doe_params.get('doe_level', 6)

        self.num_unit = doe_params.get('num_unit', None)

        height_constraint_max       = doe_params.get('height_constraint_max', 2 * mm)
        self.height_constraint_max  = torch.tensor(height_constraint_max, device=self.device)
        self.tolerance = doe_params.get('tolerance', 0.05 * mm)  # assuming mm to meters

        material            = doe_params.get('material', None)
        self.material       = torch.tensor(material, device=self.device)
        self.epsilon        = self.material[0]   # relative permittivity of doe
        self.tand           = self.material[1]   # loss tangent of doe

        # get the hyper-parameters of optimization
        self.tau_max = optim_params.get('tau_max', 400)
        self.tau_min = optim_params.get('tau_min', 1)
        self.build_weight_height_map()

    def visualize(self,
                  cmap                    ='viridis',
                  figsize                 = (4,4)):
        """  visualize the thickness of the doe
        """
        

        thickness = self.height_map.squeeze(0,1).detach().cpu().numpy()

        size_x = np.array(self.doe_dxy * self.doe_size[0] / 2)
        size_y = np.array(self.doe_dxy * self.doe_size[1] / 2)
        unit_val, unit_axis = float_to_unit_identifier(max(size_x,size_y))
        size_x = size_x / unit_val
        size_y = size_y / unit_val

        extent = [-size_x, size_x, -size_y, size_y]

        if figsize is not None:
            fig = plt.figure(figsize=figsize)
            
        # First subplot: 2D plot
        plt.subplot(1, 1, 1)
        _im1 = plt.imshow(thickness, cmap=cmap, extent=extent)  # use colormap 'viridis'
        plt.title('2D Height Map of DOE')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlabel("Position (" + unit_axis + ")")
        plt.ylabel("Position (" + unit_axis + ")")

        # Show the plots
        add_colorbar(_im1)
        plt.tight_layout()
        plt.show()

    def look_up_table(self, look_up_table):
        """
        look-up-table with manufacturable height values

        look_up_table: 
            list of manufacturable height values
        if look_up_table is None:
            use height_constraint_max and doe_level to build a look_up_table
        """
        if look_up_table == None: 
            # linear height look-up table
            lut = torch.linspace(0, self.height_constraint_max, self.doe_level+1).to(self.device)
            self.lut = lut[:-1]
    
        if look_up_table != None:
            # non-linear or pre-defined height look-up table 
            self.lut = torch.tensor(look_up_table, dtype=torch.float32).to(self.device)
            self.doe_level = len(self.lut) 

    def build_weight_height_map(self):
        height, width = self.doe_size[0], self.doe_size[1]
        
        if self.num_unit is None:
            self.weight_height_map = nn.parameter.Parameter(
                torch.randn(height, width, device=self.device), requires_grad=True)
        else:
            unit_size = [int(height / self.num_unit), int(width / self.num_unit)]
            self.weight_height_map = nn.parameter.Parameter(
                torch.randn(unit_size[0], unit_size[0], device=self.device), requires_grad=True)

    def preprocessed_height_map(self, tau):
        height, width = self.doe_size[0], self.doe_size[1]

        self.height_map = self.height_constraint_max * torch.sigmoid(torch.clamp(self.weight_height_map, min=-8.0, max=8.0))
        # The PSQ function uses sigmoid functions to achieve a gradual transition between quantization levels.
        self.height_constraint_min = 0

        delta = (self.height_constraint_max - self.height_constraint_min) / (self.doe_level - 1)
        
        x_normalized = (self.height_map - self.height_constraint_min) / delta - 0.5
        levels_range = torch.arange(self.doe_level - 1, device=self.height_map.device).unsqueeze(0).unsqueeze(2)

        height_map = self.height_constraint_min + delta * torch.sum(
            torch.sigmoid(tau * (x_normalized.unsqueeze(1) - levels_range)),
            dim=1
        )
        height_map = height_map[None, None, :, :]
        if self.num_unit is None:
            self.height_map = height_map.squeeze(0,1).to(self.device)
        #self.height_map = torch.round(height_map, decimals=3).squeeze(0,1).to(self.device)
        else:
            self.height_map = _copy_quad_to_full(height_map).squeeze(0,1).to(self.device)
        #print(self.height_map.shape)
        return self.height_map
    
    def forward(self, field: ElectricField, iter_frac=None)->ElectricField:
        def tau_iter(iter_frac, tau_min=0.5, tau_max=50, r=None):
            # linearly increasing tenperature
            delta_tau = tau_max - tau_min
            tau = tau_min + delta_tau * iter_frac
            return tau
        
        if iter_frac != None:
            tau = tau_iter(iter_frac=iter_frac, 
                           tau_min=self.tau_min, 
                           tau_max=self.tau_max)
        elif iter_frac == None:
            tau = None

        return self.modulate(input_field=field, 
                             preprocessed_height_map=self.preprocessed_height_map(tau=tau),
                             height_tolerance=self.tolerance, 
                             epsilon=self.epsilon,
                             tand=self.tand)   


class STEQuantizationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lut):
        # Quantize the input based on the look-up table
        idx = torch.argmin(torch.abs(input.unsqueeze(-1) - lut), dim=-1)
        quantized = lut[idx]
        ctx.save_for_backward(input, lut)
        return quantized

    @staticmethod
    def backward(ctx, grad_output):
        #input, lut = ctx.saved_tensors
        grad_input = grad_output.clone()
        # Straight-Through Estimator: Pass gradients through as if the quantization is an identity function
        return grad_input, None

ste_quan = STEQuantizationFunction.apply

class STEQuantizedDOELayer(DOELayer):
    # Progressive Sigmoid Quantization (PSQ) method
    def __init__(self, 
                 doe_params   : dict, 
                 optim_params : dict, 
                 device       : torch.device = None):
        super(STEQuantizedDOELayer, self).__init__()

        if device is None:
            self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device     = device
        
        # get the design parameters of DOE
        self.doe_size = doe_params.get('doe_size', None)
        self.doe_dxy = doe_params.get('doe_dxy', None)
        self.doe_level = doe_params.get('doe_level', 6)

        self.num_unit = doe_params.get('num_unit', None)

        height_constraint_max       = doe_params.get('height_constraint_max', 2 * mm)
        self.height_constraint_max  = torch.tensor(height_constraint_max, device=self.device)
        self.tolerance = doe_params.get('tolerance', 0.05 * mm)  # assuming mm to meters

        material            = doe_params.get('material', None)
        self.material       = torch.tensor(material, device=self.device)
        self.epsilon        = self.material[0]   # relative permittivity of doe
        self.tand           = self.material[1]   # loss tangent of doe
        self.build_weight_height_map()
        look_up_table = doe_params.get('look_up_table', None)
        self.look_up_table(look_up_table)

    def visualize(self,
                  cmap                    ='viridis',
                  figsize                 = (4,4)):
        """  visualize the thickness of the doe
        """
        

        thickness = self.height_map.squeeze(0,1).detach().cpu().numpy()

        size_x = np.array(self.doe_dxy * self.doe_size[0] / 2)
        size_y = np.array(self.doe_dxy * self.doe_size[1] / 2)
        unit_val, unit_axis = float_to_unit_identifier(max(size_x,size_y))
        size_x = size_x / unit_val
        size_y = size_y / unit_val

        extent = [-size_x, size_x, -size_y, size_y]

        if figsize is not None:
            fig = plt.figure(figsize=figsize)
            
        # First subplot: 2D plot
        plt.subplot(1, 1, 1)
        _im1 = plt.imshow(thickness, cmap=cmap, extent=extent)  # use colormap 'viridis'
        plt.title('2D Height Map of DOE')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlabel("Position (" + unit_axis + ")")
        plt.ylabel("Position (" + unit_axis + ")")

        # Show the plots
        add_colorbar(_im1)
        plt.tight_layout()
        plt.show()

    def look_up_table(self, look_up_table):
        """
        look-up-table with manufacturable height values

        look_up_table: 
            list of manufacturable height values
        if look_up_table is None:
            use height_constraint_max and doe_level to build a look_up_table
        """
        if look_up_table == None: 
            # linear height look-up table
            lut = torch.linspace(0, self.height_constraint_max, self.doe_level+1).to(self.device)
            self.lut = lut[:-1]
    
        if look_up_table != None:
            # non-linear or pre-defined height look-up table 
            self.lut = torch.tensor(look_up_table, dtype=torch.float32).to(self.device)
            self.doe_level = len(self.lut) 

    def build_weight_height_map(self):
        height, width = self.doe_size[0], self.doe_size[1]
        
        if self.num_unit is None:
            self.weight_height_map = nn.parameter.Parameter(
                torch.randn(1, 1, height, width, device=self.device), requires_grad=True)
        else:
            unit_size = [int(height / self.num_unit), int(width / self.num_unit)]
            self.weight_height_map = nn.parameter.Parameter(
                torch.randn(1, 1, unit_size[0], unit_size[0], device=self.device), requires_grad=True)

    def preprocessed_height_map(self):
        
        height_map = self.height_constraint_max * torch.sigmoid(torch.clamp(self.weight_height_map, min=-8.0, max=8.0))
        if self.num_unit is None:
            height_map = height_map.squeeze(0,1).to(self.device)
            self.height_map = ste_quan(height_map, self.lut)
        else:
            height_map = _copy_quad_to_full(height_map).squeeze(0,1).to(self.device)
            self.height_map = ste_quan(height_map, self.lut)
        return self.height_map
    
    def forward(self, field: ElectricField, iter_frac=None)->ElectricField:
        
        return self.modulate(input_field=field, 
                             preprocessed_height_map=self.preprocessed_height_map(),
                             height_tolerance=self.tolerance, 
                             epsilon=self.epsilon,
                             tand=self.tand)   


class RotationallySymmetricFullPrecisionDOELayer(FullPrecisionDOELayer):

    def build_weight_height_map(self):
        height, width = self.doe_size[0], self.doe_size[1]
        self.height_map_shape = int(height * torch.sqrt(torch.tensor(2)) / 2)
        self.weight_height_map = nn.parameter.Parameter(
                -torch.pi + 2 * torch.pi * torch.rand(self.height_map_shape, device=self.device), 
                requires_grad=True
                )
    
    def preprocessed_height_map(self):
        height_map_1d = self.height_constraint_max * torch.sigmoid(torch.clamp(self.weight_height_map, min=-8.0, max=8.0))

        radius = self.height_map_shape
        diameter = 2 * radius
        x, y = torch.meshgrid(torch.arange(0, diameter // 2),
                              torch.arange(0, diameter // 2))
        radius_distance = torch.sqrt(x ** 2 + y ** 2).to(self.device)

        height_map_quad = torch.where((radius_distance < 1.0) & (radius_distance >= 0.0),
                                      height_map_1d[0], 0)
        
        for r in range(1, radius - 1):
            height_map_quad += torch.where((radius_distance < float(r + 1)) & (radius_distance >= float(r)),
                                           height_map_1d[r], 0)


        height_map_full = _copy_quad_to_full(height_map_quad)

        # Crop the height map from the center with dimensions [height, height]
        height_map = height_map_full.reshape(diameter, diameter)
        center_x, center_y = diameter // 2, diameter // 2
        start_x, start_y = center_x - self.doe_size[0] // 2, center_y - self.doe_size[1] // 2
        self.height_map = height_map[start_x:start_x + self.doe_size[0], start_y:start_y + self.doe_size[1]]

        return self.height_map


class RotationallySymmetricScoreGumbelSoftQuantizedDOELayer(SoftGumbelQuantizedDOELayerv3):
    def build_init_phase(self):
        height, width = self.doe_size[0], self.doe_size[1]
        self.height_map_shape = int(height * torch.sqrt(torch.tensor(2)) / 2)
        self.weight_init_phase = nn.parameter.Parameter(
                torch.randn(1, 1, 1, self.height_map_shape, device=self.device), requires_grad=True
            )

    def preprocessed_height_map(self, wavelengths, tau, iter_frac=None):
        height, width = self.doe_size[0], self.doe_size[1]

        height_map_1d = self.height_constraint_max * torch.sigmoid(torch.clamp(self.weight_init_phase, min=-10.0, max=10.0))

        if iter_frac > 0.3 and iter_frac <= 0.8:
            beta = (iter_frac - 0.3) / (0.8 - 0.3)
            # convert look_up_table to phase lut with designed wavelength
            phase_lut = _height_to_phase_with_material_refractive_idx(self.lut, wavelengths.min(), torch.sqrt(self.epsilon))
            quantized_phase = _height_to_phase_with_material_refractive_idx(height_map_1d, wavelengths.min(), torch.sqrt(self.epsilon))
            scores = self.score_phase(quantized_phase, phase_lut.to(quantized_phase.device), (self.tau_max / tau) **1) *  self.c_s * (self.tau_max / tau) **1
            one_hot = F.gumbel_softmax(scores, tau=tau, hard=True, dim=1)
            q_height_map_1d = (self.lut.reshape(1, len(self.lut), 1, 1) * one_hot).sum(1, keepdim=True)
            height_map_1d = (1 - iter_frac) * height_map_1d + iter_frac * q_height_map_1d
            #print(height_map_1d.shape)
        
        if iter_frac > 0.8:
            phase_lut = _height_to_phase_with_material_refractive_idx(self.lut, wavelengths.min(), torch.sqrt(self.epsilon))
            quantized_phase = _height_to_phase_with_material_refractive_idx(height_map_1d, wavelengths.min(), torch.sqrt(self.epsilon))
            scores = self.score_phase(quantized_phase, phase_lut.to(quantized_phase.device), (self.tau_max / tau) **1) *  self.c_s * (self.tau_max / tau) **1
            one_hot = F.gumbel_softmax(scores, tau=tau, hard=True, dim=1)
            height_map_1d = (self.lut.reshape(1, len(self.lut), 1, 1) * one_hot).sum(1, keepdim=True)

        height_map_1d = height_map_1d.squeeze(0,1,2)
        #print(height_map_1d.shape)
        radius = self.height_map_shape
        diameter = 2 * radius
        x, y = torch.meshgrid(torch.arange(0, diameter // 2),
                              torch.arange(0, diameter // 2))
        radius_distance = torch.sqrt(x ** 2 + y ** 2).to(self.device)

        height_map_quad = torch.where((radius_distance < 1.0) & (radius_distance >= 0.0),
                                      height_map_1d[0], 0)
        #print(radius)
        for r in range(1, radius - 1):
            height_map_quad += torch.where((radius_distance < float(r + 1)) & (radius_distance >= float(r)),
                                           height_map_1d[r], 0)

        height_map_full = _copy_quad_to_full(height_map_quad)

        # Crop the height map from the center with dimensions [height, height]
        height_map = height_map_full.reshape(diameter, diameter)
        center_x, center_y = diameter // 2, diameter // 2
        start_x, start_y = center_x - self.doe_size[0] // 2, center_y - self.doe_size[1] // 2
        self.height_map = height_map[start_x:start_x + self.doe_size[0], start_y:start_y + self.doe_size[1]]

        return self.height_map


class RotationallySymmetricSTEQuantizedDOELayer(STEQuantizedDOELayer):

    def build_weight_height_map(self):
        height, width = self.doe_size[0], self.doe_size[1]
        self.height_map_shape = int(height * torch.sqrt(torch.tensor(2)) / 2)
        self.weight_height_map = nn.parameter.Parameter(
                -torch.pi + 2 * torch.pi * torch.rand(1, self.height_map_shape, device=self.device), 
                requires_grad=True
                )
    
    def preprocessed_height_map(self):
        height_map_1d = self.height_constraint_max * torch.sigmoid(torch.clamp(self.weight_height_map, min=-8.0, max=8.0))
        height_map_1d = ste_quan(height_map_1d, self.lut).squeeze(0)

        radius = self.height_map_shape
        diameter = 2 * radius
        x, y = torch.meshgrid(torch.arange(0, diameter // 2),
                              torch.arange(0, diameter // 2))
        radius_distance = torch.sqrt(x ** 2 + y ** 2).to(self.device)

        height_map_quad = torch.where((radius_distance < 1.0) & (radius_distance >= 0.0),
                                      height_map_1d[0], 0)
        
        for r in range(1, radius - 1):
            height_map_quad += torch.where((radius_distance < float(r + 1)) & (radius_distance >= float(r)),
                                           height_map_1d[r], 0)


        height_map_full = _copy_quad_to_full(height_map_quad)

        # Crop the height map from the center with dimensions [height, height]
        height_map = height_map_full.reshape(diameter, diameter)
        center_x, center_y = diameter // 2, diameter // 2
        start_x, start_y = center_x - self.doe_size[0] // 2, center_y - self.doe_size[1] // 2
        self.height_map = height_map[start_x:start_x + self.doe_size[0], start_y:start_y + self.doe_size[1]]

        return self.height_map


class RotationallySymmetricNaiveGumbelQuantizedDOELayer(NaiveGumbelQuantizedDOELayer):

    def build_init_logits(self):
        height, width = self.doe_size[0], self.doe_size[1]
        self.height_map_shape = int(height * torch.sqrt(torch.tensor(2)) / 2)
        self.weight_height_map = nn.parameter.Parameter(
                torch.rand(1, self.height_map_shape, self.doe_level, device=self.device), requires_grad=True
                )
    
    def preprocessed_height_map(self, tau):

        height, width = self.doe_size[0], self.doe_size[1]
        # Sample soft categorical using reparameterization trick
        if tau != None:
            sample_one_hot = F.gumbel_softmax(self.weight_height_map, tau=tau, hard=True)
        elif tau == None:
            sample_one_hot = F.gumbel_softmax(self.weight_height_map, tau=1, hard=True)

        height_map_1d = (self.lut[None, None, :] * sample_one_hot).sum(dim=-1).squeeze(0)
        #print(height_map_1d.shape)
        radius = self.height_map_shape
        diameter = 2 * radius
        x, y = torch.meshgrid(torch.arange(0, diameter // 2),
                              torch.arange(0, diameter // 2))
        radius_distance = torch.sqrt(x ** 2 + y ** 2).to(self.device)
        height_map_quad = torch.where((radius_distance < 1.0) & (radius_distance >= 0.0),
                                      height_map_1d[0], 0)
        
        for r in range(1, radius - 1):
            height_map_quad += torch.where((radius_distance < float(r + 1)) & (radius_distance >= float(r)),
                                           height_map_1d[r], 0)

        height_map_full = _copy_quad_to_full(height_map_quad)

        # Crop the height map from the center with dimensions [height, height]
        height_map = height_map_full.reshape(diameter, diameter)
        center_x, center_y = diameter // 2, diameter // 2
        start_x, start_y = center_x - self.doe_size[0] // 2, center_y - self.doe_size[1] // 2
        self.height_map = height_map[start_x:start_x + self.doe_size[0], start_y:start_y + self.doe_size[1]]

        return self.height_map


class RotationallySymmetricPSQuantizedQuantizedDOELayer(PSQuantizedDOELayer):

    def build_weight_height_map(self):
        height, width = self.doe_size[0], self.doe_size[1]
        self.height_map_shape = int(height * torch.sqrt(torch.tensor(2)) / 2)
        self.weight_height_map = nn.parameter.Parameter(
                torch.rand(1, self.height_map_shape, device=self.device), requires_grad=True
                )
    
    def preprocessed_height_map(self, tau):

        height, width = self.doe_size[0], self.doe_size[1]
        self.height_map = self.height_constraint_max * torch.sigmoid(torch.clamp(self.weight_height_map, min=-8.0, max=8.0))
        # The PSQ function uses sigmoid functions to achieve a gradual transition between quantization levels.
        self.height_constraint_min = 0

        delta = (self.height_constraint_max - self.height_constraint_min) / (self.doe_level - 1)
        
        x_normalized = (self.height_map - self.height_constraint_min) / delta - 0.5
        levels_range = torch.arange(self.doe_level - 1, device=self.height_map.device).unsqueeze(0).unsqueeze(2)

        height_map_1d = self.height_constraint_min + delta * torch.sum(
            torch.sigmoid(tau * (x_normalized.unsqueeze(1) - levels_range)),
            dim=1
        ).squeeze(0)
        #print(height_map_1d.shape)

        radius = self.height_map_shape
        diameter = 2 * radius
        x, y = torch.meshgrid(torch.arange(0, diameter // 2),
                              torch.arange(0, diameter // 2))
        radius_distance = torch.sqrt(x ** 2 + y ** 2).to(self.device)
        height_map_quad = torch.where((radius_distance < 1.0) & (radius_distance >= 0.0),
                                      height_map_1d[0], 0)
        
        for r in range(1, radius - 1):
            height_map_quad += torch.where((radius_distance < float(r + 1)) & (radius_distance >= float(r)),
                                           height_map_1d[r], 0)

        height_map_full = _copy_quad_to_full(height_map_quad)

        # Crop the height map from the center with dimensions [height, height]
        height_map = height_map_full.reshape(diameter, diameter)
        center_x, center_y = diameter // 2, diameter // 2
        start_x, start_y = center_x - self.doe_size[0] // 2, center_y - self.doe_size[1] // 2
        self.height_map = height_map[start_x:start_x + self.doe_size[0], start_y:start_y + self.doe_size[1]]

        return self.height_map