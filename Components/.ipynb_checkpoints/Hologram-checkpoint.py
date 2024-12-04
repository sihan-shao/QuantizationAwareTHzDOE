import sys

from utils.units import mm
sys.path.append('../')
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
        loss = torch.exp(-0.5 * wave_numbers * (height_map + BASE_PLANE_THICKNESS) * tand * torch.sqrt(epsilon))
        phase_delay = torch.exp(-1j * wave_numbers * (height_map + BASE_PLANE_THICKNESS) * (torch.sqrt(epsilon) - 1))
        
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
                 tolerance      : float = 0.1*mm,
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


class GumbelQuantizedHologramLayer(HologramLayer):
    def __init__(self, 
                 holo_size      : list = None,
                 holo_level     : int = 6,
                 look_up_table  : list = None,
                 num_unit       : int = None,
                 height_constraint_max: float = None,
                 tolerance      : float = 0.1*mm,
                 material       : list = None,
                 circ_aperture  : bool = True,
                 device         : torch.device = None) -> None:
        super(GumbelQuantizedHologramLayer, self).__init__()
        
        if device is None:
            self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device     = device
            
        self.holo_size = holo_size
        self.holo_level = holo_level
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
        self.build_weight_height_map()
        
        
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
            lut = torch.linspace(0, self.height_constraint_max, self.holo_level+1).to(self.device)
            self.lut = lut[:-1]
    
        if look_up_table != None:
            # non-linear or pre-defined look-up table 
            self.lut = torch.tensor(look_up_table, dtype=torch.float32).to(self.device)
            self.holo_level = len(self.lut)
            self.height_constraint_max = torch.max(self.lut)


    def build_weight_height_map(self):
        height, width = self.holo_size[0], self.holo_size[1]
        
        if self.num_unit is None:
            self.weight_height_map = nn.parameter.Parameter(
                torch.rand(height, width, self.holo_level, device=self.device), requires_grad=True)
        else:
            unit_size = [int(height / self.num_unit), int(width / self.num_unit)]
            self.weight_height_map = nn.parameter.Parameter(
                torch.rand(unit_size[0], unit_size[1], self.holo_level, device=self.device), requires_grad=True
            )


    def preprocessed_height_map(self, tau):
        height, width = self.holo_size[0], self.holo_size[1]
        # Sample soft categorical using reparameterization trick
        
        if tau != None:
            sample_one_hot = F.gumbel_softmax(self.weight_height_map, tau=tau, hard=True)
        elif tau == None:
            sample_one_hot = F.gumbel_softmax(self.weight_height_map, tau=1, hard=True)
        
        #level_logits = torch.arange(0, self.holo_level).to(self.device)

        self.height_map = (sample_one_hot * self.lut[None, None, :]).sum(dim=-1)
        #print(holo_sample.shape)
        #quantized_value = self.height_constraint_max / (self.holo_level - 1)
        
        #self.height_map = holo_sample * quantized_value
        
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
        #self.height_map = self.build_height_map()
        if iter_frac != None:
            tau = tau_iter(iter_frac=iter_frac)
        elif iter_frac == None:
            tau = None
        #print(tau)
        return self.modulate(input_field=field, 
                             preprocessed_height_map=self.preprocessed_height_map(tau=tau),
                             height_tolerance=self.tolerance, 
                             epsilon=self.epsilon,
                             tand=self.tand, 
                             circ_aperture=self.circ_aperture)           
    

# implement Spherical Quantized Flat DOELens 


# implement Spectral Splitter Hologram

class SpectralSplitterHologram(GumbelQuantizedHologramLayer):
    
    def build_weight_height_map(self):
        height, width = self.holo_size[0], self.holo_size[1]
        
        self.weight_height_map = nn.parameter.Parameter(
            torch.rand(1, width, self.holo_level, device=self.device), requires_grad=True) 
        
    def preprocessed_height_map(self, tau):
        height, width = self.holo_size[0], self.holo_size[1]
        # Sample soft categorical using reparameterization trick
        sample_one_hot = F.gumbel_softmax(self.weight_height_map, tau=tau, hard=True) 
        level_logits = torch.arange(0, self.holo_level).to(self.device)
        
        holo_sample = (sample_one_hot *
                       level_logits[None, None, :]).sum(dim=-1)
        
        quantized_value = self.height_constraint_max / (self.holo_level - 1)
        
        height_map_1D = holo_sample * quantized_value

        self.height_map = height_map_1D.repeat(1, 1, height, 1)
        
        self.height_map = self.height_map.squeeze(0,1).to(self.device)
        
        return self.height_map
    
    @staticmethod
    def define_FoM_metric(field : ElectricField, 
                          wavelength: float,
                          focal_length  : float, 
                          position: list, 
                          show_FoM = False) -> torch.Tensor:
        
        """
        Full-width-at-half-maximum (FWHM) by the far-field diffraction limit:
                W_i = \lambda_i / 2NA, 
        where NA is the numerical 
        
        Numerical aperture (NA): 
                NA = sin[tan^{-1}(L_X / (2 * f))]
        where L_X is the physics length of the design diffractive element, f is focal length
        
        The target function is a line point spread function (PSF) as a gaussian function:
                T_i(x') = exp{-[x' +- ((x'_{real_position} + x'_{max}) / 2 )^2] / [(W_i / 2)^2]}
        
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        height, width = field.height, field.width
        # extract dx, dy spacing
        dx = torch.tensor(field.spacing[0], device=device)
        dy = torch.tensor(field.spacing[1], device=device)
        
        focal_length = torch.tensor(focal_length, device=device)
        Length_x, Length_y = height * dx, width * dy
        effective_L = torch.sqrt(Length_x ** 2 + Length_y ** 2)
        effective_NA = torch.sin(torch.atan(effective_L / (2 * focal_length)))
        
        FWHM = wavelength / (2 * effective_NA)
        
        # Implementing the Gaussian function for PSF
        x_grid = torch.linspace(-Length_x / 2, Length_x / 2, steps=height, device=device)
        
        x_position = torch.tensor(position)
        
        is_greater_than_all = (x_position.abs() > (Length_x/2)).all().item()
        if is_greater_than_all:
            raise ValueError('The Target PSF should between the range of [{:.4f}m, {:.4f}m]'.format(-Length_x.item() / 2, Length_x.item() / 2))
            
        FoM = torch.exp(-((x_grid - x_position) ** 2) / ((FWHM / 2) ** 2))
        
        FoM = FoM.repeat(height, 1)
        
        if show_FoM == True:
            dx = dx.detach().cpu()
            dy = dy.detach().cpu()
            size_x = np.array(dx / 2.0 * height)
            size_y =np.array(dy / 2.0 * width)
            unit_val, unit_axis = float_to_unit_identifier(max(size_x,size_y))
            size_x = size_x / unit_val
            size_y = size_y / unit_val
            extent = [-size_y, size_y, -size_x, size_x]
            
            unit_val, unit = float_to_unit_identifier(wavelength)
            wavelength = wavelength / unit_val
            plt.subplot(1, 1, 1)
            _im1 = plt.imshow(FoM.detach().cpu(), extent=extent, vmax=None, vmin=None)
            plt.title("Target PSF| wavelength = " + str(round(wavelength, 2)) + str(unit))
            plt.xlabel("Position (" + unit_axis + ")")
            plt.ylabel("Position (" + unit_axis + ")")
            add_colorbar(_im1)
            plt.tight_layout()
        
        if show_FoM == False:
            return FoM[None, None, :, :].to(device)


class old_HologramElement(nn.Module):

    """
    Hologram is an abstract class that acts as an holographic element that can interact
    with a complex wavefront.

    This Hologram class is wavelength dependent, i.e. a multi-channel tensor can be used as input
    to calculate wavelength dependent output (e.g. if the phase-delays are different for
    different wavelengths)

    
    """
    
    def __init__(self,
                thickness      : torch.Tensor,
                material       : list,
                fixed_pattern  : bool = True,
                scale_thickness: int = 1,
                circ_aperture  : bool = True,
                device         : torch.device = None
                ):
        """Initializes the Hologram class

        Args:
            dx (floatortorch.Tensor): input feature size
            thickness (torch.Tensor): the thickness of the hologram at each pixel which will e.g. define phase-delay
            material (list with two number): A hologram material parameters including relative permittivity and loss tangent
            device (_device, optional): [description]. Defaults to torch.device("cpu").
            fixed_pattern (bool, optional): If True the phase delay will not be set to an nn.parameter to be optimized for . Defaults to True.
            scale_phase (bool, optional): factor to scale phase by before applying phase to input field
            dtype (dtype, optional): [description]. Defaults to torch.double.
        """ 
    
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #Set internal variables
        self.fixed_pattern       = fixed_pattern # Flag (boolean) if thickness is optimized
        self.material            = torch.Tensor(material).to(self.device)
        self.epsilon             = self.material[0] # relative permittivity of hologram
        self.tand                = self.material[1]   # loss tangent of hologram
        self.scale_thickness     = scale_thickness
        self.circ_aperture       = circ_aperture
        self.thickness           = thickness.to(self.device)
        
    @property
    def thickness(self) -> torch.Tensor:
        try:
            return self._thickness
        except AttributeError:
            return None
    
    @thickness.setter
    def thickness(self,
                  thickness : torch.Tensor
                  ) -> None:
        """ Add thickness parameter to buffer
        
        If it is parameter make it a paramter, otherwise just add to buffer/statedict

        Args:
            thickness (torch.Tensor): [description]
        """
        # The None is handling is just a check which is needed to change flags
        if thickness is None:
            thickness = self.thickness
            if thickness is None:
                return
        
        if self.thickness is not None:
            del self._thickness
        
        thickness = thickness.to(self.device)

        if self.fixed_pattern == True:
            self.register_buffer("_thickness", thickness)
        elif self.fixed_pattern == False:
            self.register_parameter("_thickness", torch.nn.Parameter(thickness))
        
    
    @property
    def fixed_pattern(self) ->bool:
        return self._fixed_pattern
    
    @fixed_pattern.setter
    def fixed_pattern(self, fixed_pattern) -> None:
        self._fixed_pattern = fixed_pattern
        self.thickness = None    
    
    def visualize(self,
                cmap                    ='viridis',
                figsize                 = (4,4)):
        """  visualize the thickness of the hologram
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        thickness = self.thickness.detach().cpu().numpy()
        
        if figsize is not None:
            fig = plt.figure(figsize=figsize)
            
        # First subplot: 2D plot
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.imshow(thickness, cmap=cmap)  # use colormap 'viridis'
        ax1.set_title('2D Height Map of Hologram')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')

        # Show the plots
        plt.tight_layout()
        plt.show()

        
    def calc_phase_shift(self, wavelengths : torch.Tensor) -> torch.Tensor:
        """Helper method to write smaller code outside of this class.
        """
        
        thickness   = self.thickness   # torch.tensor shape [H, W]
        epsilon     = self.epsilon     # torch.tensor one number
        tand        = self.tand        # torch.tensor one number

        thickness = thickness[None, :, :]
        
        # ensure wavelengths has one dimension, resulting shape: [C]
        wavelengths = wavelengths.view(-1)

        # add dimensions to wavelengths for broadcasting, resulting shape: [C, 1, 1]
        wavelengths = wavelengths[:, None, None]

        # calculate loss and phase delay
        loss = torch.exp(-0.5 * (2 * torch.pi / wavelengths) * thickness * tand * torch.sqrt(epsilon))
        phase_delay = torch.exp(-1j * (2 * torch.pi / wavelengths) * thickness * (torch.sqrt(epsilon) - 1))

        # The hologram is not a thin element, we need to consider the air thickness 
        air_phase = torch.exp(-1j * (2 * torch.pi / wavelengths) * torch.max(thickness))
        
        # calculate final phase shift combined with loss
        phase_shift = loss * phase_delay * air_phase

        return phase_shift.to(self.device)
    
    def forward(self,
            field : ElectricField,
                ) -> ElectricField:
        """  Takes in a field and applies the hologram to it


        Args:
            field (ElectricField): [description]

        Returns:
            ElectricField: [description]
        """
        
        wavelengths = field.wavelengths

        assert wavelengths.device == field.data.device, "WAVELENGTHS: " + str(wavelengths.device) + ", FIELD: " + str(field.data.device)

        phase_shift = self.calc_phase_shift(wavelengths=wavelengths)
        
        if self.circ_aperture:
            dx, dy = field.spacing[0], field.spacing[1]
            height, width = field.height, field.width
            
            r = max([dx * height, dy * width]) / 2.0
            x = torch.linspace(-dx * height / 2, dx * height / 2, height, device=self.device)
        
            y = torch.linspace(-dy * width / 2, dy * width / 2, width, device=self.device)
            
            X, Y = torch.meshgrid(x, y)
        
            R = torch.sqrt(X**2 + Y**2)

            # Create a mask that is 1 inside the circle and 0 outside
            Mask = torch.where(R <= r, 1, 0)
            #print(Mask.is_cuda, field.data.is_cuda, phase_shift.is_cuda)

            out_field = Mask[None, None, :, :] * field.data * phase_shift[None, :, :, :]   
            # For phase visualization, the phase outside the circular are either 0 or pi
            # This is because after Masking, there are negative real zero (-0 + 0j) and positive real zero (0+0j)
            # the out_field outside the circular, which will result in the angle becomes pi and zero
            # but the amplitude does not be effect and the propgation precision will not be effect since the peridoc of phase
            
        else: 
            out_field = field.data * phase_shift[None, :, :, :]     

        E_out = ElectricField(
            data = out_field,
            wavelengths=field.wavelengths,
            spacing = field.spacing
        )

        return E_out 