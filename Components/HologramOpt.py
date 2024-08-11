import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pathlib
import matplotlib
import matplotlib.pyplot as plt
from copy import copy
from utils.units import *
from utils.Visualization_Helper import float_to_unit_identifier, add_colorbar
from DataType.ElectricField import ElectricField
import torch.nn.functional as F
from utils.Helper_Functions import UniformNoise


##############################
# Helper functions
##############################


#def get_zernike_volume(resolution, n_terms, scale_factor=1e-6):
#    zernike_volume = poppy.zernike.zernike_basis(nterms=n_terms, npix=resolution, outside=0.0)
#    return zernike_volume * scale_factor


BASE_PLANE_THICKNESS = 2 * 1e-3

"""
For promoting smoothness of hologram, we use the Laplacian regularization term, which effectively penalizes rapid changes in the hologram thickness.

An L1 norm of the Laplacian will tend to make the surface piecewise constant (encouraging flat regions separated by sharp edges), 
while an L2 norm will tend to make the surface smooth. 
"""
def laplacian_filter_torch(img_batch):
    """Laplacian filter. Also considers diagonals."""
    laplacian_filter = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=torch.float32)
    laplacian_filter = laplacian_filter.view(1, 1, 3, 3)
    
    if len(img_batch.shape) == 3:  # if input is a 3D tensor, add an extra dimension for channels
        img_batch = img_batch.unsqueeze(1)
        
    filter_input = img_batch.float()
    filtered_batch = F.conv2d(filter_input, laplacian_filter, padding=1)
    return filtered_batch

def laplace_l1_regularizer(a_tensor, scale=1):

    laplace_filtered = laplacian_filter_torch(a_tensor)
    laplace_filtered = laplace_filtered[:, :, 1:-1, 1:-1]
    return scale * torch.mean(torch.abs(laplace_filtered))

def laplace_l2_regularizer(a_tensor, scale=1):

    laplace_filtered = laplacian_filter_torch(a_tensor)
    laplace_filtered = laplace_filtered[:, :, 1:-1, 1:-1]
    return scale * torch.mean(torch.square(laplace_filtered))


class HologramElementOpt(nn.Module):
    """
    Hologram is an abstract class that acts as an holographic element that can interact
    with a complex wavefront.

    This Hologram class is wavelength dependent, i.e. a multi-channel tensor can be used as input
    to calculate wavelength dependent output (e.g. if the phase-delays are different for
    different wavelengths)
    
    In this class, we implemement an Optimizable/Differentiable design for discrete multi-level hologram 
    by using gumbel-softmax trick 
     
    """
    
    def __init__(self,
                trainable      : bool = True,
                thickness      : torch.Tensor = None,
                holo_size      : list = None,
                holo_level     : int = 6,
                tolerance      : float = 1e-4,
                material       : list = None,
                circ_aperture  : bool = True,
                device         : torch.device = None
                ):
        """Initializes the Hologram class

        Args:
            dx (floatortorch.Tensor): input feature size
            thickness (torch.Tensor): the thickness of the hologram at each pixel which will e.g. define phase-delay
            material (list with two number): A hologram material parameters including relative permittivity and loss tangent
            tolerance (float): Simulated fabrication noise on hologram thickness. Defaults: 0.1 mm
            device (_device, optional): [description]. Defaults to torch.device("cpu").
            fixed_pattern (bool, optional): If True the phase delay will not be set to an nn.parameter to be optimized for . Defaults to True.
            scale_phase (bool, optional): factor to scale phase by before applying phase to input field
        """ 
    
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #Set internal variables
        self.trainable               = trainable # Flag (boolean) if thickness is optimized
        if self.trainable:      #  If it is optimizable, then make it a trainable paramter
            self.logits = nn.parameter.Parameter(
            torch.rand(holo_size[0], holo_size[1], holo_level))
            self.holo_level          = torch.tensor(holo_level, device=self.device)
            self.level_logits        = torch.arange(0, self.holo_level).to(device)
            self.Max_thickness       = None 
            self.thickness           = None
            
        else:                   #  Otherwise just add to buffer/statedict
            self.thickness           = torch.tensor(thickness, device=device, dtype=torch.float32)
            self.register_buffer("_thickness", self.thickness)
        
        self.material                = torch.tensor(material, device=self.device)
        self.epsilon                 = self.material[0] # relative permittivity of hologram
        self.tand                    = self.material[1]   # loss tangent of hologram
        
        # Simulated fabrication noise on hologram thickness
        # add fabrication noise in hologram can improve the robustness in training process
        # I don't add noise when hologram is not trainable for accurate simulation result
        self.add_noise               = UniformNoise(a=tolerance)
        
        # whether add a circ aperture after the hologram
        self.circ_aperture           = circ_aperture
        self.Mask                    = None
        
        
    def get_holo_sample(self):
        # Sample soft categorical using reparameterization trick

        #print("Trainable parameters:{}".format(self.logits))
        sample_one_hot = F.gumbel_softmax(self.logits, tau=1, hard=True) 
        #print("sample_one_hot:{}".format(sample_one_hot))
        holo_sample = (sample_one_hot *
                       self.level_logits[None, None, :]).sum(dim=-1)
        
        return holo_sample.to(self.device)
    
    def holo_level_to_thickness(self, holo_instance, max_thickness, step_height=1):
        """
        Due to perdoic properity of phase, we firstly need to calculate 
        the maximal thickness corresponding to a whole 2pi period 
        
        phase shift: e^{-jk * thickness * (n-1)} = e^{-j * 2pi} 
                ==>  k * thickness * (sqrt(epsilon)-1) = 2pi
                ==>  maximal thickness = wavlength_max / (sqrt(epsilon)-1)
        """
        thickness_step = max_thickness / self.holo_level
        holo_thickness = holo_instance * thickness_step
        
        if step_height != None:
            holo_thickness = holo_thickness + max_thickness * torch.round(holo_instance / step_height)

        return holo_thickness
    
    def surface_constrain(self, scale, type = 'L1'):
        
        if type == 'L1': 
            constrain = laplace_l1_regularizer(self.thickness.unsqueeze(0), scale)
        elif type == 'L2':
            constrain = laplace_l2_regularizer(self.thickness.unsqueeze(0), scale)
            
        return constrain
    
    
    def visualize(self,
                cmap                    ='viridis',
                figsize                 = (4,4)):
        """  visualize the thickness of the hologram
        """
        
        thickness = self.thickness.detach().cpu().numpy() * self.Mask.detach().cpu().numpy()
        
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

        
    def calc_phase_shift(self, wavelengths : torch.Tensor) -> torch.Tensor:
        """Helper method to write smaller code outside of this class.
        """
        
         # add fabrication noise on hologram thickness
        thickness   = self.add_noise(self.thickness)   # torch.tensor shape [H, W]
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

        if self.trainable:
            holo_sample = self.get_holo_sample()
            """
            Due to perdoic properity of phase, we firstly need to calculate 
            the maximal thickness corresponding to a whole 2pi period 
        
            phase shift: e^{-jk * thickness * (n-1)} = e^{-j * 2pi} 
                    ==>  k * thickness * (sqrt(epsilon)-1) = 2pi
                    ==>  maximal thickness = wavlength_max / (sqrt(epsilon)-1)
            """
            self.Max_thickness = torch.max(wavelengths) / (torch.sqrt(self.epsilon) - 1)
            
            self.thickness = self.holo_level_to_thickness(holo_instance=holo_sample, 
                                                          max_thickness=self.Max_thickness)
            
            phase_shift = self.calc_phase_shift(wavelengths=wavelengths)
        
        else:
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
            self.Mask = torch.where(R <= r, 1, 0)
            #print(Mask.is_cuda, field.data.is_cuda, phase_shift.is_cuda)

            out_field = self.Mask[None, None, :, :] * field.data * phase_shift[None, :, :, :]   
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
    

"""
class ZernikeHologramOpt(HologramElementOpt):

    #ZernikeHologramOpt is an abstract class which the thcikness is parameterized using Zernike polynomials basis representation.
     

    
    def __init__(self,
                trainable      : bool = True,
                thickness      : torch.Tensor = None,
                holo_size      : list = None,
                holo_level     : int = 6,
                tolerance      : float = 1e-4,
                material       : list = None,
                circ_aperture  : bool = True,
                device         : torch.device = None
                ):

        Initializes the Hologram class

        Args:
            dx (floatortorch.Tensor): input feature size
            thickness (torch.Tensor): the thickness of the hologram at each pixel which will e.g. define phase-delay
            holo_size (list with two number): The resolution of hologram
            holo_level (int): In ZernikeHologramOpt class, it represents the number of coefficiences of Zernike polynomial
            material (list with two number): A hologram material parameters including relative permittivity and loss tangent
            tolerance (float): Simulated fabrication noise on hologram thickness. Defaults: 0.1 mm
            device (_device, optional): [description]. Defaults to torch.device("cpu").
            fixed_pattern (bool, optional): If True the phase delay will not be set to an nn.parameter to be optimized for . Defaults to True.

    
        super().__init__()
        
    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #Set internal variables
        self.trainable               = trainable # Flag (boolean) if thickness is optimized
        
        
        if self.trainable:      #  If it is optimizable, then make it a trainable paramter
            self.logits = nn.parameter.Parameter(
            torch.zeros(holo_level, 1, 1))    # the coefficients of Zernike polynomial
            self.holo_level          = torch.tensor(holo_level, device=self.device)
            if not os.path.exists('zernike_volume_%d.npy'%holo_size):
                #zernike_volume = poppy.
            self.Max_thickness       = None 
            self.thickness           = None
            
        else:                   #  Otherwise just add to buffer/statedict
            self.thickness           = thickness.to(self.device)
            self.register_buffer("_thickness", self.thickness)
            
            
        pass
"""

def _copy_quad_to_full(quad_map):
    height_map_half_left = torch.cat([torch.flip(quad_map, dims=[0]), quad_map], dim=0)
    height_map_full = torch.cat([torch.flip(height_map_half_left, dims=[1]), height_map_half_left], dim=1)
    return height_map_full


class Rank_x_Hologram(nn.Module):
    def __init__(self,
                holo_unit      : torch.Tensor = 1,
                holo_size      : list = None,
                holo_rank      : int = 6,
                tolerance      : float = 1e-4,
                material       : list = None,
                circ_aperture  : bool = False,
                device         : torch.device = None
                ):
        
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.material                = torch.tensor(material, device=self.device)
        self.epsilon                 = self.material[0] # relative permittivity of hologram
        self.tand                    = self.material[1]   # loss tangent of hologram
        
        self.holo_unit               = int(holo_unit)
        self.height                  = int(holo_size[0] / (self.holo_unit * 2))
        self.width                   = int(holo_size[1] / (self.holo_unit * 2))
        self.holo_rank               = int(holo_rank)
        
        self.add_noise               = UniformNoise(a=tolerance)
        
        # whether add a circ aperture after the hologram
        self.circ_aperture           = circ_aperture
        
        self.unit_thickness          = None
        self.thickness               = None
        self.Mask                    = None
        
    def get_unit_holo_thickness(self):      
        
        column_init_value   = nn.parameter.Parameter(
                torch.ones([self.holo_rank, self.width], dtype=torch.float32) * 1e-2
        )
        
        row_init_value      = nn.parameter.Parameter(
                torch.ones([self.height, self.holo_rank], dtype=torch.float32) * 1e-2
        )
        
        quad_thickness = torch.mm(row_init_value, column_init_value)
        quad_thickness = 5 * mm * torch.sigmoid(quad_thickness)  # clip to [0, 5 mm]
        
        unit_thickness = _copy_quad_to_full(quad_thickness)
        
        return unit_thickness
    
    def surface_constrain(self, scale, type = 'L1'):
        
        if type == 'L1': 
            constrain = laplace_l1_regularizer(self.unit_thickness.unsqueeze(0), scale)
        elif type == 'L2':
            constrain = laplace_l2_regularizer(self.unit_thickness.unsqueeze(0), scale)
            
        return constrain
    
    
    def visualize(self,
                cmap                    ='viridis',
                figsize                 = (4,4)):
        """  visualize the thickness of the hologram
        """
        
        unit_thickness = self.unit_thickness.detach().cpu().numpy() # * self.Mask.detach().cpu().numpy()
        
        if figsize is not None:
            fig = plt.figure(figsize=figsize)
            
        # First subplot: 2D plot
        plt.subplot(1, 1, 1)
        _im1 = plt.imshow(unit_thickness, cmap=cmap)  # use colormap 'viridis'
        plt.title('2D Height Map of Hologram')
        plt.xlabel('X')
        plt.ylabel('Y')
        
        # Show the plots
        add_colorbar(_im1)
        plt.tight_layout()
        plt.show()
        
    def calc_phase_shift(self, wavelengths : torch.Tensor) -> torch.Tensor:
        """Helper method to write smaller code outside of this class.
        """
        
        # get the full height map 
        self.thickness = self.unit_thickness.repeat(self.holo_unit, self.holo_unit)
        
        # add fabrication noise on hologram thickness
        thickness   = self.add_noise(self.thickness)   # torch.tensor shape [H, W]
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

        self.unit_thickness = self.get_unit_holo_thickness()
        
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
            self.Mask = torch.where(R <= r, 1, 0)
            #print(Mask.is_cuda, field.data.is_cuda, phase_shift.is_cuda)

            out_field = self.Mask[None, None, :, :] * field.data * phase_shift[None, :, :, :]   
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
    
        
        
        
        
    
    