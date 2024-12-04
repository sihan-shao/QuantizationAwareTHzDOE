import sys
sys.path.append('../')
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import pad
from DataType.ElectricField import ElectricField
from torch.fft import fft2, ifft2
from utils.units import *

class Fraunhofer_Prop(nn.Module):
    
    def __init__(self, 
                 z_distance         : float = 0.0, 
                 device             : str = None):
        super().__init__()
        self._z = z_distance
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pass

    @property
    def z(self) -> torch.Tensor:
        return self._z
    
    @z.setter
    def z(self, z : torch.Tensor) -> None:
        if not isinstance(z, torch.Tensor):
            value = torch.tensor(z, device=self.device)
        elif z.device != self.device:
            z = z.to(self.device)
        self._z = z
    

    def compute_padding(self, H, W, return_size_of_padding = False):
        
        # get the shape of processing
        if not self.do_padding:
            paddingH = 0
            paddingW = 0
            paddedH = int(H)
            paddedW = int(W)
        else:
            paddingH = int(np.floor((self.DEFAULT_PADDING_SCALE[0] * H) / 2))
            paddingW = int(np.floor((self.DEFAULT_PADDING_SCALE[1] * W) / 2))
            paddedH = H + 2*paddingH
            paddedW = W + 2*paddingW
        
        if not return_size_of_padding:
            return paddedH, paddedW
        else:
            return paddingH, paddingW
    
    def create_spatial_grid(self, H, W, dx, dy):
        """
        Returns the grid where the transfer function is defined. [Ref1.Eq.12]
        """
        x = torch.linspace(-H * dx / 2, H * dx / 2, H)
        y = torch.linspace(-W * dx / 2, W * dx / 2, W)
        meshx, meshy = torch.meshgrid(x, y, indexing='ij')

        return meshx.to(device=self.device), meshy.to(device=self.device)
    
    def check_far_field_z(self, quality_factor=1, dx=None, dy=None, wavelength=None):
        pass
    
    def create_kernel(self,
		field : ElectricField,
			):
        
        # orginal size of E-field
        tempShape = torch.tensor(field.shape)
        tempShapeH = tempShape[-2]
        tempShapeW = tempShape[-1]
        # padding size of E-field
        Pad_tempShapeH, Pad_tempShapeW = self.compute_padding(tempShape[-2], tempShape[-1], return_size_of_padding=False)
        
        # extract dx, dy spacing
        dx = field.spacing[0]
        dy = field.spacing[1]
        
        # create grid for RS transfer function
        self.meshx, self.meshy = self.create_spatial_grid(Pad_tempShapeH, Pad_tempShapeW, dx, dy)

        #################################################################
		# Prepare Dimensions to shape to be able to process 4D data
		# ---------------------------------------------------------------
		# NOTE: This just means we're broadcasting lower-dimensional
		# tensors to higher dimensional ones

        # Extract and expand wavelengths for H and W dimension
        wavelengths = field.wavelengths
        wavelengths_expand  = wavelengths[:,None,None]
        k = 2 * torch.pi / wavelengths_expand
        
        # RS transfer function
        r = torch.sqrt(self.meshx**2 + self.meshy**2 + self._z**2)
        factor = 1 / (2 * torch.pi) * self._z / r**2 *(1 / r - 1j * k)
        kernel = torch.exp(1j * k * r) * factor
        
        if self.check_Zc:
            self.check_RS_minimum_z(quality_factor=1, dx=dx, dy=dy, wavelength=torch.min(wavelengths))
            self.check_Zc = False # only check once during any loop operation

        return kernel[None, ...]         
        
    
    def forward(self, 
                field: ElectricField
                ) -> ElectricField:
        
        """
		Rayleigh-Sommerfeld convolution

		Args:
			field (ElectricField): Complex field 4D tensor object

		Returns:
			ElectricField: The electric field after the wave propagation model
		"""
  
        # extract the data tensor from the field
        wavelengths = field.wavelengths
        field_data  = field.data
        B,C,H,W = self.shape = field_data.shape
        dx = field.spacing[0]
        dy = field.spacing[1]
        
        #################################################################
        # Apply Rayleigh-Sommerfeld convolution
        #################################################################
        
        # Returns a spatial domain kernel with padding size
        RSC_Kernel = self.create_kernel(field=field)

        U = torch.zeros_like(RSC_Kernel)

        U[..., 0:H, 0:W] = field_data  # Ref 1 Eq.11

        # convert to angular spectrum
        field_data_spectrum = fft2(U) * fft2(RSC_Kernel) * dx * dy
                
        # Convert 'field_data' back to the space domain
        # Ref 1 Eq.15 'lower right submatrix'
        field_data = ifft2(field_data_spectrum)[..., H:, W:]	# Dimensions: B=1 x C x H x W

        Eout = ElectricField(
				data=field_data,
				wavelengths=wavelengths,
				spacing=field.spacing
				)
        
        return Eout