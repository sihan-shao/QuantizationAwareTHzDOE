import sys
sys.path.append('../')
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.nn.functional import pad
import matplotlib.pyplot as plts
import matplotlib
import matplotlib.pyplot as plt

from DataType.ElectricField import ElectricField
from utils.Helper_Functions import ft2, ift2, generateGrid
from torch.fft import fft2, ifft2
from utils.units import *

class RSC_prop(nn.Module):
    
    def __init__(self, 
                 z_distance         : float = 0.0, 
                 device             : str = None
                 ) -> None:
        """
        Rayleigh-Sommerfeld convolution
        [Ref 1: F. Shen and A. Wang, Appl. Opt. 45, 1102-1110 (2006)].

        Args:
            z_distance (float, optional): propagation distance. Defaults to 0.0.
			
        """
        super().__init__()

        self.do_padding = True
        self.DEFAULT_PADDING_SCALE = torch.tensor([1,1])
            
        # store the input params
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._z                  = torch.tensor(z_distance, device=self.device)
        
        # the normalized spatial grid
		# we don't actually know dimensions until forward is called
        self.meshx = None
        self.meshy = None

        self.check_Zc = True
    
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
        
        self.meshx, self.meshy = torch.meshgrid(x, y, indexing='ij')
        self.meshx = self.meshx.to(device=self.device)
        self.meshy = self.meshy.to(device=self.device)
    
    def check_RS_minimum_z(self, quality_factor=1, dx=None, dy=None, wavelength=None):
        """
        Given a quality factor, determines the minimum available (trustworthy) distance for VRS_propagation().
        [Ref 2: Laser Phys. Lett., 10(6), 065004 (2013)] for the perspective of energy conservation in FFT
        [Ref 3: J. Opt. Soc. Am. A 37, 1748-1766 (2020)] for the perspective of sampling in FFT 

        Parameters:
            quality_factor (int): Defaults to 1.
        
        part of code is adopted by https://github.com/artificial-scientist-lab/XLuminA/blob/main/xlumina

        Returns the minimum distance z necessary to achieve qualities larger than quality_factor in [Ref 2] or satisfy the sampling issue in [Ref 3].
        """

        # Ref 2 
        range_x, range_y = self.meshx.shape[0] * dx, self.meshx.shape[1] * dy
        # Delat rho
        dr_real = torch.sqrt(dx**2 + dy**2)
        # Rho
        rmax = torch.sqrt(range_x**2 + range_y**2)
        n = 1 # free space
        factor = (((quality_factor * dr_real + rmax)**2 - (wavelength / n)**2 - rmax**2) / (2 * wavelength / n))**2 - rmax**2

        if factor > 0:
            z_min1 = torch.sqrt(factor)
        else:
            z_min1 = 0
        
        print("Minimum propagation distance to satisfy energy conservation: {} mm".format(z_min1.detach().cpu().numpy() / mm))
        # Ref 3 Eq.34
        z_min2 = self.meshx.shape[0] * dx**2 / wavelength * torch.sqrt(1 - (wavelength / (2 * dx))**2)
        print("Minimum propagation distance to satisfy sampling for FT: {} mm".format(z_min2.detach().cpu().numpy() / mm))

        Zc = min(z_min1, z_min2)

        if self._z > Zc:
            print("The simulation will be accurate !")
        else:
            print("The propagation distance should be larger than minimum propagation distance to keep simulation accurate!")
    
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
        self.create_spatial_grid(Pad_tempShapeH, Pad_tempShapeW, dx, dy)


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
            self.check_RS_minimum_z(quality_factor=1, dx=dx, dy=dy, wavelength=torch.max(wavelengths))
            self.check_Zc = False # only check once during any loop operation

        return kernel           
        
    
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
        B,C,H,W = field_data.shape
        
        #################################################################
        # Apply Rayleigh-Sommerfeld convolution
        #################################################################
        
        # Returns a spatial domain kernel with padding size
        RSC_Kernel = self.create_kernel(field=field)
        print(RSC_Kernel.shape)
        
        kernelOut = fft2(h) * dx * dy


        U = torch.zeros_like(RSC_Kernel)
        U = U[..., 0:H, 0:W] = field_data  # Ref 1 Eq.11
                
        # convert to angular spectrum
        field_data_spectrum = fft2(U) * fft2(RSC_Kernel) * dx * dy
                
        # Convert 'field_data' back to the space domain
        # Ref 1 Eq.15 'lower right submatrix'
        field_data = ifft2(field_data_spectrum)[..., H-1:, W-1:]	# Dimensions: B x C x H x W

        Eout = ElectricField(
				data=field_data,
				wavelengths=wavelengths,
				spacing=field.spacing
				)
        
        return Eout


if __name__=='__main__':
    
    from utils.units import *
    import numpy as np
    import torch
    from LightSource.Gaussian_beam import Guassian_beam
    
    c0 = 2.998e8
    f1 = 340e9
    f2 = 360e9
    wavelength1 = c0 / f1
    wavelength2 = c0 / f2

    fs = torch.range(f1, f2, 4e9)

    wavelengths = c0 / fs
    print(wavelengths)

    gm = Guassian_beam(height=1000, width=1000, 
                       beam_waist=1 * mm, wavelengths=wavelength2, 
                       spacing=0.1 * mm)
    field = gm()
    
    asm_prop = RSC_prop(z_distance=1 * m)
    
    field_propagated = asm_prop.forward(
    field = field
    )
    
    