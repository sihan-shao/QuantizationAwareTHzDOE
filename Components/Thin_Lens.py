import sys
sys.path.append('../')
import torch
import torch.nn as nn
import numpy as np
import pathlib
import matplotlib
import matplotlib.pyplot as plt
from copy import copy
from utils.units import *
from utils.Visualization_Helper import float_to_unit_identifier, add_colorbar
from DataType.ElectricField import ElectricField



class Thin_LensElement(nn.Module):
    
    def __init__(self,
                 focal_length, 
                 device             : torch.device = None
                 ):
        """
		This implements a thin lens by applying the phase shift described in Equation (6-10) of Goodman's Fourier optics book
		"""
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.focal_length =torch.Tensor([focal_length]).to(self.device)

    def create_lens_phase_shift_kernel(self,
		field : ElectricField,
			):

        # extract dx, dy spacing
        dx = torch.tensor([field.spacing[0]], device=self.device)
        dy = torch.tensor([field.spacing[1]], device=self.device)

        dx_expand = dx[:, None, None] # Expand to H and W
        dy_expand = dy[:, None, None] # Expand to H and W
        
        wavelengths = field.wavelengths
        # Expand wavelengths for H and W dimension
        wavelengths_expand  = wavelengths[:,None,None]
        
        height, width = field.height, field.width
        xCoords = torch.linspace(-((height - 1) // 2), (height - 1) // 2, height)
        yCoords = torch.linspace(-((width - 1) // 2), (width - 1) // 2, width)
        xGrid, yGrid = torch.meshgrid(xCoords, yCoords)
        
        xGrid = xGrid[None,None,:,:].to(self.device) * dx_expand
        yGrid = yGrid[None,None,:,:].to(self.device) * dy_expand
        
        ang = -(np.pi / (wavelengths_expand * self.focal_length)) * ((xGrid ** 2) + (yGrid ** 2))
        
        ker = torch.exp(1j * ang)
        
        return ker
    
    def forward(self,
                field: ElectricField
                ) -> ElectricField:
        
        """
		In this function we apply a phase delay that simulates a thin lens

		Args:
			field(torch.complex128) : Complex field (MxN).
		"""

        # extract the data tensor from the field
        wavelengths = field.wavelengths
        field_data  = field.data
        
        phase_shift_ker = self.create_lens_phase_shift_kernel(field=field)
        
        field_data = field_data * phase_shift_ker
        
        Eout = ElectricField(
				data=field_data,
				wavelengths=wavelengths,
				spacing = field.spacing
		)
        
        return Eout
    
    



