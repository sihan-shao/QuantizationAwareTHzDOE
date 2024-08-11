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



class ApertureElement(nn.Module):
    
    def __init__(self,
                 aperture_type      : str = 'circ',
                 aperture_size      : float = None, 
                 device             : torch.device = None
                 ):
        """
		This implements an aperture
        aperture_type:
            circ: add a circle aperture to the field
            rect: add a rectangle aperture to the field
        
        aperture_size:
            The size of defined aperture, can't be larger than field size in the simulation
            if aperture_type is circ, the aperture_size should be the radius
            if aperture_type is rect, the aperture_size should be width and length

		"""
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.aperture_type = aperture_type
        self.aperture_size = aperture_size

    
    
    def add_circ_aperture_to_field(self,
                                   input_field    : ElectricField, 
                                   radius=None)-> torch.Tensor: 
        

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
        
        return Mask.to(self.device)
    
    def add_rect_aperture_to_field(self,
                                   input_field, 
                                   rect_width=None, 
                                   rect_height=None)-> torch.Tensor:
        
        
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
        
        return Mask.to(self.device)
    
    
    def forward(self,
                field: ElectricField
                ) -> ElectricField:
        
        """
		Args:
			field(torch.complex128) : Complex field (MxN).
		"""

        if self.aperture_type == 'circ':
            self.aperture = self.add_circ_aperture_to_field(field, 
                                                            radius=self.aperture_size)
        
        elif self.aperture_type == 'rect':
            self.aperture = self.add_rect_aperture_to_field(field, 
                                                            rect_height=self.aperture_size, 
                                                            rect_width=self.aperture_size)
        elif self.aperture_type == None:
            self.aperture = torch.ones_like(field.data)
        
        else:
            ValueError('No exisiting aperture shape, please define by yourself')
            
        out_field = self.aperture * field.data
        
        Eout = ElectricField(
				data=out_field,
				wavelengths=field.wavelengths,
				spacing = field.spacing
		)
        
        return Eout