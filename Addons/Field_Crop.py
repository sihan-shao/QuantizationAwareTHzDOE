import sys
sys.path.append('../')
import torch
import torch.nn as nn
from torch.nn.functional import grid_sample
import numpy as np
import pathlib
import matplotlib
import matplotlib.pyplot as plt
from copy import copy
from utils.units import *
from utils.Visualization_Helper import float_to_unit_identifier, add_colorbar
from utils.Helper_Functions import generateGrid
from DataType.ElectricField import ElectricField


class Field_Cropper(nn.Module):
    
    def __init__(self, 
                 outputHeight						: int,
				 outputWidth						: int,
				 device								: torch.device = None
     )-> None:
        
        super().__init__()
        """
        Crops complex fields
        
        field: the field to be padded. May have as many leading dimensions as necessary
        (e.g., batch or channel dimensions) shape: [B, wavelength, H, W]
        
        outputHeight(outputWidth): the 2D target output dimensions. If any dimensions are larger
        than field, no cropping is applied
         
        """
        

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        if (type(outputHeight) is not int) or (outputHeight <= 0):
            raise Exception("Bad argument: 'outputHeight' should be a positive integer.")
        if (type(outputWidth) is not int) or (outputWidth <= 0):
            raise Exception("Bad argument: 'outputWidth' should be a positive integer.")

        self.outputHeight = outputHeight
        self.outputWidth = outputWidth
        
        
    def forward(self, field):
        
        _,_,Hf,Wf = field.data.shape
        field_data = field.data

        crop_height_front = int(round(Hf - self.outputHeight) / 2.0)
        crop_width_front = int(round(Wf - self.outputWidth) / 2.0)

        cropped_field = field_data[:, :, crop_height_front: crop_height_front + self.outputHeight, crop_width_front: crop_width_front + self.outputWidth]
        
        Eout = 	ElectricField(
					data = cropped_field,
					wavelengths = field.wavelengths,
					spacing = field.spacing
				)

        
        return Eout