import sys
sys.path.append('./')
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


class Field_Resampler(nn.Module):
    
    def __init__(self, 
                 outputHeight						: int,
				 outputWidth						: int,
				 outputPixel_dx						: float,
				 outputPixel_dy						: float,
				 device								: torch.device = None
     )-> None:
        
        super().__init__()
        """
        resample an electric field to a new grid with a specified output resolution and pixel size
        """
        

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        if (type(outputHeight) is not int) or (outputHeight <= 0):
            raise Exception("Bad argument: 'outputHeight' should be a positive integer.")
        if (type(outputWidth) is not int) or (outputWidth <= 0):
            raise Exception("Bad argument: 'outputWidth' should be a positive integer.")
        if ((type(outputPixel_dx) is not float) and (type(outputPixel_dx) is not int)) or (outputPixel_dx <= 0):
            raise Exception("Bad argument: 'outputPixel_dx' should be a positive real number.")
        if ((type(outputPixel_dy) is not float) and (type(outputPixel_dy) is not int)) or (outputPixel_dy <= 0):
            raise Exception("Bad argument: 'outputPixel_dy' should be a positive real number.")

        self.outputResolution = [outputHeight, outputWidth]
        self.outputPixel_dx = outputPixel_dx
        self.outputPixel_dy = outputPixel_dy
        self.outputSpacing = [outputPixel_dx, outputPixel_dy]
        
        
        outputGridX, outputGridY = generateGrid(self.outputResolution, outputPixel_dx, outputPixel_dy, device=self.device)
        self.outputGridX = outputGridX
        self.outputGridY = outputGridY
        
        self.calculateOutputCoordGrid()
        
        self.grid = None
        self.prevFieldSpacing = None
        self.prevFieldSize = None
        
    def calculateOutputCoordGrid(self):
        """Can assume that coordinate (0,0) is in the center due to how generateGrid(...) works
        """
        gridX = self.outputGridX
        gridY = self.outputGridY
        
        grid = torch.zeros(self.outputResolution[0], self.outputResolution[1], 2, device=self.device)
        
        grid[:,:,0] = gridY
        grid[:,:,1] = gridX
        
        self.gridPrototype = grid.to(self.device)
        
    def forward(self, field):
        
        Bf,Cf,Hf,Wf = field.data.shape
        field_data = field.data
        
        # convert spacing to 4D tensor
        # extract dx, dy spacing
        dx = torch.tensor([field.spacing[0]], device=self.device)
        dy = torch.tensor([field.spacing[1]], device=self.device)
        dx_expand = dx[:, None, None] # Expand to H and W
        dy_expand = dy[:, None, None] # Expand to H and W
        xNorm = dx_expand * ((Hf - 1) // 2)
        xNorm = xNorm[:,:,None,:]
        yNorm = dy_expand * ((Wf - 1) // 2)
        yNorm = yNorm[:,:,None,:]
        
        self.grid = self.gridPrototype.repeat(Bf,1,1,1)
        
        if self.grid.type() != field_data.real.type():
            self.grid = self.grid.type(field_data.real.type())
        
        # Stuff is ordered this way because torch.nn.functiona.grid_sample(...) has x as the coordinate in the width direction
		# and y as the coordinate in the height dimension.  This is the opposite of the convention used by this code.
        self.grid[... , 0] = self.grid[... , 0] / yNorm
        self.grid[... , 1] = self.grid[... , 1] / xNorm
    
        self.prevFieldSpacing = field.spacing
        self.prevFieldSize = field.data.shape
        
        new_data = grid_sample(field_data.real, self.grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        new_data = new_data + (1j * grid_sample(field_data.imag, self.grid, mode='bilinear', padding_mode='zeros', align_corners=True))


        
        new_spacing_x = self.outputPixel_dx
        new_spacing_y = self.outputPixel_dy
        
        Eout = 	ElectricField(
					data = new_data,
					wavelengths = field.wavelengths,
					spacing = [new_spacing_x, new_spacing_y]
				)

        
        return Eout
        

  
    