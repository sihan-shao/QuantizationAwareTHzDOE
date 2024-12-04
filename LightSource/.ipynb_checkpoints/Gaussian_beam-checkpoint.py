import sys
sys.path.append('../')
import math
import torch
import torch.nn as nn
import numpy as np
from utils.units import *
from DataType.ElectricField import ElectricField

class Guassian_beam(nn.Module):
    
    """[summary]


    Source is 4D - (B or 1) x C x H X W
    
    """    
    def __init__(self, 
                 height: int,
                 width: int, 
                 beam_waist: float,
                 wavelengths : torch.Tensor or float = None, 
                 spacing : torch.Tensor or float =None, 
                 device = None):
        
        super().__init__()
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
        self._height = height
        self._width = width
        self._beam_waist = self.check_wavelengths_beam_waist(beam_waist)
        self._wavelengths = self.check_wavelengths_beam_waist(wavelengths)
        self._spacing =self.check_spacing(spacing)

    
    def check_spacing(self, spacing):
        if torch.is_tensor(spacing):
            if len(spacing.shape) != 1 or spacing.shape[0] != 2:
                raise ValueError('spacing should be a 2-element Tensor')
        elif isinstance(spacing, (float, int)):
            spacing = torch.tensor([spacing, spacing], dtype=torch.float32)
        elif isinstance(spacing, list) and len(spacing) == 2 and all(isinstance(x, (float, int)) for x in spacing):
            spacing = torch.tensor(spacing, dtype=torch.float32)
        else:
            raise ValueError('spacing should be a float or a list of two numbers')
        return spacing.to(self.device)

    def check_wavelengths_beam_waist(self, wavelengths):
        if torch.is_tensor(wavelengths):
            if wavelengths.ndim == 0:
                wavelengths = torch.tensor([wavelengths])
        elif isinstance(wavelengths, (float, int)):
            wavelengths = torch.tensor([wavelengths], dtype=torch.float32)
        elif isinstance(wavelengths, list) and all(isinstance(x, (float, int)) for x in wavelengths):
            wavelengths = torch.tensor(wavelengths, dtype=torch.float32)
        else:
            raise ValueError('wavelengths/beam_waist should be a float, a list of floats, or a torch.Tensor')
        return wavelengths.to(self.device)
    
    ##############################################################################
    #
    # SETTER METHODS
    #
    ##############################################################################
    
    @property
    def height(self) -> torch.Tensor:
        return self._height
    
    @height.setter
    def height(self, height : torch.Tensor) -> None:
        self._height = height
    
    @property
    def width(self) -> torch.Tensor:
        return self._width
    
    @width.setter
    def width(self, width : torch.Tensor) -> None:
        self._width = width
    
    @property
    def beam_waist(self) -> torch.Tensor:
        return self._beam_waist
    
    @beam_waist.setter
    def beam_waist(self, beam_waist : torch.Tensor) -> None:
        self._beam_waist = self.check_wavelengths_beam_waist(beam_waist)
                  
    @property
    def spacing(self) -> torch.Tensor:
        return self._spacing
    
    @spacing.setter
    def spacing(self, spacing : torch.Tensor) -> None:
        self._spacing = self.check_spacing(spacing)
    
    @property
    def wavelengths(self):
        return self._wavelengths

    @wavelengths.setter
    def wavelengths(self, wavelengths: torch.Tensor) -> None:
        self._wavelengths = self.check_wavelengths_beam_waist(wavelengths)
        
    def forward(self)-> ElectricField:
        
        # you need to implement Gaussian beam light source with speific beam waist
        dx, dy = self._spacing[0], self._spacing[1]
        
        x = torch.linspace(-dx * self._height / 2, 
                           dx * self._height / 2, 
                           self._height, device=self.device)
        
        y = torch.linspace(-dy * self._width / 2, 
                           dy * self._width / 2, 
                           self._width, device=self.device)
        
        X, Y = torch.meshgrid(x, y)
        
        R = torch.sqrt(X**2 + Y**2)
        
        # check if the beam waist match the wavelength
        if len(self._beam_waist) != len(self._wavelengths):
            if len(self._beam_waist) == 1:  # If beam waist is a single number, repeat it for all wavelengths
                self._beam_waist = self._beam_waist.repeat(len(self._wavelengths))
            else:
                raise ValueError('Mismatch between beam waist and wavelength parameters')

        # Create the Gaussian beam amplitude field for each wavelength
        amplitude_fields = torch.exp(-(R[None, ...] / self._beam_waist[:, None, None])**2)
        #print(amplitude_fields.is_cuda)
        # Assuming flat phase, so phase fields are just zeros with the same shape as the amplitude fields
        phase_fields = torch.zeros_like(amplitude_fields, device=self.device)

        # Combine amplitude and phase to form complex electric fields
        electric_fields = amplitude_fields * torch.exp(1j * phase_fields).unsqueeze(dim=0) # add Batch dimension

        out = ElectricField(
                data=electric_fields.type(torch.complex64), 
                wavelengths=self._wavelengths,
                spacing=self._spacing
        )
        
        
        return out
    

if __name__=='__main__':
    
    from utils.units import *
    import numpy as np
    import torch
    
    gm = Guassian_beam(height=500, width=500, 
                       beam_waist=1 * mm, wavelengths=[300 * nm, 354 * nm], 
                       spacing=0.2 * mm)
    field = gm()
    print(field.shape)