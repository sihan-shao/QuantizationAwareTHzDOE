from __future__ import annotations
from torch.types import _device, _dtype, _size
import sys
sys.path.append('../')
import torch
import numpy as np
import pathlib
import matplotlib
import matplotlib.pyplot as plt
from copy import copy
from utils.units import *
from utils.Visualization_Helper import float_to_unit_identifier, add_colorbar

class ElectricField():

    def __init__(self, 
                data                                : torch.Tensor,
                wavelengths                         : torch.Tensor or float = None,
                spacing                             : torch.Tensor or float = None,
                requires_grad                       : bool = None, 
                device								: torch.device = None
                ):
        
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._spacing = self.check_spacing(spacing)
        self._wavelengths = self.check_wavelengths(wavelengths)
        
        self._data : torch.Tensor = self.check_data(data)

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
        self._wavelengths = self.check_wavelengths(wavelengths)

    @property
    def requires_grad(self):
        return self._data.requires_grad
          
    @property
    def data(self) -> torch.Tensor:
        return self._data

    @data.setter
    def data(self, data):
        self._data = self.check_data(data)
        
    def check_spacing(self, spacing):
        if torch.is_tensor(spacing):
            if len(spacing.shape) != 1 or spacing.shape[0] != 2:
                raise ValueError('spacing should be a 2-element Tensor')
        elif isinstance(spacing, (float, int)):
            spacing = torch.tensor([spacing, spacing])
        elif isinstance(spacing, list) and len(spacing) == 2 and all(isinstance(x, (float, int)) for x in spacing):
            spacing = torch.tensor(spacing)
        else:
            raise ValueError('spacing should be a float or a list of two numbers')
        return spacing.to(self.device)

    def check_wavelengths(self, wavelengths):
        if torch.is_tensor(wavelengths):
            if wavelengths.ndim == 0:
                wavelengths = torch.tensor([wavelengths])
        elif isinstance(wavelengths, (float, int)):
            wavelengths = torch.tensor([wavelengths])
        elif isinstance(wavelengths, list) and all(isinstance(x, (float, int)) for x in wavelengths):
            wavelengths = torch.tensor(wavelengths)
        else:
            raise ValueError('wavelengths should be a float, a list of floats, or a torch.Tensor')
        return wavelengths.to(self.device)
    
    def check_data(self, data):
        assert torch.is_tensor(data) and data.ndim == 4, "Data must be a 4D torch tensor with BATCH x Channel (Wavelength) x Height x Width"
        # Check if number of channels in data equals number of wavelengths
        if data.shape[1] != len(self._wavelengths):
            raise ValueError('The number of channels in data should be equal to the number of wavelengths')
        return data.to(self.device)
        
        
    def abs(self) -> ElectricField:
        """ Detaches the data and wavelength tensor from the computational graph
        """        
        
        data = self._data.abs()
        wavelengths = self._wavelengths   
        spacing = self._spacing
             
        return ElectricField(
            data = data,
            wavelengths=wavelengths,
            spacing = spacing
                     )   
    
    def angle(self) -> ElectricField:
        """ Detaches the data and wavelength tensor from the computational graph
        """        
        
        data = self._data.angle()
        wavelengths = self._wavelengths   
        spacing = self._spacing
             
        return ElectricField(
            data = data,
            wavelengths=wavelengths,
            spacing = spacing
                     )
    
    def detach(self) -> ElectricField:
        """ Detaches the data and wavelength tensor from the computational graph
        """        
        
        data = self._data.detach()
        wavelengths = self._wavelengths.detach()
        spacing = self._spacing.detach()

        
        return ElectricField(
            data = data,
            wavelengths=wavelengths,
            spacing = spacing
                     ) 
    
    def cpu(self) -> ElectricField:
        """ Detaches the data and wavelength tensor from the computational graph
        """        
        
        data = self._data.cpu()
        wavelengths = self._wavelengths.detach()
        spacing = self._spacing.detach()

        return ElectricField(
            data = data,
            wavelengths=wavelengths,
            spacing = spacing
                     )    
    
    _BATCH = 0
    _WAVELENGTH = 1
    _HEIGHT = 2
    _WIDTH = 3
    
    
    @property
    def ndim(self):
        return self._data.ndim 
    
    @property
    def shape(self):
        return self._data.shape 
    
    @property
    def num_batches(self):
        return self.shape[self._BATCH] 
    
    @property
    def num_wavelengths(self):
        return self.shape[self._WAVELENGTH]
    
    @property
    def height(self):
        return self.shape[self._HEIGHT]
    
    @property
    def width(self):
        return self.shape[self._WIDTH]
    
    def visualize(self, 
                flag_colorbar : bool    = True,
                flag_axis : str         = True,
                cmap                    ='viridis',
                wavelength              = None,
                figsize                 = (8,8),
                vmax                    = None,
                vmin                    = None,
                intensity               =True):
        
        assert wavelength != None, "Select the wavelength to be visualized"
        wavelength = float(wavelength)
        
        if figsize is not None:
            plt.figure(figsize=figsize)
        
        # Prepare data for plotting
        # Find the index of the specified wavelength
        idx = (self._wavelengths == wavelength).nonzero()[0]
        # Get the data for the specified wavelength
        data = self._data[:,idx, ...].detach().cpu().squeeze()
        dx = self._spacing[0].detach().cpu()
        dy = self._spacing[1].detach().cpu()
        
        abs_data = data.abs()
        phase_data = data.angle()
        
        if flag_axis == True:
            size_x = np.array(dx / 2.0 * self.height)
            size_y =np.array(dy / 2.0 * self.width)
            
            unit_val, unit_axis = float_to_unit_identifier(max(size_x,size_y))
            size_x = size_x / unit_val
            size_y = size_y / unit_val
            
            extent = [-size_y, size_y, -size_x, size_x]

        else:
            extent = None
            size_x = self.height
            size_y = self.width
        
        if data.ndim == 2:
            unit_val, unit = float_to_unit_identifier(wavelength)
            wavelength = wavelength / unit_val
            plt.subplot(1, 2, 1)
            if intensity == True:
                _im1 = plt.imshow(abs_data**2, cmap=cmap, extent=extent, vmax=vmax, vmin=vmin)
                plt.title("Intensity| wavelength = " + str(round(wavelength, 2)) + str(unit))
            else:    
                _im1 = plt.imshow(abs_data, cmap=cmap, extent=extent, vmax=vmax, vmin=vmin)
                plt.title("Amplitude| wavelength = " + str(round(wavelength, 2)) + str(unit))
            if flag_axis:
                if unit != "":
                    plt.xlabel("Position (" + unit_axis + ")")
                    plt.ylabel("Position (" + unit_axis + ")")
                
            plt.subplot(1, 2, 2)
            _im2 = plt.imshow(phase_data, cmap=cmap, extent=extent)
            plt.title("Phase| wavelength = " + str(round(wavelength, 2)) + str(unit))
            if flag_axis:
                if unit != "":
                    plt.xlabel("Position (" + unit_axis + ")")
                    plt.ylabel("Position (" + unit_axis + ")")
        
        if flag_colorbar:
            add_colorbar(_im1)
            add_colorbar(_im2)
        
        if flag_axis:
            plt.axis("on")
        else:
            plt.axis("off")
                
        if figsize is not None:
            plt.tight_layout()


if __name__=='__main__':
    
    from utils.units import *
    import numpy as np
    import torch


    N = 1024
    wanglengths = torch.tensor(np.array([532 * nm, 450 * nm]))
    field_data = torch.zeros(1,2,N,N) + 0j # 0j to make it complex

    # Set ones to the field
    field_data[...,N//4 : 3 * N//4, N//4 : 3 * N//4] = 1
    # Cast into our Holotorch Datatype
    field_input = ElectricField(
        data = field_data, 
        wavelengths = wanglengths,
        spacing = [8 * um, 8 * um],
    )
    
    field_input.spacing = 2 * um
    print(field_input.spacing)
    field_data1 = torch.zeros(1,1,N,N) + 0j # 0j to make it complex
    field_input.wavelengths = 532 * nm
    field_input.data = field_data1
    print(field_input.ndim)
    
    field_abs = field_input.abs()
    print(field_abs.shape)
