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
    """
    Class to represent and manipulate electric field data.
    """
    _BATCH = 0
    _WAVELENGTH = 1
    _HEIGHT = 2
    _WIDTH = 3

    def __init__(self, 
                data          : torch.Tensor,
                wavelengths   : Union[torch.Tensor, float] = None,
                spacing       : Union[torch.Tensor, float] = None,
                requires_grad : bool = None, 
                device		  : torch.device = None
                ):
        
        """
        Initialize an ElectricField object with data and optional wavelengths, spacing, and device parameters.
        """

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._spacing = self.check_spacing(spacing)
        self._wavelengths = self.check_wavelengths(wavelengths)

        self.field_type = None # scalar or vectorial EM field

        # build an empty data
        if data == None:
            data = torch.empty(1, len(self._wavelengths), 1, 1)

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
        if isinstance(spacing, (list, tuple)) and len(spacing) == 2:
            spacing = torch.tensor(spacing, dtype=torch.float32)
        elif isinstance(spacing, (float, int)):
            spacing = torch.tensor([spacing, spacing], dtype=torch.float32)
        if not torch.is_tensor(spacing) or spacing.numel() != 2:
            raise ValueError("Spacing must be a 2-element tensor.")
        return spacing.to(self.device)

    def check_wavelengths(self, wavelengths):
        if isinstance(wavelengths, (list, float, int)):
            wavelengths = torch.tensor([wavelengths] if isinstance(wavelengths, (float, int)) else wavelengths, dtype=torch.float32)
        if not torch.is_tensor(wavelengths):
            raise ValueError("Wavelengths must be a tensor.")
        return wavelengths.to(self.device)
    
    def check_data(self, data):
        """
        Verifies the data tensor's shape and dimensions, and categorizes it as scalar or vectorial.
        """
        assert torch.is_tensor(data) and data.ndim == 4, "Data must be a 4D torch tensor with BATCH x Channel (Wavelength) x Height x Width"
        # Check if number of channels in data equals number of wavelengths
        if data.shape[self._WAVELENGTH] != len(self._wavelengths):
            raise ValueError('The number of channels in data should be equal to the number of wavelengths')
        
        if data.shape[self._BATCH] == 1:
            self.field_type = 'scalar'
        elif data.shape[self._BATCH] == 3:
            self.field_type = 'vectorial'
        else:
            raise ValueError(f'The field must be scalar (B=1) or vectorial (B=3), but the first dimension (batch size) of data is {data.shape[self._BATCH]}.')

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
    
    @property
    def Ex(self):
        return self._data[[0], ...]
    
    @property
    def Ey(self):
        return self._data[[1], ...]
    
    @property
    def Ez(self):
        return self._data[[2], ...]
    
    def _get_data_for_wavelength(self, wavelength):
        # Find the index of the specified wavelength
        idx = (self._wavelengths == wavelength).nonzero()[0]
        return self._data[:, idx, ...]

    def _plot_scalarfield(self, 
                          data, 
                          flag_colorbar : bool    = True,
                          flag_axis : str         = True,
                          cmap                    ='viridis',
                          wavelength              = None,
                          figsize                 = (8,8),
                          intensity               =True):

        wavelength = float(wavelength)
        if figsize is not None:
            plt.figure(figsize=figsize)
        
        dx = self._spacing[0].detach().cpu()
        dy = self._spacing[1].detach().cpu()

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
        
        unit_val, unit = float_to_unit_identifier(wavelength)
        wavelength = wavelength / unit_val

        # plot intensity of amplitude of scalar field
        plt.subplot(1, 2, 1)
        if intensity == True:
            I = data.abs()**2
            _im1 = plt.imshow(I, cmap=cmap, extent=extent, vmax=torch.max(I), vmin=torch.min(I))
            plt.title("Intensity| wavelength = " + str(round(wavelength, 2)) + str(unit))
        else:    
            A = data.abs()
            _im1 = plt.imshow(A, cmap=cmap, extent=extent, vmax=torch.max(A), vmin=torch.min(A))
            plt.title("Amplitude| wavelength = " + str(round(wavelength, 2)) + str(unit))
        if flag_axis:
            if unit != "":
                plt.xlabel("Position (" + unit_axis + ")")
                plt.ylabel("Position (" + unit_axis + ")")

        # plot phase of scalar field
        plt.subplot(1, 2, 2)
        Phi = data.angle()
        _im2 = plt.imshow(Phi, cmap=cmap, extent=extent, vmax=torch.pi, vmin=-torch.pi)
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

    def _plot_vectorialfield(self, 
                             data, 
                             flag_colorbar : bool    = True,
                             flag_axis : str         = True,
                             cmap                    ='viridis',
                             wavelength              = None,
                             figsize                 = (8,8),
                             intensity               =True):
        wavelength = float(wavelength)
        if figsize is not None:
            plt.figure(figsize=figsize)

        dx = self._spacing[0].detach().cpu()
        dy = self._spacing[1].detach().cpu()

        # three componenets of vectorial EM field
        Ex = data[0, ...]
        Ey = data[1, ...]
        Ez = data[2, ...]
        
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
        
        unit_val, unit = float_to_unit_identifier(wavelength)
        wavelength = wavelength / unit_val
        if intensity == True:
                
            Ix = Ex.abs()**2
            Iy = Ey.abs()**2
            Iz = Ez.abs()**2

            plt.subplot(3, 2, 1)
            _im1 = plt.imshow(Ix, cmap=cmap, extent=extent, vmax=torch.max(Ix), vmin=torch.min(Ix))
            plt.title("Intensity x| wavelength = " + str(round(wavelength, 2)) + str(unit))
            if flag_axis:
                if unit != "":
                    plt.xlabel("Position (" + unit_axis + ")")
                    plt.ylabel("Position (" + unit_axis + ")")
                
            plt.subplot(3, 2, 3)
            _im2 = plt.imshow(Iy, cmap=cmap, extent=extent, vmax=torch.max(Iy), vmin=torch.min(Iy))
            plt.title("Intensity y| wavelength = " + str(round(wavelength, 2)) + str(unit))
            if flag_axis:
                if unit != "":
                    plt.xlabel("Position (" + unit_axis + ")")
                    plt.ylabel("Position (" + unit_axis + ")")

            plt.subplot(3, 2, 5)
            _im3 = plt.imshow(Iz, cmap=cmap, extent=extent, vmax=torch.max(Iz), vmin=torch.min(Iz))
            plt.title("Intensity z| wavelength = " + str(round(wavelength, 2)) + str(unit))
            if flag_axis:
                if unit != "":
                    plt.xlabel("Position (" + unit_axis + ")")
                    plt.ylabel("Position (" + unit_axis + ")")

        else:    
            Ax = Ex.abs()
            Ay = Ey.abs()
            Az = Ez.abs()

            plt.subplot(3, 2, 1)
            _im1 = plt.imshow(Ax, cmap=cmap, extent=extent, vmax=torch.max(Ax), vmin=torch.min(Ax))
            plt.title("Amplitude x| wavelength = " + str(round(wavelength, 2)) + str(unit))
            if flag_axis:
                if unit != "":
                    plt.xlabel("Position (" + unit_axis + ")")
                    plt.ylabel("Position (" + unit_axis + ")")
                
            plt.subplot(3, 2, 3)
            _im2 = plt.imshow(Ay, cmap=cmap, extent=extent, vmax=torch.max(Ay), vmin=torch.min(Ay))
            plt.title("Amplitude y| wavelength = " + str(round(wavelength, 2)) + str(unit))
            if flag_axis:
                if unit != "":
                    plt.xlabel("Position (" + unit_axis + ")")
                    plt.ylabel("Position (" + unit_axis + ")")

            plt.subplot(3, 2, 5)
            _im3 = plt.imshow(Az, cmap=cmap, extent=extent, vmax=torch.max(Az), vmin=torch.min(Az))
            plt.title("Amplitude z| wavelength = " + str(round(wavelength, 2)) + str(unit))
            if flag_axis:
                if unit != "":
                    plt.xlabel("Position (" + unit_axis + ")")
                    plt.ylabel("Position (" + unit_axis + ")")
                
        Phi_x = Ex.angle()
        Phi_y = Ey.angle()
        Phi_z = Ez.angle()

        plt.subplot(3, 2, 2)
        _im4 = plt.imshow(Phi_x, cmap=cmap, extent=extent, vmax=-torch.pi, vmin=torch.pi)
        plt.title("Phase x (in radians)| wavelength = " + str(round(wavelength, 2)) + str(unit))
        if flag_axis:
            if unit != "":
                plt.xlabel("Position (" + unit_axis + ")")
                plt.ylabel("Position (" + unit_axis + ")")
            
        plt.subplot(3, 2, 4)
        _im5 = plt.imshow(Phi_y, cmap=cmap, extent=extent, vmax=-torch.pi, vmin=torch.pi)
        plt.title("Phase y (in radians)| wavelength = " + str(round(wavelength, 2)) + str(unit))
        if flag_axis:
            if unit != "":
                plt.xlabel("Position (" + unit_axis + ")")
                plt.ylabel("Position (" + unit_axis + ")")
            
        plt.subplot(3, 2, 6)
        _im6 = plt.imshow(Phi_z, cmap=cmap, extent=extent, vmax=-torch.pi, vmin=torch.pi)
        plt.title("Phase z (in radians)| wavelength = " + str(round(wavelength, 2)) + str(unit))
        if flag_axis:
            if unit != "":
                plt.xlabel("Position (" + unit_axis + ")")
                plt.ylabel("Position (" + unit_axis + ")")
        
        if flag_colorbar:
            add_colorbar(_im1)
            add_colorbar(_im2)
            add_colorbar(_im3)
            add_colorbar(_im4)
            add_colorbar(_im5)
            add_colorbar(_im6)
        
        if flag_axis:
            plt.axis("on")
        else:
            plt.axis("off")
                
        if figsize is not None:
            plt.tight_layout()

    def visualize(self, 
                  flag_colorbar : bool    = True,
                  flag_axis : str         = True,
                  cmap                    ='viridis',
                  wavelength              = None,
                  figsize                 = (8,8),
                  intensity               =True):

        """
        Visualize the electric field data for a specified wavelength.
        """
        assert wavelength is not None, "Wavelength must be specified."

        data = self._get_data_for_wavelength(wavelength).detach().cpu().squeeze()

        if self.field_type == 'scalar':  # plot for scalar EM field
            self._plot_scalarfield(data, flag_colorbar, flag_axis, cmap, wavelength, figsize, intensity)
        
        if self.field_type == 'vectorial': # plot for vectorial EM field
            self._plot_vectorialfield(data, flag_colorbar, flag_axis, cmap, wavelength, figsize, intensity)


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
