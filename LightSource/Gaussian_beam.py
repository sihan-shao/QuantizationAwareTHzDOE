import sys
sys.path.append('./')
import math
import torch
import torch.nn as nn
import numpy as np
from utils.units import *
from DataType.ElectricField import ElectricField

LIGHT_SPEED = 2.998e8

class Guassian_beam(nn.Module):
    
    """[summary]

    Define a Vectorial Guassian Beam
    Source is 4D - 3 x C x H X W

    Params:
        height (int): resolution of height
        width (int): resolution of width
        beam_waist_x (float): beam waist in x-direction
        beam_waist_y (float): beam waist in y-direction
        center (float, float): Position of the center of the beam
        z_w0 (float, float): Position of the waist for (x, y) 
        alpha (float, float): Amplitude rotation (in radians).
        wavelength (float or list): wavelength of source field
        spacing (float or list): physical length of field
    
    """    
    def __init__(self, 
                 height: int,
                 width: int, 
                 beam_waist_x: float,
                 beam_waist_y: float,
                 center=(0, 0), 
                 z_w0=(0, 0), 
                 alpha=0,
                 wavelengths : torch.Tensor or float = None, 
                 spacing : torch.Tensor or float =None, 
                 device = None):
        
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.height = height
        self.width = width

        self.field = ElectricField(data=None, wavelengths=wavelengths, spacing=spacing, device=self.device)

        # Waist radius
        if beam_waist_x == None and beam_waist_y == None:
            freqs = LIGHT_SPEED / self.field.wavelengths
            self.beam_waist_x, self.beam_waist_y = self.BeamWaistCorruagtedTK(freqs)
            #self._beam_waist_x = self.check_wavelengths_beam_waist(beam_waist_x)
            #self._beam_waist_y = self.check_wavelengths_beam_waist(beam_waist_y)
        else:
            self.beam_waist_x = torch.tensor([beam_waist_x], device=self.device)
            self.beam_waist_y = torch.tensor([beam_waist_y], device=self.device)

        # (x, y) center position
        self.x0, self.y0 = torch.tensor(center, device=self.device)

        # z-position of the beam waist
        self.z_w0x, self.z_w0y = torch.tensor(z_w0, device=self.device)

        self.alpha = torch.tensor(alpha, device=self.device)

    def BeamWaistCorruagtedTK(self, freqs):
            """_summary_
            Gaussian-beam waist fits to the measured beam patterns by Rasmus
            Args:
                freqs (_type_): frequency range of beam source in 220~330 GHz
            """
            if min(freqs) < 220e9 and max(freqs) > 330e9:
                ValueError('WARNING! Frequency out of range (220-330 GHz)!')
            
            freqs = freqs / 1e9
            p_E = [2.70171433587848e-13,3.10350492358753e-10,-6.35088689290759e-07,0.000322826804965868,-0.0665921902050336,6.08799187520401]
            p_H = [-1.01507121315420e-11,1.70791445624058e-08,-1.12281052414283e-05,0.00360605624858374,-0.564799749943028,35.5588926870041]
            
            beam_waist_x = 1e-3 * (p_E[0] * freqs**5 + p_E[1] * freqs**4 + p_E[2] * freqs**3 + p_E[3] * freqs**2 + p_E[4] * freqs + p_E[5])
            beam_waist_y = 1e-3 * (p_H[0] * freqs**5 + p_H[1] * freqs**4 + p_H[2] * freqs**3 + p_H[3] * freqs**2 + p_H[4] * freqs + p_H[5])
            
            return torch.tensor(beam_waist_x, device=self.device), torch.tensor(beam_waist_y, device=self.device)

    def forward(self)-> ElectricField:
        
        dx, dy = self.field.spacing[0], self.field.spacing[1]

        x = torch.linspace(-dx * self.height / 2, 
                           dx * self.height / 2, 
                           self.height, device=self.device)
        y = torch.linspace(-dy * self.width / 2, 
                           dy * self.width / 2, 
                           self.width, device=self.device)

        X, Y = torch.meshgrid(x, y)
        X = X.unsqueeze(0)  # Add wavelength dimension for broadcasting: [1, height, width]
        Y = Y.unsqueeze(0)

        # check if the beam waist match the wavelength
        # wavenumber
        wavelengths = self.field.wavelengths[:, None, None]
        k = 2 * torch.pi / wavelengths


        if len(self.beam_waist_x) != len(wavelengths) or len(self.beam_waist_y) != len(wavelengths):
            if len(self.beam_waist_x) == 1 and len(self.beam_waist_x) == 1:  # If beam waist is a single number, repeat it for all wavelengths
                self.beam_waist_x = self.beam_waist_x.repeat(len(wavelengths))
                self.beam_waist_y = self.beam_waist_y.repeat(len(wavelengths))
            else:
                raise ValueError('Mismatch between beam waist and wavelength parameters')

        self.beam_waist_x = self.beam_waist_x[:, None, None]
        self.beam_waist_y = self.beam_waist_y[:, None, None]
        # Rayleigh range 
        Rayleigh_x = torch.pi * self.beam_waist_x**2 / wavelengths
        Rayleigh_y = torch.pi * self.beam_waist_y**2 / wavelengths

        # Gouy phase
        Gouy_phase_x = torch.arctan2(self.z_w0x, Rayleigh_x)
        Gouy_phase_y = torch.arctan2(self.z_w0y, Rayleigh_y)

        # spot size (At a position z along the beam)
        w_x = self.beam_waist_x * torch.sqrt(1 + (self.z_w0x / Rayleigh_x) ** 2)
        w_y = self.beam_waist_y * torch.sqrt(1 + (self.z_w0y / Rayleigh_y) ** 2)

        # Radius of curvature
        if self.z_w0x == 0:
            R_x = 1e12
        else:
            R_x = self.z_w0x * (1 + (Rayleigh_x / self.z_w0x) ** 2)
        if self.z_w0y == 0:
            R_y = 1e12
        else:
            R_y = self.z_w0y * (1 + (Rayleigh_y / self.z_w0y) ** 2)

        # Gaussian beam coordinate
        # introduce the rotation of the coordinates by alpha
        x_rot = X * torch.cos(self.alpha) + Y * torch.sin(self.alpha)
        y_rot = -X * torch.sin(self.alpha) + Y * torch.cos(self.alpha)

        # define the phase
        phase = torch.exp(
            -1j * (
                (k * self.z_w0x + k * X**2 / (2 * R_x) - Gouy_phase_x) + (k * self.z_w0y + k * Y**2 / (2 * R_y) - Gouy_phase_y)
                  )
                        )
        # define amplitude
        A = (self.beam_waist_x / w_x) * (self.beam_waist_y / w_y) * torch.exp(
            -(x_rot - self.x0)**2 / (w_x**2) - (y_rot - self.y0)**2 / (w_y**2)
        )

        E = A * phase

        self.field.data = E.unsqueeze(0)

        return self.field



class VectorialGuassian_beam(nn.Module):
    
    """[summary]

    Define a Vectorial Guassian Beam
    Source is 4D - 3 x C x H X W

    Params:
        height (int): resolution of height
        width (int): resolution of width
        beam_waist_x (float): beam waist in x-direction
        beam_waist_y (float): beam waist in y-direction
        jones_vector (float, float): (Ex, Ey) at the origin (r=0, z=0). Doesn't need to be normalized. 
        center (float, float): Position of the center of the beam
        z_w0 (float, float): Position of the waist for (x, y) 
        alpha (float, float): Amplitude rotation (in radians).
        wavelength (float or list): wavelength of source field
        spacing (float or list): physical length of field
    
    """    
    def __init__(self, 
                 height: int,
                 width: int, 
                 beam_waist_x: float,
                 beam_waist_y: float,
                 jones_vector, 
                 center=(0, 0), 
                 z_w0=(0, 0), 
                 alpha=0,
                 wavelengths : torch.Tensor or float = None, 
                 spacing : torch.Tensor or float =None, 
                 device = None):
        
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.height = height
        self.width = width

        self.field = ElectricField(data=None, wavelengths=wavelengths, spacing=spacing, device=self.device)

        # Waist radius
        if beam_waist_x == None and beam_waist_y == None:
            freqs = LIGHT_SPEED / self.field.wavelengths
            self.beam_waist_x, self.beam_waist_y = self.BeamWaistCorruagtedTK(freqs)
            #self._beam_waist_x = self.check_wavelengths_beam_waist(beam_waist_x)
            #self._beam_waist_y = self.check_wavelengths_beam_waist(beam_waist_y)
        else:
            self.beam_waist_x = torch.tensor([beam_waist_x], device=self.device)
            self.beam_waist_y = torch.tensor([beam_waist_y], device=self.device)

        # Normalize Jones vector:
        jones_vector = np.array(jones_vector) / np.linalg.norm(np.array(jones_vector))
        self.jones_vector = torch.tensor(jones_vector, device=self.device)

        # (x, y) center position
        self.x0, self.y0 = torch.tensor(center, device=self.device)

        # z-position of the beam waist
        self.z_w0x, self.z_w0y = torch.tensor(z_w0, device=self.device)

        self.alpha = torch.tensor(alpha, device=self.device)

    def BeamWaistCorruagtedTK(self, freqs):
            """_summary_
            Gaussian-beam waist fits to the measured beam patterns by Rasmus
            Args:
                freqs (_type_): frequency range of beam source in 220~330 GHz
            """
            if min(freqs) < 220e9 and max(freqs) > 330e9:
                ValueError('WARNING! Frequency out of range (220-330 GHz)!')
            
            freqs = freqs / 1e9
            p_E = [2.70171433587848e-13,3.10350492358753e-10,-6.35088689290759e-07,0.000322826804965868,-0.0665921902050336,6.08799187520401]
            p_H = [-1.01507121315420e-11,1.70791445624058e-08,-1.12281052414283e-05,0.00360605624858374,-0.564799749943028,35.5588926870041]
            
            beam_waist_x = 1e-3 * (p_E[0] * freqs**5 + p_E[1] * freqs**4 + p_E[2] * freqs**3 + p_E[3] * freqs**2 + p_E[4] * freqs + p_E[5])
            beam_waist_y = 1e-3 * (p_H[0] * freqs**5 + p_H[1] * freqs**4 + p_H[2] * freqs**3 + p_H[3] * freqs**2 + p_H[4] * freqs + p_H[5])
            
            return torch.tensor(beam_waist_x, device=self.device), torch.tensor(beam_waist_y, device=self.device)

    def forward(self)-> ElectricField:
        
        dx, dy = self.field.spacing[0], self.field.spacing[1]

        x = torch.linspace(-dx * self.height / 2, 
                           dx * self.height / 2, 
                           self.height, device=self.device)
        y = torch.linspace(-dy * self.width / 2, 
                           dy * self.width / 2, 
                           self.width, device=self.device)

        X, Y = torch.meshgrid(x, y)
        X = X.unsqueeze(0)  # Add wavelength dimension for broadcasting: [1, height, width]
        Y = Y.unsqueeze(0)

        # check if the beam waist match the wavelength
        # wavenumber
        wavelengths = self.field.wavelengths[:, None, None]
        k = 2 * torch.pi / wavelengths


        if len(self.beam_waist_x) != len(wavelengths) or len(self.beam_waist_y) != len(wavelengths):
            if len(self.beam_waist_x) == 1 and len(self.beam_waist_x) == 1:  # If beam waist is a single number, repeat it for all wavelengths
                self.beam_waist_x = self.beam_waist_x.repeat(len(wavelengths))
                self.beam_waist_y = self.beam_waist_y.repeat(len(wavelengths))
            else:
                raise ValueError('Mismatch between beam waist and wavelength parameters')

        self.beam_waist_x = self.beam_waist_x[:, None, None]
        self.beam_waist_y = self.beam_waist_y[:, None, None]
        # Rayleigh range 
        Rayleigh_x = torch.pi * self.beam_waist_x**2 / wavelengths
        Rayleigh_y = torch.pi * self.beam_waist_y**2 / wavelengths

        # Gouy phase
        Gouy_phase_x = torch.arctan2(self.z_w0x, Rayleigh_x)
        Gouy_phase_y = torch.arctan2(self.z_w0y, Rayleigh_y)

        # spot size (At a position z along the beam)
        w_x = self.beam_waist_x * torch.sqrt(1 + (self.z_w0x / Rayleigh_x) ** 2)
        w_y = self.beam_waist_y * torch.sqrt(1 + (self.z_w0y / Rayleigh_y) ** 2)

        # Radius of curvature
        if self.z_w0x == 0:
            R_x = 1e12
        else:
            R_x = self.z_w0x * (1 + (Rayleigh_x / self.z_w0x) ** 2)
        if self.z_w0y == 0:
            R_y = 1e12
        else:
            R_y = self.z_w0y * (1 + (Rayleigh_y / self.z_w0y) ** 2)

        # Gaussian beam coordinate
        # introduce the rotation of the coordinates by alpha
        x_rot = X * torch.cos(self.alpha) + Y * torch.sin(self.alpha)
        y_rot = -X * torch.sin(self.alpha) + Y * torch.cos(self.alpha)

        # define the phase
        phase = torch.exp(
            -1j * (
                (k * self.z_w0x + k * X**2 / (2 * R_x) - Gouy_phase_x) + (k * self.z_w0y + k * Y**2 / (2 * R_y) - Gouy_phase_y)
                  )
                        )
        # define amplitude
        Ax = self.jones_vector[0] * (self.beam_waist_x / w_x) * (self.beam_waist_y / w_y) * torch.exp(
            -(x_rot - self.x0)**2 / (w_x**2) - (y_rot - self.y0)**2 / (w_y**2)
        )
        Ay = self.jones_vector[1] * (self.beam_waist_x / w_x) * (self.beam_waist_y / w_y) * torch.exp(
            -(x_rot - self.x0)**2 / (w_x**2) - (y_rot - self.y0)**2 / (w_y**2)
        )

        Ex = Ax * phase
        Ey = Ay * phase
        Ez = torch.zeros_like(Ex)

        vectorialBeam = torch.stack((Ex, Ey, Ez), dim=0)

        self.field.data = vectorialBeam

        return self.field




if __name__=='__main__':
    
    from utils.units import *
    import numpy as np
    import torch
    
    gm = Guassian_beam(height=500, width=500, 
                       beam_waist=1 * mm, wavelengths=[300 * nm, 354 * nm], 
                       spacing=0.2 * mm)
    field = gm()
    print(field.shape)