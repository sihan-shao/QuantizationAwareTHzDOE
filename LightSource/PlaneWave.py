import sys
sys.path.append('../')
import torch
import torch.nn as nn
from utils.units import *
from DataType.ElectricField import ElectricField

class PlaneWave(nn.Module):
    def __init__(self, 
                 height: int,
                 width: int, 
                 A: float = 1.0,
                 theta: float = 0.0,
                 phi: float = 0.0,
                 z0: float = 0.0,
                 wavelengths : torch.Tensor or float = None, 
                 spacing : torch.Tensor or float = None, 
                 device = None):
        """
        Defines a scalar plane wave.

        Parameters:
            height (int): Resolution in height.
            width (int): Resolution in width.
            A (float): Amplitude of the plane wave.
            theta (float): Angle in radians.
            phi (float): Angle in radians.
            z0 (float): Phase shift.
            wavelengths (torch.Tensor or float): Wavelength(s) of the plane wave.
            spacing (torch.Tensor or float): Physical size of the grid.
            device: Torch device to use.

        Returns:
            ElectricField object containing the plane wave field.
        """
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.height = height
        self.width = width

        self.A = A
        self.theta = torch.tensor(theta, device=self.device)
        self.phi = torch.tensor(phi, device=self.device)
        self.z0 = torch.tensor(z0, device=self.device)

        self.field = ElectricField(data=None, wavelengths=wavelengths, spacing=spacing, device=self.device)

    def forward(self) -> ElectricField:
        dx, dy = self.field.spacing[0], self.field.spacing[1]

        x = torch.linspace(-dx * self.width / 2, dx * self.width / 2, self.width, device=self.device)
        y = torch.linspace(-dy * self.height / 2, dy * self.height / 2, self.height, device=self.device)

        X, Y = torch.meshgrid(x, y, indexing='ij')
        X = X.unsqueeze(0)  # Shape [1, H, W]
        Y = Y.unsqueeze(0)

        wavelengths = self.field.wavelengths[:, None, None]  # Shape [C, 1, 1]
        k = 2 * torch.pi / wavelengths  # Shape [C, 1, 1]

        theta = self.theta
        phi = self.phi
        z0 = self.z0
        A = self.A

        # Compute the exponent argument
        exp_arg = 1j * k * (X * torch.sin(theta) * torch.cos(phi) +
                            Y * torch.sin(theta) * torch.sin(phi) +
                            z0 * torch.cos(theta))  # Shape [C, H, W]

        field = A * torch.exp(exp_arg)  # Shape [C, H, W]

        self.field.data = field.unsqueeze(0)  # Shape [1, C, H, W]

        return self.field


class VectorialPlaneWave(nn.Module):
    def __init__(self, 
                 height: int,
                 width: int, 
                 jones_vector, 
                 theta: float = 0.0,
                 phi: float = 0.0,
                 z0: float = 0.0,
                 wavelengths : torch.Tensor or float = None, 
                 spacing : torch.Tensor or float = None, 
                 device = None):
        """
        Defines a vectorial plane wave.

        Parameters:
            height (int): Resolution in height.
            width (int): Resolution in width.
            jones_vector (tuple or list): (Ex, Ey) components at the origin.
            theta (float): Angle in radians.
            phi (float): Angle in radians.
            z0 (float): Phase shift.
            wavelengths (torch.Tensor or float): Wavelength(s) of the plane wave.
            spacing (torch.Tensor or float): Physical size of the grid.
            device: Torch device to use.

        Returns:
            ElectricField object containing the vectorial plane wave field.
        """
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.height = height
        self.width = width

        # Normalize Jones vector
        jones_vector = torch.tensor(jones_vector, device=self.device, dtype=torch.complex64)
        jones_vector = jones_vector / torch.linalg.norm(jones_vector)
        self.jones_vector = jones_vector

        self.theta = torch.tensor(theta, device=self.device)
        self.phi = torch.tensor(phi, device=self.device)
        self.z0 = torch.tensor(z0, device=self.device)

        self.field = ElectricField(data=None, wavelengths=wavelengths, spacing=spacing, device=self.device)

    def forward(self) -> ElectricField:
        dx, dy = self.field.spacing[0], self.field.spacing[1]

        x = torch.linspace(-dx * self.width / 2, dx * self.width / 2, self.width, device=self.device)
        y = torch.linspace(-dy * self.height / 2, dy * self.height / 2, self.height, device=self.device)

        X, Y = torch.meshgrid(x, y, indexing='ij')
        X = X.unsqueeze(0)  # Shape [1, H, W]
        Y = Y.unsqueeze(0)

        wavelengths = self.field.wavelengths[:, None, None]  # Shape [C, 1, 1]
        k = 2 * torch.pi / wavelengths  # Shape [C, 1, 1]

        theta = self.theta
        phi = self.phi
        z0 = self.z0

        # Compute the exponent argument
        exp_arg = 1j * k * (X * torch.sin(theta) * torch.cos(phi) +
                            Y * torch.sin(theta) * torch.sin(phi) +
                            z0 * torch.cos(theta))  # Shape [C, H, W]

        pw = torch.exp(exp_arg)  # Shape [C, H, W]

        # Ex and Ey components
        Ex = self.jones_vector[0][None, None, None] * pw  # Shape [C, H, W]
        Ey = self.jones_vector[1][None, None, None] * pw  # Shape [C, H, W]
        Ez = torch.zeros_like(Ex)  # Shape [C, H, W]

        field = torch.stack((Ex, Ey, Ez), dim=0)  # Shape [3, C, H, W]

        self.field.data = field

        return self.field