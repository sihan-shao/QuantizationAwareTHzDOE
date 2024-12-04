from __future__ import annotations
from torch.types import _device, _dtype, _size
import sys
sys.path.append('../')
import torch
import numpy as np
import pathlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm as cmap
from copy import copy
from utils.units import *
from utils.Visualization_Helper import float_to_unit_identifier, add_colorbar
from DataType.ElectricField import ElectricField


eps = 1e-6 # avoid numerical error 

class PolarizationAnalyser():
    
    """[summary]

    tools for analyzing the polarization state of a Vectorial field
    vectorial field is 4D - 3 x C x H X W

    [Ref 1: https://en.wikipedia.org/wiki/Stokes_parameters]

    Params:
        field: vectorial field (ElectricField)
    """    

    def __init__(self, 
                 field,
                 device = None):
        
        super().__init__()
        
        # store the input params
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if field.field_type != 'vectorial':
            raise ValueError("The polarization analyser is only valid for vectorial field")
        self.field = field

    def polarization_state(self, field, numpy=False):
        """
        return the Stokes vector: [S1, S2, S3, S4]
        """
        Ex = field[0, ...].squeeze()
        Ey = field[1, ...].squeeze()

        # [Ref 1: Representations in fixed bases]
        I = torch.abs(Ex)**2 + torch.abs(Ey)**2
        Q = torch.abs(Ex)**2 - torch.abs(Ey)**2
        U = 2 * torch.real(Ex * torch.conj(Ey))
        V = 2 * torch.imag(Ex * torch.conj(Ey))

        # convert torch tensor to numpy array
        if numpy:
            I = torch.Tensor.numpy(I, force=True)
            Q = torch.Tensor.numpy(Q, force=True)
            U = torch.Tensor.numpy(U, force=True)
            V = torch.Tensor.numpy(V, force=True)

        return I, Q, U, V

    def polarization_ellipse(self, field, numpy=True):
        """
        Returns A, B, theta, h polarization parameter of elipses
        """

        I, Q, U, V = self.polarization_state(field, numpy=False)
        
        # [Ref 1: Relation to the polarization ellipse]

        # total polarization intensity (for coherent radiation)
        Ip = torch.sqrt(Q**2 + U**2 + V**2)
        # complex intensity of linear polarization
        L = Q + 1j * U + eps
        
        A = torch.real(torch.sqrt(0.5 * (Ip + torch.abs(L) + eps)))
        B = torch.real(torch.sqrt(0.5 * (Ip - torch.abs(L) + eps)))
        theta = 0.5 * torch.angle(L)
        h = torch.sign(V + eps) 

        if numpy:
            A = torch.Tensor.numpy(A, force=True)
            B = torch.Tensor.numpy(B, force=True)
            theta = torch.Tensor.numpy(theta, force=True)
            h = torch.Tensor.numpy(h, force=True)

        return A, B, theta, h
    
    def visualize_param_stokes(self, wavelength, normalize=False, figsize=(8, 8), cmap=cmap.seismic):
        field = self.field._get_data_for_wavelength(wavelength)
        wavelength = float(wavelength)
        I, Q, U, V = self.polarization_state(field, numpy=True)
        if normalize:
            I = I / (np.abs(I).max() + eps)
            Q = Q / (np.abs(Q).max() + eps)
            U = U / (np.abs(U).max() + eps)
            V = V / (np.abs(V).max() + eps)
        
        intensity_max = I.max()

        if figsize is not None:
            plt.figure(figsize=figsize)

        size_x = np.array(self.field._spacing[0].detach().cpu() / 2.0 * self.field.height)
        size_y = np.array(self.field._spacing[1].detach().cpu() / 2.0 * self.field.width)
        unit_val, unit_axis = float_to_unit_identifier(max(size_x,size_y))
        size_x = size_x / unit_val
        size_y = size_y / unit_val
        extent = [-size_y, size_y, -size_x, size_x]

        unit_val, unit = float_to_unit_identifier(wavelength)
        wavelength = wavelength / unit_val

        plt.subplot(2, 2, 1)
        _im1 = plt.imshow(I, cmap=cmap, extent=extent, vmax=intensity_max, vmin=0)
        plt.title("S1 | wavelength = " + str(round(wavelength, 2)) + str(unit))
        plt.xlabel("Position (" + unit_axis + ")")
        plt.ylabel("Position (" + unit_axis + ")")

        plt.subplot(2, 2, 2)
        _im2 = plt.imshow(Q, cmap=cmap, extent=extent, vmax=intensity_max, vmin=-intensity_max)
        plt.title("S2 | wavelength = " + str(round(wavelength, 2)) + str(unit))
        plt.xlabel("Position (" + unit_axis + ")")
        plt.ylabel("Position (" + unit_axis + ")")

        plt.subplot(2, 2, 3)
        _im3 = plt.imshow(U, cmap=cmap, extent=extent, vmax=intensity_max, vmin=-intensity_max)
        plt.title("S3 | wavelength = " + str(round(wavelength, 2)) + str(unit))
        plt.xlabel("Position (" + unit_axis + ")")
        plt.ylabel("Position (" + unit_axis + ")")

        plt.subplot(2, 2, 4)
        _im4 = plt.imshow(V, cmap=cmap, extent=extent, vmax=intensity_max, vmin=-intensity_max)
        plt.title("S4 | wavelength = " + str(round(wavelength, 2)) + str(unit))
        plt.xlabel("Position (" + unit_axis + ")")
        plt.ylabel("Position (" + unit_axis + ")")

        add_colorbar(_im1)
        add_colorbar(_im2)
        add_colorbar(_im3)
        add_colorbar(_im4)

        plt.axis("on")
        plt.tight_layout()


    def visualize_param_ellipse(self, wavelength, figsize=(8, 8), cmap=cmap.seismic):
        field = self.field._get_data_for_wavelength(wavelength)
        wavelength = float(wavelength)
        A, B, theta, h = self.polarization_ellipse(field, numpy=True)

        intensity_max = max(A.max(), B.max())

        if figsize is not None:
            plt.figure(figsize=figsize)

        size_x = np.array(self.field._spacing[0].detach().cpu() / 2.0 * self.field.height)
        size_y = np.array(self.field._spacing[1].detach().cpu() / 2.0 * self.field.width)
        unit_val, unit_axis = float_to_unit_identifier(max(size_x,size_y))
        size_x = size_x / unit_val
        size_y = size_y / unit_val
        extent = [-size_y, size_y, -size_x, size_x]

        unit_val, unit = float_to_unit_identifier(wavelength)
        wavelength = wavelength / unit_val

        plt.subplot(2, 2, 1)
        _im1 = plt.imshow(A, cmap=cmap, extent=extent, vmax=intensity_max, vmin=0)
        plt.title("A | wavelength = " + str(round(wavelength, 2)) + str(unit))
        plt.xlabel("Position (" + unit_axis + ")")
        plt.ylabel("Position (" + unit_axis + ")")

        plt.subplot(2, 2, 2)
        _im2 = plt.imshow(B, cmap=cmap, extent=extent, vmax=intensity_max, vmin=0)
        plt.title("B | wavelength = " + str(round(wavelength, 2)) + str(unit))
        plt.xlabel("Position (" + unit_axis + ")")
        plt.ylabel("Position (" + unit_axis + ")")

        plt.subplot(2, 2, 3)
        _im3 = plt.imshow(theta, cmap=cmap, extent=extent, vmax=np.pi, vmin=-np.pi)
        plt.title("theta | wavelength = " + str(round(wavelength, 2)) + str(unit))
        plt.xlabel("Position (" + unit_axis + ")")
        plt.ylabel("Position (" + unit_axis + ")")

        plt.subplot(2, 2, 4)
        _im4 = plt.imshow(h, cmap=cmap, extent=extent, vmax=None, vmin=None)
        plt.title("h | wavelength = " + str(round(wavelength, 2)) + str(unit))
        plt.xlabel("Position (" + unit_axis + ")")
        plt.ylabel("Position (" + unit_axis + ")")

        add_colorbar(_im1)
        add_colorbar(_im2)
        add_colorbar(_im3)
        add_colorbar(_im4)

        plt.axis("on")
        plt.tight_layout()

    def visualize_ellipse_field(self, 
                                wavelength, 
                                percentage_intensity=0.005,
                                num_ellipses=(21, 21), 
                                figsize=(8, 8),
                                cm_intensity=cmap.gist_heat, 
                                amplification=0.75,
                                color_line='w',
                                line_width=0.75):

        field = self.field._get_data_for_wavelength(wavelength)
        wavelength = float(wavelength)

        # get three components of vectorial field
        Ex, Ey, Ez = field[0, ...].squeeze(), field[1, ...].squeeze(), field[2, ...].squeeze()

        intensity_max = (torch.abs(Ex)**2 + torch.abs(Ey)**2).max()

        if figsize is not None:
            plt.figure(figsize=figsize)

        size_x = np.array(self.field._spacing[0].detach().cpu() / 2.0 * self.field.height)
        size_y = np.array(self.field._spacing[1].detach().cpu() / 2.0 * self.field.width)
        unit_val, unit_axis = float_to_unit_identifier(max(size_x,size_y))
        size_x = size_x / unit_val
        size_y = size_y / unit_val
        extent = [-size_y, size_y, -size_x, size_x]
        unit_val, unit = float_to_unit_identifier(wavelength)
        wavelength = wavelength / unit_val

        x = np.linspace(-size_x, size_x, self.field.height)
        y = np.linspace(-size_y, size_y, self.field.width)


        Dx, Dy = size_x * 2, size_y * 2
        size_x, size_y = Dx / num_ellipses[0], Dx / num_ellipses[1]
        x_centers = size_x / 2 + size_x * np.array(range(0, num_ellipses[0]))
        y_centers = size_y / 2 + size_y * np.array(range(0, num_ellipses[1]))

        ix_centers = self.field.height / (num_ellipses[0])
        iy_centers = self.field.width / (num_ellipses[1])

        ix_centers = (np.round(
            ix_centers / 2 +
            ix_centers * np.array(range(0, num_ellipses[0])))).astype('int')
        iy_centers = (np.round(
            iy_centers / 2 +
            iy_centers * np.array(range(0, num_ellipses[1])))).astype('int')
        
        # Clip the indices to ensure they are within the bounds of the field dimensions
        ix_centers = np.clip(ix_centers, 0, Ex.shape[-1] - 1)
        iy_centers = np.clip(iy_centers, 0, Ey.shape[-2] - 1)

        Ix_centers, Iy_centers = np.meshgrid(ix_centers.astype('int'),
                                             iy_centers.astype('int'))
        
        E0x = Ex[Iy_centers, Ix_centers]
        E0y = Ey[Iy_centers, Ix_centers]

        angles = np.linspace(0, 360 * np.pi / 180, 64)

        # plot the amplitude of total electric field as background of polarization ellipse
        I = torch.abs(Ex)**2 + torch.abs(Ey)**2 + torch.abs(Ey)**2
        _im = plt.imshow(I.detach().cpu().numpy(), cmap=cm_intensity, extent=extent, vmax=torch.max(I), vmin=torch.min(I))
        plt.title("Intensity | wavelength = " + str(round(wavelength, 2)) + str(unit))
        plt.xlabel("Position (" + unit_axis + ")")
        plt.ylabel("Position (" + unit_axis + ")")
        ax = plt.gca()

        for i, xi in enumerate(ix_centers):
            for j, yj in enumerate(iy_centers):
                Ex_plot = np.real(E0x[j, i].detach().cpu().numpy() * np.exp(1j * angles))
                Ey_plot = np.real(E0y[j, i].detach().cpu().numpy() * np.exp(1j * angles))


                max_r = np.sqrt(np.abs(Ex_plot)**2 + np.abs(Ey_plot)**2).max()
                size_dim = min(size_x, size_y)

                if max_r > 0 and max_r**2 > percentage_intensity * intensity_max:
                    
                    # Ensure xi and yj are treated as scalars
    
                    xi_scalar = int(xi)
                    yj_scalar = int(yj)


                    Ex_plot = Ex_plot / max_r * size_dim * amplification / 2 + x[xi_scalar]
                    Ey_plot = Ey_plot / max_r * size_dim * amplification / 2 + y[yj_scalar]

                    ax.plot(Ex_plot, Ey_plot, color_line, lw=line_width)

                    ax.arrow(Ex_plot[0],
                             Ey_plot[0],
                             Ex_plot[0] - Ex_plot[1],
                             Ey_plot[0] - Ey_plot[1],
                             width=0,
                             head_width=1,
                             fc=color_line,
                             ec=color_line,
                             length_includes_head=False)

        plt.show()

    def analyze(self, 
                wavelength, 
                type='ellipse_field', 
                percentage_intensity=0.005,
                figsize=(8, 8), 
                num_ellipses=(21, 21), 
                color_line='w',
                line_width=0.75):

        if type == 'stokes':
            self.visualize_param_stokes(wavelength, 
                                        figsize=figsize)

        elif type == 'param_ellipse':
            self.visualize_param_ellipse(wavelength, 
                                         figsize=figsize)

        elif type == 'ellipse_field':
            self.visualize_ellipse_field(wavelength, 
                                         figsize=figsize, 
                                         percentage_intensity=percentage_intensity, 
                                         num_ellipses=num_ellipses, 
                                         color_line=color_line, 
                                         line_width=line_width)

        else:
            raise ValueError("No existing analysis type in vector field")