import sys
sys.path.append('./')
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import pad
from DataType.ElectricField import ElectricField
from torch.fft import fft2, ifft2


class CZT_prop(nn.Module):
    
    def __init__(self, 
                 z_distance         : float = 0.0, 
                 device             : str = None
                 ) -> None:
        """
        Chirped z-transform propagation - efficient diffraction using the Bluestein method.
        Useful for imaging light in the focal plane: allows high resolution zoom in z-plane. 
        [Ref 1] Hu, Y., et al. Light Sci Appl 9, 119 (2020).
        
        Args:
            z_distance (float, optional): propagation distance along the z-direction. Defaults to 0.0.
        """
        super().__init__()

        # store the input params
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._z = torch.tensor(z_distance, device=self.device)
    
    @property
    def z(self) -> torch.Tensor:
        return self._z
    
    @z.setter
    def z(self, z : torch.Tensor) -> None:
        if not isinstance(z, torch.Tensor):
            value = torch.tensor(z, device=self.device)
        elif z.device != self.device:
            z = z.to(self.device)
        self._z = z

    def RS_kernel(self, z, meshx, meshy, wavelengths):
        """
        function for RS transfer function
        """
        
        wavelengths_expand  =  wavelengths[None, :, None, None]
        k = 2 * torch.pi / wavelengths_expand

        r = torch.sqrt(meshx**2 + meshy**2 + z**2)
        factor = 1 / (2 * torch.pi) * z / r**2 * (1 / r - 1j*k)
        kernel = torch.exp(1j * k * r) * factor
        # Do we need to check the minimum propagation distance here? 

        return kernel
    
    def build_CZT_grid(self, 
                       z, 
                       wavelengths,
                       InputHeight, 
                       InputWidth,
                       InputPixel_dx, 
                       InputPixel_dy,
                       outputHeight,
				       outputWidth, 						
				       outputPixel_dx, 
				       outputPixel_dy):
        """
        [From CZT]: Defines the resolution / sampling of initial and output planes.
        
        Parameters:
            z, 
            InputHeight
            InputWidth
            InputPixel_dx,
            InputPixel_dy
            outputHeight
		    outputWidth					
		    outputPixel_dx
		    outputPixel_dy
    
        Returns the set of parameters: 
            Inmeshx: 
            Inmeshy: 
            Outmeshx: 
            Outmeshy: 
            Dm:         dimension of the output ﬁeld
            fy_1:       Starting point along y-direction in frequency range.
            fy_2:       End point along y-direction in frequency range.
            fx_1:       Starting point along x-direction in frequency range.
            fx_2:       End point along x-direction in frequency range.
        """
        # create grid for input plane
        x_in = torch.linspace(-InputHeight * InputPixel_dx / 2, InputHeight * InputPixel_dx / 2, InputHeight, device=self.device)
        y_in = torch.linspace(-InputWidth * InputPixel_dy / 2, InputWidth * InputPixel_dy / 2, InputWidth, device=self.device)
        Inmeshx, Inmeshy = torch.meshgrid(x_in, y_in)

        # create grid for output plane
        x_out = torch.linspace(-outputHeight * outputPixel_dx / 2, outputHeight * outputPixel_dx / 2, outputHeight, device=self.device)
        y_out = torch.linspace(-outputWidth * outputPixel_dy / 2, outputWidth * outputPixel_dy / 2, outputWidth, device=self.device)
        Outmeshx, Outmeshy = torch.meshgrid(x_out, y_out)

        wavelengths_expand = wavelengths[None, :, None, None]  # reshape to [1, C, :, :] for broadcasting

        # For Bluestein method implementation: 
        # Dimension of the output field - Eq. (11) in [Ref].
        Dm = wavelengths_expand * z / InputPixel_dx

        # (1) for FFT in X-dimension:
        fx_1 = x_out[0] + Dm / 2
        fx_2 = x_out[-1] + Dm / 2
        # (1) for FFT in Y-dimension:
        fy_1 = y_out[0] + Dm / 2
        fy_2 = y_out[-1] + Dm / 2

        return Inmeshx, Inmeshy, Outmeshx, Outmeshy, Dm, fx_1, fx_2, fy_1, fy_2

    def compute_np2(self, x):
        """
        [For Bluestein method]: Exponent of next higher power of 2. 

        Parameters:
        x (float): value

        Returns the exponent for the smallest powers of two that satisfy 2**p >= X for each element in X.
        This function is useful for optimizing FFT operations, which are most efficient when sequence length is an exact power of two.
        """
        return 2**(np.ceil(np.log2(x))).astype(int)

    def compute_fft(self, x, D1, D2, Dm, m, n, mp, M_out, np2):
        """
        [From Bluestein_method]: the FFT part of the algorithm. 
        Parameters:
            x  (float)  : signal
            D1 (float)  : start intermeidate frequency
            D2 (float)  : end intermediate frequency
            Dm (float)  : dimension of the imaging plane.
            m     (int) : original x dimension of signal x
            n     (int) : original y dimension of signal x
            mp    (int) : length of output sequence needed
            M_out (int) : the length of the chirp z-transform of signal x
            np2   (int) : length of the output sequence for efficient FFT computation (exact power of two)

        [Ref] : https://www.osti.gov/servlets/purl/1004350
        """
        # A-Complex exponential term
        A = torch.exp(1j * 2 * torch.pi * D1 / Dm)
        # W-complex exponential term
        W = torch.exp(-1j * 2 * torch.pi * (D1 - D2) / (M_out * Dm))

        # window function (Premultiply data)
        h = torch.arange(-m + 1, max(M_out - 1, m - 1 ) + 1, device=self.device)
        h = W**(h**2 / 2) 
        #print(w.shape)
        h_sliced = h[:mp + 1]
        #print(w_sliced.shape)

        # Compute the 1D Fourier Transform of 1/h up to length 2**nextpow2(mp)
        ft = torch.fft.fft(1 / h_sliced, n=np2, dim=-1) # FFT for Chirp filter [Ref Eq.10 last term]
        #print(ft_w.shape)
        # Compute intermediate result for Bluestein's algorithm [Ref Eq.10 third term]
        b = A**(-(torch.arange(0, m, device=self.device))) * h[..., torch.arange(m - 1, 2 * m - 1, device=self.device)]
        #print(AW.shape)  # torch.Size([1, 28, 1, 200])
        tmp = torch.tile(b, (1, 1, n, 1)).transpose(-2, -1)
        print(tmp.shape)
        # Compute the 1D Fourier transform of input data
        print(x.shape)
        b = torch.fft.fft(x * tmp, np2, dim=-2)
        print(b.shape)
        # Compute the inverse Fourier transform
        s = torch.tile(ft, (1, 1, n, 1)).transpose(-2, -1)
        print(s.shape)
        b = torch.fft.ifft(b * torch.tile(ft, (1, 1, n, 1)).transpose(-2, -1), dim=-2)
        print(b.shape)
        return b, h

    def Bluestein_method(self, x, f1, f2, Dm, M_out):
        """
        [From CZT]: Performs the DFT using Bluestein method. 
        [Ref1]: Hu, Y., et al. Light Sci Appl 9, 119 (2020).
        [Ref2]: L. Bluestein, IEEE Trans. Au. and Electro., 18(4), 451-455 (1970).
        [Ref3]: L. Rabiner, et. al., IEEE Trans. Au. and Electro., 17(2), 86-92 (1969).
        [Ref4]: https://www.osti.gov/servlets/purl/1004350
    
        Parameters:
            x (jnp.array): Input sequence, x[n] in Eq.(12) in [Ref 1].
            f1 (float): Starting point in frequency range.
            f2 (float): End point in frequency range. 
            Dm (float): Dimension of the imaging plane.
            M_out (float): Length of the transform (resolution of the output plane).
    
        Returns the output X[m].
        """

        # Correspond to the length of the input sequence.  
        _, _, m, n =x.shape
        # intermediate frequency  [1, C, 1, 1]
        D1 = f1 + (M_out * Dm + f2 - f1) / (2 * M_out)   # D1 refer to f1 in [Ref1 Eq.S15]
        # Upper frequency limit
        D2 = f2 + (M_out * Dm + f2 - f1) / (2 * M_out)   # D2 refer to f2 in [Ref1 Eq.S15]

        # Length of the output sequence
        mp = m + M_out - 1
        np2 = self.compute_np2(mp)   # FFT is more efficient when sequence length is an exact power of two.
        b, h = self.compute_fft(x, D1, D2, Dm, m, n, mp, M_out, np2)
        #print(b.shape, w.shape)

        # Extract the relevant portion and multiply by the window function [Ref4 Eq.10 first term]
        b = b[..., m:mp + 1, 0:n].transpose(-2, -1) * torch.tile(h[..., m - 1:mp], (1, 1, n, 1))
        #print(b.shape)
        # create a linearly speed array from 0 to M_out-1
        l = torch.linspace(0, M_out-1, M_out, device=self.device)[None, None, None, :]
        # scale array to the frequency range [D1, D2]
        l = l / M_out * (D2 - D1) + D1  
        print(l)

        # Eq. S14 in Supplementaty Information Section 3 in [Ref1]. Frequency shift to center the spectrum.
        M_shift = -m / 2
        M_shift = torch.tile(torch.exp(-1j * 2 * torch.pi * l * (M_shift + 1 / 2) / Dm), (1, 1, n, 1))
        #print(M_shift)
        # Apply the frequency shift to the final output
        b = b * M_shift
        return b

    def CZT(self, field, z, wavelengths, outputHeight, outputWidth, outputdx, outputdy, outputmeshx, outputmeshy, inputmeshx, inputmeshy, Dm, fx_1, fx_2, fy_1, fy_2):
        """
        [From CZT]: Diffraction integral implementation using Bluestein method.
        [Ref] Hu, Y., et al. Light Sci Appl 9, 119 (2020).
        """  

        # compute the RS transfer function for input and output plane
        F0 = self.RS_kernel(z, outputmeshx, outputmeshy, wavelengths)  # kernel shape should be [1, wavelength, H, W]
        F  = self.RS_kernel(z, inputmeshx, inputmeshy, wavelengths)
    
        # Compute (E0 x F) in Eq.(6) in [Ref].
        field = field.data * F

        # Bluestein method implementation:

        # (1) FFT in Y-dimension:
        U = self.Bluestein_method(field, fy_1, fy_2, Dm, outputWidth)

        # (2) FFT in X-dimension using output from (1):
        U = self.Bluestein_method(U, fx_1, fx_2, Dm, outputHeight)

        field = F0 * U * z * outputdx * outputdy * wavelengths[None, :, None, None]

        return field

    def forward(self, 
                field: ElectricField, 
                outputHeight=None,
				outputWidth=None, 						
				outputPixel_dx=None, 
				outputPixel_dy=None,					
                ) -> ElectricField:
        """
        Chirped z-transform propagation - efficient diffraction using the Bluestein method.
        Useful for imaging light in the focal plane: allows high resolution zoom in z-plane. 
        [Ref] Hu, Y., et al. Light Sci Appl 9, 119 (2020).

        Parameters:
            field (ElectricField): Complex field 4D tensor object
            outputHeight (int): resolution of height for the output plane.
            outputWidth (int): resolution of width the output plane.
            outputPixel_dx (float): physical length (height) for the output plane
            outputPixel_dy (float): physical length (width) for the putput plane

        Returns ScalarLight object after propagation.
        """

        InputHeight = field.height
        InputWidth = field.width
        InputPixel_dx = field.spacing[0]
        InputPixel_dy = field.spacing[1]
        wavelengths = field.wavelengths

        # Set default values for outputHeight and outputPixel_dx if they are None
        if outputHeight is None:
            outputHeight = InputHeight
        if outputPixel_dx is None:
            outputPixel_dx = InputPixel_dx

        # Set default values for outputWidth and outputPixel_dy if they are None
        if outputWidth is None:
            outputWidth = InputWidth
        if outputPixel_dy is None:
            outputPixel_dy = InputPixel_dy

        Inmeshx, Inmeshy, Outmeshx, Outmeshy, Dm, fx_1, fx_2, fy_1, fy_2 = self.build_CZT_grid(self._z, wavelengths,
                                                                                            InputHeight, InputWidth, InputPixel_dx, InputPixel_dy, 
                                                                                            outputHeight, outputWidth, outputPixel_dx, outputPixel_dy)

        # Compute the diffraction integral using Bluestein method
        
        field_out = self.CZT(field, self._z, wavelengths, 
                            outputHeight, outputWidth, 
                            outputPixel_dx, outputPixel_dy, 
                            Outmeshx, Outmeshy, 
                            Inmeshx, Inmeshy, 
                            Dm, 
                            fx_1, fx_2, 
                            fy_1, fy_2)


        Eout = ElectricField(
				data=field_out,
				wavelengths=wavelengths,
				spacing=[outputPixel_dx, outputPixel_dy]
				)

        return Eout


class VCZT_prop(CZT_prop):
    
    def __init__(self, 
                 z_distance         : float = 0.0, 
                 device             : str = None
                 ) -> None:
        """
        Chirped z-transform propagation - efficient diffraction using the Bluestein method.
        Useful for imaging light in the focal plane: allows high resolution zoom in z-plane. 
        [Ref 1] Hu, Y., et al. Light Sci Appl 9, 119 (2020).
        
        Args:
            z_distance (float, optional): propagation distance along the z-direction. Defaults to 0.0.
        """
        super().__init__()

        # store the input params
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._z                  = torch.tensor(z_distance, device=self.device)
    
    @property
    def z(self) -> torch.Tensor:
        return self._z
    
    @z.setter
    def z(self, z : torch.Tensor) -> None:
        if not isinstance(z, torch.Tensor):
            value = torch.tensor(z, device=self.device)
        elif z.device != self.device:
            z = z.to(self.device)
        self._z = z