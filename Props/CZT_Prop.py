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

    def RS_kernel(self, z, meshx, meshy, wavelengths):
        """
        function for RS transfer function
        """
        # TODO: modify it into mutiple-wavelength process [Done]
        wavelengths_expand  =  wavelengths[:,None,None]
        k = 2 * torch.pi / wavelengths_expand

        r = torch.sqrt(meshx**2 + meshy**2 + z**2)
        factor = 1 / (2 * torch.pi) * z / r**2 * (1 / r - 1j*k)
        kernel = torch.exp(1j * k * r) * factor
        return kernel
    
    def build_CZT_grid(self, 
                       field, 
                       z, 
                       wavelengths, 
                       outputHeight,
				       outputWidth, 						
				       outputPixel_dx, 
				       outputPixel_dy):
        """
        [From CZT]: Defines the resolution / sampling of initial and output planes.
        
        Parameters:
        xin (jnp.array): Array with the x-positions of the input plane.
        yin (jnp.array): Array with the y-positions of the input plane.
        xout (jnp.array): Array with the x-positions of the output plane.
        yout (jnp.array): Array with the y-positions of the output plane.
    
        Returns the set of parameters: nx, ny, Xout, Yout, dx, dy, delta_out, Dm, fy_1, fy_2, fx_1 and fx_2.
        """
        pass

    def compute_np2(self, x):
        """
        [For Bluestein method]: Exponent of next higher power of 2. 

        Parameters:
        x (float): value

        Returns the exponent for the smallest powers of two that satisfy 2**p >= X for each element in X.
        This function is useful for optimizing FFT operations, which are most efficient when sequence length is an exact power of two.
        """
        np2 = int(2**(torch.ceil(torch.log2(x))))
        return np2

    def compute_fft(self, x, D1, D2, Dm, m, n, mp, M_out, np2):
        """
        [From Bluestein_method]: JIT-computes the FFT part of the algorithm. 
        Parameters:
            x  (float)  : signal
            D1 (float)  : start intermeidate frequency
            D2 (float)  : end intermediate frequency
            Dm (float)  : dimension of the imaging plane.
            m     (int) : original x dimension of signal x
            n     (int) : original y dimension of signal x
            mp    (int) : length of output sequence needed (corresponding to np2)
            M_out (int) : the length of the chirp z-transform of signal x
            np2   (int) : length of the output sequence for efficient FFT computation (exact power of two)

        [Ref] : https://www.osti.gov/servlets/purl/1004350
        """
        # A-Complex exponential term
        A = torch.exp(1j * 2 * torch.pi * D1 / Dm)
        # W-complex exponential term
        W = torch.exp(-1j * 2 * torch.pi * (D1 - D2) / (M_out * Dm))

        # window function (Premultiply data)
        h = torch.arange(-m + 1, max(M_out - 1, m - 1) + 1)
        h2 = h**2 / 2
        w = W**h2 
        w_sliced = w[:mp + 1]

        # Compute the 1D Fourier Transform of 1/h up to length 2**nextpow2(mp)
        ft_w = torch.fft.fft(1 / w_sliced, np2) # FFT for Chirp filter [Ref Eq.10 last term]
        # Compute intermediate result for Bluestein's algorithm [Ref Eq.10 third term]
        AW = A**(-torch.arange(m)) * w[torch.arange(m - 1, 2 * m - 1)]
        tmp = torch.tile(AW, (n, 1)).T
        # Compute the 1D Fourier transform of input data
        b = torch.fft.fft(x * tmp, np2, axis=0)
        # Compute the inverse Fourier transform
        b = torch.fft.fft(b * torch.tile(ft2, (n, 1)).T, axis=0)

        return b, w

    def Bluestein_method(x, f1, f2, Dm, M_out):
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
        m, n =x.shape

        # intermediate frequency
        D1 = f1 + (M_out * Dm + f2 - f1) / (2 * M_out)   # D1 refer to f1 in [Ref1 Eq.S15]
        # Upper frequency limit
        D2 = f2 + (M_out * Dm + f2 - f1) / (2 * M_out)   # D2 refer to f2 in [Ref1 Eq.S15]

        # Length of the output sequence
        mp = m + M_out - 1
        np2 = self.compute_np2(mp)   # FFT is more efficient when sequence length is an exact power of two.
        b, w = self.compute_fft(x, D1, D2, Dm, m, n, mp, M_out, np2)

        # Extract the relevant portion and multiply by the window function [Ref4 Eq.10 first term]
        b = b[m:mp + 1, 0:n].T * torch.tile(w[m - 1:mp], (n, 1))
        
        

        pass

    def CZT(self, field, z, wavelengths, outputHeight, outputWidth, outputdx, outputdy, outputmeshx, outputmeshy, inputmeshx, inputmeshy, Dm, fx_1, fx_2, fy_1, fy_2):
        """
        [From CZT]: Diffraction integral implementation using Bluestein method.
        [Ref] Hu, Y., et al. Light Sci Appl 9, 119 (2020).
        """  

        # compute the RS transfer function for input and output plane
        F0 = self.RS_kernel(z, outputmeshx, outputmeshy, wavelengths)  # kernel shape should be [1, wavelength, H, W]
        F  = self.RS_kernel(z, inputmeshx, inputmeshy, wavelengths)
        
        # compute the intermedaite results
        field = field.data * F

        # Bluestein method implementation:

        # (1) FFT in Y-dimension:
        U = self.Bluestein_method(field, fy_1, fy_2, Dm, outputWidth)

        # (2) FFT in X-dimension using output from (1):
        U = self.Bluestein_method(U, fx_1, fx_2, Dm, outputHeight)

        field = F0 * U * z * outputdx * outputdy * wavelengths

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

        if outputHeight and outputPixel_dx is None:
            outputHeight = field.shape[-2]
            outputPixel_dx = field.spacing[0]
        
        if outputWidth and outputPixel_dy is None:
            outputWidth = field.shape[-1]
            outputPixel_dy = field.spacing[1]

        # Defines the resolution / sampling of initial and output planes.
        # parameters of the output plane:
        x_out = torch.linspace(-int(outputHeight) * outputPixel_dx / 2, int(outputHeight) * outputPixel_dx / 2, int(outputHeight))
        y_out = torch.linspace(-int(outputWidth) * outputPixel_dy / 2, int(outputWidth) * outputPixel_dy / 2, int(outputWidth))
        meshx_out, meshy_out = torch.meshgrid(x_out, y_out, indexing='ij').to(self.device)
        # parameters of the input plane:
        wavelengths = field.wavelengths  
        dx_in, dy_in = field.spacing[0], field.spacing[1]
        # For Bluestein method implementation: 
        # Dimension of the output field - Eq. (11) in [Ref].
        Dm = wavelengths * z / dx_in

        # (1) for FFT in X-dimension:
        fx_1 = x_out[0] + Dm / 2
        fx_2 = x_out[-1] + Dm / 2
        # (1) for FFT in Y-dimension:
        fy_1 = y_out[0] + Dm / 2
        fy_2 = y_out[-1] + Dm / 2


        pass


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