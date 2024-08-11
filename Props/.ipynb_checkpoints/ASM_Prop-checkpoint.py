import sys
sys.path.append('./')
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.nn.functional import pad
import matplotlib.pyplot as plts
import matplotlib
import matplotlib.pyplot as plt

from DataType.ElectricField import ElectricField
from utils.Helper_Functions import ft2, ift2


class ASM_prop(nn.Module):
    
    def __init__(self, 
                 z_distance         : float = 0.0, 
                 do_padding         : bool = True,
                 do_unpad_after_pad : bool = True, 
                 padding_scale      : float or torch.Tensor = None, 
                 bandlimit_kernel   : bool = True, 
                 bandlimit_type     : str = 'exact', 
                 device             : str = None
                 ) -> None:
        """
        Angular Spectrum method with bandlimited ASM from Digital Holographic Microscopy
        Principles, Techniques, and Applications by K. Kim 
        Eq. 4.22 (page 50)
        
        Args:
            init_distance (float, optional): initial propagation distance. Defaults to 0.0.
            z_opt (bool, optional): is the distance parameter optimizable or not. Defaults to False
            do_padding (bool, optional):	Determines whether or not to pad the input field data before doing calculations.
											Padding can help reduce convolution edge artifacts, but will increase the size of the data processed.
											Defaults to True.
           o_unpad_after_pad (bool, optional):	This determines whether or not to unpad the field data before returning an ElectricField object.
													If 'do_padding' is set to False, 'do_unpad_after_pad' has no effect
													Otherwise:
														- If 'do_unpad_after_pad' is set to True, then the field data is unpadded to its original size, i.e. the size of the input field's data.
														- If 'do_unpad_after_pad' is set to False, then no unpadding is done.  The field data returned will be of the padded size.
													Defaults to True.

			padding_scale (float, tuple, tensor; optional):		Determines how much padding to apply to the input field.
																Padding is applied symmetrically so the data is centered in the height and width dimensions.
																'padding_scale' must be a non-negative real-valued number, a 2-tuple containing non-negative real-valued numbers, or a tensor containing two non-negative real-valued numbers.

																Examples:
																	Example 1:
																		- Input field dimensions: height=50, width=100
																		- padding_scale = 1
																		- Padded field dimensions: height=100, width=200	<--- (50 + 1*50, 100 + 1*100)
																	Example 1:
																		- Input field dimensions: height=50, width=100
																		- padding_scale = torch.tensor([1,2])
																		- Padded field dimensions: height=100, width=300	<--- (50 + 1*50, 100 + 2*100)
            
            bandlimit_kernel (bool, optional):	Determines whether or not to apply the bandlimiting described in Band-Limited ASM (Matsushima et al, 2009) to the ASM kernel
													- bandlimit_kernel = True will apply the bandlimiting, bandlimit_kernel = False will not apply the bandlimiting
												Note that evanescent wave components will be filtered out regardless of what this is set to.
												Defaults to True

			bandlimit_type (str, optional):		If bandlimit_kernel is set to False, then this option does nothing.
												If bandlimit_kernel is set to True, then:
													'approx' - Bandlimits the propagation kernel based on Equations 21 and 22 in Band-Limited ASM (Matsushima et al, 2009)
													'exact' - Bandlimits the propagation kernel based on Equations 18 and 19 in Band-Limited ASM (Matsushima et al, 2009)
												Note that for aperture sizes that are small compared to the propagation distance, 'approx' and 'exact' will more-or-less the same results.
												Defaults to 'exact'.
			
        """
        super().__init__()

        DEFAULT_PADDING_SCALE = torch.tensor([1,1])
        if do_padding:
            paddingScaleErrorFlag = False
            if not torch.is_tensor(padding_scale):
                if padding_scale == None:
                    padding_scale = DEFAULT_PADDING_SCALE
                elif np.isscalar(padding_scale):
                    padding_scale = torch.tensor([padding_scale, padding_scale])
                else:
                    padding_scale = torch.tensor(padding_scale)
                    if padding_scale.numel() != 2:
                        paddingScaleErrorFlag = True
            elif padding_scale.numel() == 1:
                padding_scale = padding_scale.squeeze()
                padding_scale = torch.tensor([padding_scale, padding_scale])
            elif padding_scale.numel() == 2:
                padding_scale = padding_scale.squeeze()
            else:
                paddingScaleErrorFlag = True
			
            if (paddingScaleErrorFlag):
                raise Exception("Invalid value for argument 'padding_scale'.  Should be a real-valued non-negative scalar number or a two-element tuple/tensor containing real-valued non-negative scalar numbers.")
        else:
            padding_scale = None
            
        # store the input params
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._z                  = torch.tensor(z_distance, device=self.device)
        self.do_padding         = do_padding
        self.do_unpad_after_pad = do_unpad_after_pad
        self.padding_scale      = padding_scale
        self.bandlimit_kernel   = bandlimit_kernel
        self.bandlimit_type     = bandlimit_type
        
        # the normalized frequency grid
		# we don't actually know dimensions until forward is called
        self.Kx = None
        self.Ky = None

        # initialize the shape
        self.shape = None
        self.check_Zc = True
    
    def compute_padding(self, H, W, return_size_of_padding = False):
        
        # get the shape of processing
        if not self.do_padding:
            paddingH = 0
            paddingW = 0
            paddedH = int(H)
            paddedW = int(W)
        else:
            paddingH = int(np.floor((self.padding_scale[0] * H) / 2))
            paddingW = int(np.floor((self.padding_scale[1] * W) / 2))
            paddedH = H + 2*paddingH
            paddedW = W + 2*paddingW
        
        if not return_size_of_padding:
            return paddedH, paddedW
        else:
            return paddingH, paddingW
        
    def create_frequency_grid(self, H, W):
        # precompute the frequency grid for ASM kernel
        with torch.no_grad():
            # creates the frequency coordinate grid in x and y direction
            kx = (torch.linspace(0, H - 1, H) - (H // 2)) / H
            ky = (torch.linspace(0, W - 1, W) - (W // 2)) / W
            
            self.Kx, self.Ky = torch.meshgrid(kx, ky)
    
    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        if (shape is None):
            self._shape = None
            return
        try:
            _,_,H_new,W_new = shape
            if (self.shape is None):
                self._shape  = shape
                self.create_frequency_grid(H_new,W_new)
            else:
                _,_,H_old,W_old = self.shape
                self._shape  = shape
                if H_old != H_new or W_old != W_new:
                    self.create_frequency_grid(H_new,W_new)
        except AttributeError:
            self._shape  = shape

    @property
    def Kx(self):
        return self._Kx

    @Kx.setter
    def Kx(self, Kx):
        self.register_buffer("_Kx", Kx)

    @property
    def Ky(self):
        return self._Ky

    @Ky.setter
    def Ky(self, Ky):
        self.register_buffer("_Ky", Ky)

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


    def visualize_kernel(self,
			field : ElectricField,
		):
        
        kernel = self.create_kernel(field = field)

        plt.subplot(121)
        plt.imshow(kernel.abs().cpu().squeeze(),vmin=0)
        plt.title("Amplitude")
        plt.subplot(122)
        plt.imshow(kernel.angle().cpu().squeeze())
        plt.title("Phase")
        plt.tight_layout()
    
    def create_kernel(self,
		field : ElectricField,
			):
        
        tempShape = torch.tensor(field.shape)
        tempShapeH, tempShapeW = self.compute_padding(tempShape[-2], tempShape[-1])
        tempShape[-2] = tempShapeH
        tempShape[-1] = tempShapeW
        tempShape = torch.Size(tempShape)
        self.shape = tempShape
        
        # extract dx, dy spacing
        dx = torch.tensor([field.spacing[0]], device=self.device)
        dy = torch.tensor([field.spacing[1]], device=self.device)
        
        # extract wavelengths from the field
        wavelengths = field.wavelengths
        
        #################################################################
		# Prepare Dimensions to shape to be able to process 4D data
		# ---------------------------------------------------------------
		# NOTE: This just means we're broadcasting lower-dimensional
		# tensors to higher dimensional ones

        # Expand wavelengths for H and W dimension
        wavelengths_expand  = wavelengths[:,None,None]
        
        dx_expand = dx[:, None, None] # Expand to H and W
        dy_expand = dy[:, None, None] # Expand to H and W

        self.Kx = self.Kx.to(device=self.device)
        self.Ky = self.Ky.to(device=self.device)
        
        Kx = 2 * torch.pi * self.Kx[None, None, :, :] / dx_expand
        Ky = 2 * torch.pi * self.Ky[None, None, :, :] / dy_expand
        
        # create the frequency grid for each wavelength/spacing
        K2 = Kx**2 + Ky**2
        #print(K2.is_cuda)
        
        # compute ASM kernel for each wavelengths
        K_lambda = 2 * torch.tensor(np.pi) / wavelengths_expand
        K_lambda_2 = K_lambda**2    # shape : B x C x H x W
        
        # information about ASM in Goodman's Fourier optics book (3rd edition)
        ang = self._z * torch.sqrt(K_lambda_2 - K2)
        
        # Compute the kernel without bandlimiting
        kernelOut =  torch.exp(1j * ang)
        # Remove evanescent components
        kernelOut[(K_lambda_2 - K2) < 0] = 0
        
        if (self.bandlimit_kernel):
            #################################################################
			# Bandlimit the kernel
			# see band-limited ASM - Matsushima et al. (2009)
			# K. Matsushima and T. Shimobaba,
			# "Band-Limited Angular Spectrum Method for Numerical Simulation of Free-Space Propagation in Far and Near Fields,"
			#  Opt. Express  17, 19662-19673 (2009).
			#################################################################
            
            # the physical size of the padded field
            length_x = tempShapeH * dx_expand
            length_y = tempShapeH * dy_expand
            
            #################################################################
            # Check the critical distance Z_c, if the propagation is greater than Z_c the transfer function will be undersampled
            if self.check_Zc is True:
                Zc = (tempShapeH * dx**2) * torch.sqrt(1 - (torch.max(wavelengths) / (2 * dx))**2) / torch.max(wavelengths)
                if self._z > Zc:
                    print("The propagation distance is greater than critical distance {} m, the TF will be undersampled!".format(Zc.detach().cpu().numpy()))
                else:
                    print("The critical distance is {} m, the TF will be fine during the sampling !".format(Zc.detach().cpu().numpy()))
                self.check_Zc = False # only check this paramter once during any loop operation
            #################################################################
                
            
            # band-limited ASM - Matsushima et al. (2009)
            delta_u = ((2*np.pi / dx_expand) / (2 * tempShapeH)) / (2*np.pi)
            delta_v = ((2*np.pi / dy_expand) / (2 * tempShapeH)) / (2*np.pi)

            u_limit = 1 / torch.sqrt( ((2 * delta_u * self._z)**2) + 1 ) / wavelengths_expand
            v_limit = 1 / torch.sqrt( ((2 * delta_v * self._z)**2) + 1 ) / wavelengths_expand
            
            if (self.bandlimit_type == 'exact'):
                constraint1 = (((Kx**2) / ((2*np.pi*u_limit)**2)) + ((Ky**2) / (K_lambda**2))) <= 1
                constraint2 = (((Kx**2) / (K_lambda**2)) + ((Ky**2) / ((2*np.pi*v_limit)**2))) <= 1
                
                combinedConstraints = constraint1 & constraint2
                kernelOut[~combinedConstraints] = 0
            elif (self.bandlimit_type == 'approx'):
                k_x_max_approx = 2*np.pi / torch.sqrt( ((2*(1/length_x)*self._z)**2) + 1 ) / wavelengths_expand
                k_y_max_approx = 2*np.pi / torch.sqrt( ((2*(1/length_y)*self._z)**2) + 1 ) / wavelengths_expand
                
                kernelOut[ ( torch.abs(Kx) > k_x_max_approx) | (torch.abs(Ky) > k_y_max_approx) ] = 0
                
            else:
                raise Exception("Should not be in this state.")

        return kernelOut            
        
    
    def forward(self, 
                field: ElectricField
                ) -> ElectricField:
        
        """
		Takes in optical field and propagates it to the instantiated distance using ASM from KIM
		Eq. 4.22 (page 50)

		Args:
			field (ElectricField): Complex field 4D tensor object

		Returns:
			ElectricField: The electric field after the wave propagation model
		"""
  
        # extract the data tensor from the field
        wavelengths = field.wavelengths
        field_data  = field.data
        B,C,H,W = field_data.shape
        
        #################################################################
        # Apply the convolution in Angular Spectrum
        #################################################################
        
        try:
            # Pad 'field_data' avoid convolution wrap-around effects
            if (self.do_padding):
                pad_x, pad_y = self.compute_padding(H, W, return_size_of_padding=True)
                field_data = pad(field_data, (pad_y, pad_y, pad_x, pad_x), mode='constant', value=0)

            _, _, H_pad, W_pad = field_data.shape
                
            # convert to angular spectrum
            field_data_spectrum = ft2(field_data)
                
            # Returns a frequency domain kernel
            ASM_Kernel_freq_domain = self.create_kernel(field=field)
            #print(ASM_Kernel_freq_domain.shape)
                
            field_data = field_data_spectrum * ASM_Kernel_freq_domain
                
            # Convert 'field_data' back to the space domain
            field_data = ift2(field_data)	# Dimensions: B x C x H_pad x W_pad
                
            # Unpad the image after convolution, if necessary
            if (self.do_padding and self.do_unpad_after_pad):
                center_crop = torchvision.transforms.CenterCrop([H,W])
                field_data = center_crop(field_data)
                
        except Exception as err:
            if (type(err) is RuntimeError):
                print("##################################################")
                print("An error occurred.  If the error was due to insufficient memory, try decreasing the size of the input field or the size of the padding (i.e. decrease 'padding_scale').")
                print("For the best results (e.g. to avoid convolution edge artifacts), the support of the input field should be at most 1/2 the size of the input field after padding.")
                print("If limiting the support like that is not feasible, try to make it so that most of the input field energy is contained in a region that is 1/2 the size of the input field after padding.")
                print("##################################################")
            raise err

        Eout = ElectricField(
				data=field_data,
				wavelengths=wavelengths,
				spacing=field.spacing
				)
        
        return Eout
    
    

if __name__=='__main__':
    
    from utils.units import *
    import numpy as np
    import torch
    from LightSource.Gaussian_beam import Guassian_beam
    
    c0 = 2.998e8
    f1 = 340e9
    f2 = 360e9
    wavelength1 = c0 / f1
    wavelength2 = c0 / f2

    fs = torch.range(f1, f2, 4e9)

    wavelengths = c0 / fs
    print(wavelengths)

    gm = Guassian_beam(height=1000, width=1000, 
                       beam_waist=1 * mm, wavelengths=wavelength2, 
                       spacing=0.1 * mm)
    field = gm()
    
    asm_prop = ASM_prop(z_distance=1 * m, bandlimit_type='exact', padding_scale=2)
    
    field_propagated = asm_prop.forward(
    field = field
    )
    
    
