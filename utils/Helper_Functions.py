########################################################
# Copyright (c) 2022 Meta Platforms, Inc. and affiliates
#
# Holotorch is an optimization framework for differentiable wave-propagation written in PyTorch 
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
#
# Contact:
# florianschiffers (at) gmail.com
# ocossairt ( at ) fb.com
#
########################################################



from __future__ import print_function
import torch
import torch.nn as nn
import warnings
from typing import Union
import copy

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

def set_default_device(device: Union[str, torch.device]):
    if not isinstance(device, torch.device):
        device = torch.device(device)
        
    print(device)

    if device.type == 'cuda':
        print("TEST")
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.cuda.set_device(device.index)
        print("CUDA1")
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
        print("CUDA2")

def total_variation(input: torch.tensor):
    '''
    compute centerered finite difference derivative for input along dimensions dim
    zero pad the boundary

    input: 4D torch tensor with dimension 2,3 being the spatial difference
    returns: scalar value |dx|_1 + |dy|_1  
    '''
    # reshape if its a 6D tensor
    if input.ndim == 6:
        B,T,P,C,H,W = input.shape
        input = input.view(B*T*P,C,H,W)

    dx, dy = center_difference(input)
    return dx.abs().mean() + dy.abs().mean()

def center_difference(input: torch.tensor):
    '''
    compute centerered finite difference derivative for input along dimensions dim
    zero pad the boundary

    input: 4D torch tensor with dimension 2,3 being the spatial difference
    returns: dx, dy - 4D tensors same size as input 
    '''
    # create a new tensor of zeros for zeropadding
    dx = torch.zeros_like(input)
    dy = torch.zeros_like(input)
    _, _, H, W = input.shape
    dx[:,:,:,1:-1] = W/4*(-input[:,:,:,0:-2] + 2*input[:,:,:,1:-1] - input[:,:,:,2:])
    dy[:,:,1:-1,:] = H/4*(-input[:,:,0:-2,:] + 2*input[:,:,1:-1,:] - input[:,:,2:,:])
    return dx, dy

def tt(x):
    return torch.tensor(x)

def regular_grid4D(M,N,H,W, range=tt([[-1,1],[-1,1],[-1,1],[-1,1]]), device=torch.device("cpu")):
    '''
    Create a regular grid 4D tensor with dims M x N x H x W specified within a range 
    '''
    #Coordinates                 
    x = torch.linspace(range[0,0], range[0,1], M, device=device)  
    y = torch.linspace(range[1,0], range[1,1], N, device=device)  
    u = torch.linspace(range[2,0], range[2,1], H, device=device)  
    v = torch.linspace(range[3,0], range[3,1], W, device=device)  

    #Generate Coordinate Mesh and store it in the model
    return torch.meshgrid(x,y,u,v)    

def regular_grid2D(H,W, range=tt([[-1,1],[-1,1]]), device=torch.device("cpu")):
    '''
    Create a regular grid 2D tensor with dims H x W specified within a range 
    '''
    #XCoordinates                 
    x_c = torch.linspace(range[0,0], range[0,1], W, device=device)  
    #YCoordinates 
    y_c = torch.linspace(range[1,0], range[1,1], H, device=device)  
    #Generate Coordinate Mesh and store it in the model
    return torch.meshgrid(x_c,y_c)    

def ft2(input, delta=1, norm = 'ortho', pad = False):
    """
    Helper function computes a shifted fourier transform with optional scaling
    """
    return perform_ft(
        input=input,
        delta = delta,
        norm = norm,
        pad = pad,
        flag_ifft=False
    )

def ift2(input, delta=1, norm = 'ortho', pad = False):
    
    return perform_ft(
        input=input,
        delta = delta,
        norm = norm,
        pad = pad,
        flag_ifft=True
    )

def perform_ft(input, delta=1, norm = 'ortho', pad = False, flag_ifft : bool = False):
    
    # Get the initial shape (used later for transforming 6D to 4D)
    tmp_shape = input.shape

    # Save Size for later crop
    Nx_old = int(input.shape[-2])
    Ny_old = int(input.shape[-1])
        
    # Pad the image for avoiding convolution artifacts
    if pad == True:
        
        pad_scale = 1
        
        pad_nx = int(pad_scale * Nx_old / 2)
        pad_ny = int(pad_scale * Ny_old / 2)
        
        input = torch.nn.functional.pad(input, (pad_ny,pad_ny,pad_nx,pad_nx), mode='constant', value=0)
    
    if flag_ifft == False:
        myfft = torch.fft.fft2
        my_fftshift = torch.fft.fftshift
    else:
        myfft = torch.fft.ifft2
        my_fftshift = torch.fft.ifftshift


    
    # Compute the Fourier Transform
    out = (delta**2)* my_fftshift( myfft (  my_fftshift (input, dim=(-2,-1))  , dim=(-2,-1), norm=norm)  , dim=(-2,-1))
    
    if pad == True:
        input_size = [Nx_old, Ny_old]
        pool = torch.nn.AdaptiveAvgPool2d(input_size)
        
        if out.is_complex():
            out = pool(out.real) + 1j * pool(out.imag)
        else:
            out = pool(out)
    return out
        

def generateGrid(res, deltaX, deltaY, centerGrids = True, centerAroundZero = True, device=None):
	if (torch.is_tensor(deltaX)):
		deltaX = copy.deepcopy(deltaX).squeeze().to(device=device)
	if (torch.is_tensor(deltaY)):
		deltaY = copy.deepcopy(deltaY).squeeze().to(device=device)

	if (centerGrids):
		if (centerAroundZero):
			xCoords = torch.linspace(-((res[0] - 1) // 2), (res[0] - 1) // 2, res[0]).to(device=device) * deltaX
			yCoords = torch.linspace(-((res[1] - 1) // 2), (res[1] - 1) // 2, res[1]).to(device=device) * deltaY
		else:
			xCoords = (torch.linspace(0, res[0] - 1, res[0]) - (res[0] // 2)).to(device=device) * deltaX
			yCoords = (torch.linspace(0, res[1] - 1, res[1]) - (res[1] // 2)).to(device=device) * deltaY
	else:
		xCoords = torch.linspace(0, res[0] - 1, res[0]).to(device=device) * deltaX
		yCoords = torch.linspace(0, res[1] - 1, res[1]).to(device=device) * deltaY

	xGrid, yGrid = torch.meshgrid(xCoords, yCoords)

	return xGrid, yGrid


def normalize(x):
    """normalize to range [0-1]"""
    batch_size, num_obj, height, width = x.shape

    x = x.view(batch_size, -1)
    # x -= x.min(1, keepdim=True)[0]
    x /= x.max(1, keepdim=True)[0]
    x = x.view(batch_size, num_obj, height, width)
    return x



##################### Noise Model ########################################
import torch


class GaussianNoise(torch.nn.Module):
    r"""

    Additive gaussian noise with standard deviation :math:`\sigma`, i.e., :math:`y=z+\epsilon` where :math:`\epsilon\sim \mathcal{N}(0,I\sigma^2)`.

    It can be added to a physics operator in its construction or by setting the ``noise_model``
    attribute of the physics operator.

    :param float sigma: Standard deviation of the noise.

    """

    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = torch.nn.Parameter(torch.tensor(sigma), requires_grad=False)

    def forward(self, x):
        r"""
        Adds the noise to measurements x

        :param torch.Tensor x: measurements
        :returns: noisy measurements
        """
        return x + torch.randn_like(x) * self.sigma


class PoissonNoise(torch.nn.Module):
    r"""

    Poisson noise is defined as
    :math:`y = \mathcal{P}(\frac{x}{\gamma})`
    where :math:`\gamma` is the gain.

    If ``normalize=True``, the output is divided by the gain, i.e., :math:`\tilde{y} = \gamma y`.

    :param float gain: gain of the noise.
    :param bool normalize: normalize the output.

    """

    def __init__(self, gain=1.0, normalize=True):
        super().__init__()
        self.normalize = torch.nn.Parameter(
            torch.tensor(normalize), requires_grad=False
        )
        self.gain = torch.nn.Parameter(torch.tensor(gain), requires_grad=False)

    def forward(self, x):
        r"""
        Adds the noise to measurements x

        :param torch.Tensor x: measurements
        :returns: noisy measurements
        """
        y = torch.poisson(x / self.gain)
        if self.normalize:
            y *= self.gain
        return y


class PoissonGaussianNoise(torch.nn.Module):
    r"""
    Poisson-Gaussian noise is defined as
    :math:`y = \gamma z + \epsilon` where :math:`z\sim\mathcal{P}(\frac{x}{\gamma})`
    and :math:`\epsilon\sim\mathcal{N}(0, I \sigma^2)`.

    :param float gain: gain of the noise.
    :param float sigma: Standard deviation of the noise.

    """

    def __init__(self, gain=1.0, sigma=0.1):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(gain), requires_grad=False)
        self.sigma = torch.nn.Parameter(torch.tensor(sigma), requires_grad=False)

    def forward(self, x):
        r"""
        Adds the noise to measurements x

        :param torch.Tensor x: measurements
        :returns: noisy measurements
        """
        y = torch.poisson(x / self.gain) * self.gain

        y += torch.randn_like(x) * self.sigma
        return y


class UniformNoise(torch.nn.Module):
    r"""
    Uniform noise is defined as
    :math:`y = x + \epsilon` where :math:`\epsilon\sim\mathcal{U}(-a,a)`.

    :param float a: amplitude of the noise.
    """

    def __init__(self, a=0.1):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(a), requires_grad=False)

    def forward(self, x):
        r"""
        Adds the noise to measurements x

        :param torch.Tensor x: measurements
        :returns: noisy measurements
        """
        return x + (torch.rand_like(x) - 0.5) * 2 * self.a
    

########################## Look-up table helper function ########################

def lut_mid(lut):
    return [(a + b) / 2 for a, b in zip(lut[:-1], lut[1:])]


def nearest_neighbor_search(input_val, lut, lut_midvals=None):
    """
    Quantize to nearest neighbor values in lut
    :param input_val: input tensor
    :param lut: list of discrete values supported
    :param lut_midvals: set threshold to put into torch.searchsorted function.
    :return:
    """
    # if lut_midvals is None:
    #     lut_midvals = torch.tensor(lut_mid(lut), dtype=torch.float32).to(input_val.device)
    idx = nearest_idx(input_val, lut_midvals)
    assert not torch.isnan(idx).any()
    return lut[idx], idx


def nearest_idx(input_val, lut_midvals):
    """ Return nearest idx of lut per pixel """
    input_array = input_val.detach()
    len_lut = len(lut_midvals)
    # print(lut_midvals.shape)
    # idx = torch.searchsorted(lut_midvals.to(input_val.device), input_array, right=True)
    idx = torch.bucketize(input_array, lut_midvals.to(input_val.device), right=True)

    return idx % len_lut    
    