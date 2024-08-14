import torch
import sys
sys.path.append('../')
from DataType.ElectricField import ElectricField
from LightSource.Gaussian_beam import Guassian_beam, VectorialGuassian_beam
from utils.Helper_Functions import normalize
from Props.RSC_Prop import *
from Props.CZT_Prop import *
from utils.units import *
from Addons.Polarization import PolarizationAnalyser

torch.set_default_dtype(torch.float64)

wavelengths = 1 * mm


gm = Guassian_beam(height=200, width=200, 
                   beam_waist_x = 2*mm,
                   beam_waist_y = 2 * mm,
                   wavelengths=wavelengths, 
                   alpha= 0,
                   spacing=1*mm)
field = gm()
    
asm_prop = RSC_prop(z_distance=500 * mm)
czt_prop = CZT_prop(z_distance=500 * mm)
    
field_propagated = asm_prop.forward(
    field = field
    )

field_propagated_2 = czt_prop.forward(
    field = field, 
    outputHeight=200,
	outputWidth=200, 						
	outputPixel_dx=1*mm, 
	outputPixel_dy=1*mm,
    )