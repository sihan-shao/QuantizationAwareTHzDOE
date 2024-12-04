import matplotlib.pyplot as plt
import numpy as np
import pylab as plt

import torch

import imageio as io

import torch.nn.functional as F

from utils.units import *

def float_to_unit_identifier(val):
    """
    Takes a float value (e.g. 5*mm) and identifies which range it is
    e.g. mm , m, um etc.

    We always round up to the next 1000er decimal

    e.g.
    - 55mm will return mm
    - 100*m will return m
    - 0.1*mm will return um
    """
    exponent = np.floor(np.log10( val) / 3)
    unit_val = 10**(3*exponent)

    if unit_val == m:
        unit = "m"
    elif unit_val == mm:
        unit = "mm"
    elif unit_val == um:
        unit = "um"
    elif unit_val == nm:
        unit = "nm"
    return unit_val, unit


def add_colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar