import sys
sys.path.append('../')
import torch
from utils.Helper_Functions import lut_mid

class DiscreteDOE:
    """
    Class for Discrete DOE that supports discrete LUT
    """
    _lut_midvals = None
    _lut = None
    prev_idx = 0.
    
    @property
    def lut_midvals(self):
        return self._lut_midvals
    
    @lut_midvals.setter
    def lut_midvals(self, new_midvals):
        self._lut_midvals = torch.tensor(new_midvals)#, device=torch.device('cuda'))
        
    @property
    def lut(self):
        return self._lut
    
    @lut.setter
    def lut(self, new_lut):
        if new_lut is None:
            self._lut = None
        else:
            self.lut_midvals = lut_mid(new_lut)
            if torch.is_tensor(new_lut):
                self._lut = new_lut.clone().detach()
            else:
                self._lut = torch.tensor(new_lut)#, device=torch.device('cuda'))

