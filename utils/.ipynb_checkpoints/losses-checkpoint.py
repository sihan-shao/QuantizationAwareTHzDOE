import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import SSIM


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class BinaryCEloss(nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None) -> None:
        super().__init__()
        """
        This loss combines a Sigmoid layer and the BCELoss in one single class. 
        This version is more numerically stable than using a plain Sigmoid followed by a BCELoss
        """
        self.loss = torch.nn.BCEWithLogitsLoss(weight, size_average, reduce, reduction, pos_weight)
        
    def forward(self, predict, target):

        return self.loss(predict, target)
    


class SSIMloss(nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1) -> None:
        super().__init__()
        
        self.ssim = SSIM(win_size=win_size, 
                         win_sigma=win_sigma, 
                         data_range=data_range, 
                         size_average=size_average, 
                         channel=channel)
        
    def forward(self, pred, target):
        
        return 1 - self.ssim(pred, target)



class Hierarchyloss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.bce = BinaryCEloss()
        self.dice = BinaryDiceLoss()
        self.ssimloss = SSIMloss()
        
    def forward(self, pred, ground_truth):
        
        return self.bce(pred, ground_truth) + self.dice(pred, ground_truth) + self.ssimloss(pred, ground_truth)
    
