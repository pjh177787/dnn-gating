import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


##########
##  Quant
##########

class FastSign(nn.Module):
    """
    This is a fast version of the SignActivation.
    """
    def __init__(self):
        super(FastSign, self).__init__()

    def forward(self, input):
        """
        Add a small value 1e-6 to the input to increase the
        numerical stability
        """
        out_forward = torch.sign(input + 1e-6)
        out_backward = torch.clamp(input, -1.3, 1.3)
        return out_forward.detach() - out_backward.detach() + out_backward



##########
##  Layer
##########


class BinaryConv2d(nn.Conv2d):
    """
    A convolutional layer with its weight tensor binarized to {-1, +1}.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super(BinaryConv2d, self).__init__(in_channels, out_channels,
                                              kernel_size, stride,
                                              padding, dilation, groups,
                                              bias, padding_mode)
        self.binarize = FastSign()

    def forward(self, input):
        return F.conv2d(input, self.binarize(self.weight),
                        self.bias, self.stride, self.padding,
                        self.dilation, self.groups)

