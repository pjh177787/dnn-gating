import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import utils.pg_utils as q
import numpy as np

class _fastSign(nn.Module):
    """
    This is a fast version of the SignActivation.
    """
    def __init__(self):
        super(_fastSign, self).__init__()

    def forward(self, input):
        """
        Add a small value 1e-6 to the input to increase the
        numerical stability
        """
        out_forward = torch.sign(input + 1e-6)
        out_backward = torch.clamp(input, -1.3, 1.3)
        return out_forward.detach() - out_backward.detach() + out_backward


class binaryConv2d(nn.Conv2d):
    """ 
    A convolutional layer with its weight tensor and input tensor quantized. 
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', wbits=1, abits=1):
        super(binaryConv2d, self).__init__(in_channels, out_channels, 
                                              kernel_size, stride, 
                                              padding, dilation, groups, 
                                              bias, padding_mode)
        self.binarize = _fastSign()
        self.weight_rescale = 1
        # self.weight_rescale = \
        #     np.sqrt(1.0/(kernel_size**2 * in_channels)) if (wbits == 1) else 1.0

    def forward(self, input):
        """ 
        1. Quantize the input tensor
        2. Quantize the weight tensor
        3. Rescale via McDonnell 2018 (https://arxiv.org/abs/1802.08530)
        4. perform convolution
        """
        return F.conv2d(self.binarize(input),
                        self.binarize(self.weight) * self.weight_rescale,
                        self.bias, self.stride, self.padding, 
                        self.dilation, self.groups)

class binaryLinear(nn.Linear):
    """ 
    A fully connected layer with its weight tensor and input tensor quantized. 
    """
    def __init__(self, in_features, out_features, bias=True, wbits=1, abits=1):
        super(binaryLinear, self).__init__(in_features, out_features, bias)
        self.binarize = _fastSign()
        self.weight_rescale = 1
        # self.weight_rescale = np.sqrt(1.0/in_features) if (wbits == 1) else 1.0

    def forward(self, input):
        """ 
        1. Quantize the input tensor
        2. Quantize the weight tensor
        3. Rescale via McDonnell 2018 (https://arxiv.org/abs/1802.08530)
        4. perform matrix multiplication 
        """
        return F.linear(self.binarize(input), 
                        self.binarize(self.weight) * self.weight_rescale, 
                        self.bias)


class DownsampleA(nn.Module):

  def __init__(self, nIn, nOut, stride):
    super(DownsampleA, self).__init__()
    assert stride == 2
    self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

  def forward(self, x):
    x = self.avg(x)
    return torch.cat((x, x.mul(0)), 1)

class ResNetBasicblock(nn.Module):
  expansion = 1
  """
  RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
  """
  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(ResNetBasicblock, self).__init__()

    self.conv_a = binaryConv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn_a = nn.BatchNorm2d(planes)

    self.conv_b = binaryConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn_b = nn.BatchNorm2d(planes)

    self.downsample = downsample
    # self.binarize = _fastSign()
    self.relu = nn.PReLU() 

  def forward(self, x):
    residual = x

    basicblock = self.conv_a(x)
    basicblock = self.bn_a(basicblock)
    # basicblock = self.relu(basicblock)

    basicblock = self.conv_b(basicblock)
    basicblock = self.bn_b(basicblock)

    if self.downsample is not None:
      residual = self.downsample(x)
    
    # return self.relu(residual + basicblock)
    return residual + basicblock

class CifarResNet(nn.Module):
  """
  ResNet optimized for the Cifar dataset, as specified in
  https://arxiv.org/abs/1512.03385.pdf
  """
  def __init__(self, block, depth, num_classes):
    """ Constructor
    Args:
      depth: number of layers.
      num_classes: number of classes
      base_width: base width
    """
    super(CifarResNet, self).__init__()

    #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
    assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
    layer_blocks = (depth - 2) // 6
    print ('CifarResNet : Depth : {} , Layers for each block : {}'.format(depth, layer_blocks))

    self.num_classes = num_classes

    self.conv_1_3x3 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn_1 = nn.BatchNorm2d(64)

    self.inplanes = 64
    self.stage_1 = self._make_layer(block, 64, layer_blocks, 1)
    self.stage_2 = self._make_layer(block, 128, layer_blocks, 2)
    self.stage_3 = self._make_layer(block, 256, layer_blocks, 2)
    self.avgpool = nn.AvgPool2d(8)
    self.classifier = nn.Linear(256*block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        #m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        init.kaiming_normal(m.weight)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv_1_3x3(x)
    # x = F.relu(self.bn_1(x), inplace=True)
    x = self.stage_1(x)
    x = self.stage_2(x)
    x = self.stage_3(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return self.classifier(x)

def bin_resnet20(num_classes=10):
  """Constructs a ResNet-20 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNet(ResNetBasicblock, 20, num_classes)
  return model

def bin_resnet32(num_classes=10):
  """Constructs a ResNet-32 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNet(ResNetBasicblock, 32, num_classes)
  return model

def bin_resnet44(num_classes=10):
  """Constructs a ResNet-44 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNet(ResNetBasicblock, 44, num_classes)
  return model

def bin_resnet56(num_classes=10):
  """Constructs a ResNet-56 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNet(ResNetBasicblock, 56, num_classes)
  return model

def bin_resnet110(num_classes=10):
  """Constructs a ResNet-110 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNet(ResNetBasicblock, 110, num_classes)
  return model

