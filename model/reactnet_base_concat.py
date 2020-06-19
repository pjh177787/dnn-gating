"""
Author: Yichi Zhang ( yz2499 -AT- cornell -DOT- edu )
A slightly modified implementation of ReActNet baseline as described in paper [2].
ReActNet is a binary model modified from MobileNet V1.

Reference: 
    [1] MobileNet V1 code:
        https://github.com/marvis/pytorch-mobilenet
    [2] Author: Zechun Liu, Zhiqiang Shen, Marios Savvides, 
                Kwang-Ting Cheng
        Title:  ReActNet: Towards Precise Binary Neural 
                Network with Generalized Activation Functions
        URL:    https://arxiv.org/abs/2003.03488
"""


import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

#sys.path.append('../../')
import model.quantization as q


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


"""
    Baseline Network: Skip connection in each conv layer.
"""

class ReductionBlock(nn.Module):
    def __init__(self, inp, stride):
        super(ReductionBlock, self).__init__()

        self.binarize = q.FastSign()

        self.conv3x3 = nn.Sequential(
                            q.BinaryConv2d(inp, inp, 3, stride, 1, bias=False),
                            nn.BatchNorm2d(inp)
                         )

        self.shortcut1 = nn.Sequential()
        if stride == 2:
            self.shortcut1 = nn.Sequential(
                                nn.AvgPool2d(kernel_size=2, stride=2),
                             )

        self.shortcut2 = LambdaLayer(lambda x: torch.cat((x, x), dim=1))
        # print(input)
        self.pointwise = nn.Sequential(
                           q.BinaryConv2d(inp, 2*inp, 1, 1, 0, bias=False),
                           nn.BatchNorm2d(2*inp)
                         )

    def forward(self, input):
        # input = self.conv3x3(self.binarize(input)) + self.shortcut1(input)
        """ * Duplicate the activations on the shortcut """
        # input = self.pointwise(self.binarize(input)) + self.shortcut2(input)
        input = self.conv3x3(self.binarize(input)) + self.shortcut1(input)
        input = self.pointwise(self.binarize(input)) + self.shortcut2(input)
        # print(input.shape)
        return input

class NormalBlock(nn.Module):
    def __init__(self, inp, stride):
        super(NormalBlock, self).__init__()

        self.binarize = q.FastSign()

        self.conv3x3 = nn.Sequential(
                            q.BinaryConv2d(inp, inp, 3, stride, 1, bias=False),
                            nn.BatchNorm2d(inp)
                         )

        self.pointwise = nn.Sequential(
                            q.BinaryConv2d(inp, inp, 1, 1, 0, bias=False),
                            nn.BatchNorm2d(inp)
                         )

    def forward(self, input):
        input = self.conv3x3(self.binarize(input)) + input
        input = self.pointwise(self.binarize(input)) + input
        return input


class ReActNet(nn.Module):

    def __init__(self):
        super(ReActNet, self).__init__()
        print("ReActNet base with duplicated activations on the bypass path.") 
        print("No RPReLU and RSign.")
        
        """ input layer """
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup)
                #nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            if inp == oup:
                return NormalBlock(inp, stride)
            elif 2*inp == oup:
                return ReductionBlock(inp, stride)
            else:
                raise NotImplementedError("Neither inp == oup nor 2*inp == oup")

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, 200)

    def forward(self, x):
        x = self.model(x)
        x = self.pool(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x



def speed(model, name):
    t0 = time.time()
    input = torch.rand(1,3,224,224).cuda()
    t1 = time.time()

    model(input)
    t2 = time.time()

    model(input)
    t3 = time.time()
    
    print('%10s : %f' % (name, t3 - t2))


if __name__ == '__main__':
    reactnet = ReActNet().cuda()
    speed(reactnet, 'reactnet')
