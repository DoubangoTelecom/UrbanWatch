import torch
import torch.nn as nn
import torch.nn.functional as F
from .activations import ACTIVATIONS
from .hsv import rgb_to_hsv


class VGG(nn.Module):
    def __init__(self, input_channels, activation='ReLU'):
        super(VGG, self).__init__()      
        
        mult = 10
        self.output_channels = (3*mult*32) #+ (4*mult*32)
        self.leg1 = VGG.BuildLeg(input_channels=3, activation=activation, mult=mult)
        self.leg2 = VGG.BuildLeg(input_channels=4, activation=activation, mult=mult)
        

    @staticmethod
    def BuildLeg(input_channels, activation, mult=5):
        activation_fn = lambda: ACTIVATIONS[activation]
        output_channels = [input_channels*mult*2, input_channels*mult*4, input_channels*mult*8, input_channels*mult*16, input_channels*mult*32]
        return nn.Sequential(
            nn.BatchNorm2d(input_channels),
            VGG.DepthwiseConv(input_channels, output_channels[0], 3, 2, 1),
            nn.BatchNorm2d(output_channels[0]),
            activation_fn(),
            
            VGG.DepthwiseConv(output_channels[0], output_channels[1], 3, 1, 1),
            VGG.DepthwiseConv(output_channels[1], output_channels[1], 3, 2, 1),
            nn.BatchNorm2d(output_channels[1]),
            activation_fn(), 
            
            VGG.DepthwiseConv(output_channels[1], output_channels[2], 3, 1, 1),
            VGG.DepthwiseConv(output_channels[2], output_channels[2], 3, 2, 1),
            nn.BatchNorm2d(output_channels[2]),
            activation_fn(),

            VGG.DepthwiseConv(output_channels[2], output_channels[3], 3, 1, 1),
            VGG.DepthwiseConv(output_channels[3], output_channels[3], 3, 2, 1),
            nn.BatchNorm2d(output_channels[3]),
            activation_fn(),
            
            VGG.DepthwiseConv(output_channels[3], output_channels[4], 3, 1, 1),
            nn.BatchNorm2d(output_channels[4]),
            activation_fn(),
        )

    @staticmethod
    def DepthwiseConv(in_channels, out_channels, kernel_size, stride=1, padding=0):
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=in_channels)
    
    def forward(self, x):
        '''r, g, b = x.unbind(1)
        y = torch.stack([r-g, r-b, g-b, r+g+b ], dim=1)
        a = self.leg1(x)
        b = self.leg2(y)
        return torch.cat([a, b], dim=1)'''
        return self.leg1(rgb_to_hsv(x))
        