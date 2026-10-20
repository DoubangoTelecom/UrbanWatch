import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import ResNet
from .hsv import rgb_to_hsv

class ColorNet(nn.Module):
    def __init__(self, input_channels, width_mult=1.0, out_stage=5, activation='ReLU'):
        super(ColorNet, self).__init__()
        assert input_channels == 3, f'Input channels({input_channels}) must be equal to 3'
        self.input_channels = input_channels
        self.leg0 = ResNet(input_channels=3, width_mult=width_mult, out_stage=out_stage, activation=activation, colorNet=True)
        self.leg1 = ResNet(input_channels=4, width_mult=width_mult*0.5, out_stage=out_stage, activation=activation, colorNet=True)

        self.output_channels = (self.leg0.output_channels + self.leg1.output_channels)
    
    def forward(self, x):
        r, g, b = x.unbind(1)
        y = torch.stack([r-g, r-b, g-b, r+g+b], dim=1)
        return torch.cat([self.leg0(x), self.leg1(y)], dim=1)