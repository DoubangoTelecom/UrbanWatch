import torch
import torch.nn as nn
import torch.nn.functional as F
from .activations import ACTIVATIONS

# Backbone from https://github.com/samcw/ResNet18-Pytorch/blob/master/ResNet18.ipynb

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, activation='ReLU'):
        super(ResidualBlock, self).__init__()
        self.activation = activation
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            ACTIVATIONS[self.activation],
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            
    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = ACTIVATIONS[self.activation](out)
        
        return out

class ResNet18(nn.Module):
    def __init__(self, input_channels, width_mult=1.0, out_stage=3, activation='ReLU'):
        super(ResNet18, self).__init__()
        assert out_stage > 0 and out_stage <= 4, 'OutStage must be within [1,4]'
        self.out_stage = out_stage
        self.channels = [
            int(width_mult*32), int(width_mult*64), int(width_mult*128), int(width_mult*256), int(width_mult*512)
        ]
        self.output_channels = self.channels[out_stage]
        self.inchannel = self.channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, self.channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channels[0]),
            ACTIVATIONS[activation]
        )
        self.layer1 = self.make_layer(ResidualBlock, self.channels[1], 2, stride=2, activation=activation)
        self.layer2 = self.make_layer(ResidualBlock, self.channels[2], 2, stride=2, activation=activation)
        self.layer3 = self.make_layer(ResidualBlock, self.channels[3], 2, stride=2, activation=activation)        
        self.layer4 = self.make_layer(ResidualBlock, self.channels[4], 2, stride=2, activation=activation)
        
    def make_layer(self, block, channels, num_blocks, stride, activation):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride, activation))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        for i in range(1, 5):
            layer = getattr(self, "layer{}".format(i))
            x = layer(x)
            if i == self.out_stage:
                return x

        raise IndexError('{} not valid out stage index. Must be within [1,4]'.format(self.out_stage))