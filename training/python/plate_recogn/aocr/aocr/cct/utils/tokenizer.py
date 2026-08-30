import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .mobilenetv4 import MobileNetV4
from .resnet18 import ResNet18
from .blurpool import BlurPool
from .activations import ACTIVATIONS
from einops.layers.torch import Rearrange

class Projection(nn.Module):
    def __init__(self, out_shape, out_seq_len, dropout=0.1):
        super(Projection, self).__init__()
        assert len(out_shape) == 4, 'Shape must be 4D'
        self.sequence_length_in = (out_shape[-1] * out_shape[-2])
        self.net = nn.Sequential(
            Rearrange('b c h w -> b (h w) c ()'),
            nn.Dropout2d(dropout),
            nn.Conv2d(self.sequence_length_in, out_seq_len, 1),
            Rearrange('b s c 1 -> b s (c 1)')
        )
        net_shape = self.net(torch.zeros(out_shape)).shape
        self.sequence_length_out = net_shape[-2]
        self.output_channels = net_shape[-1]
    def forward(self, input):
        return self.net(input)
    
class MNV4Tokenizer(nn.Module):
    def __init__(self,
                 imgH, imgW,
                 input_channels,
                 seq_len,
                 proj_dropout=0.1,
                 block_size='medium',
                 width_mult=1.0,
                 out_stage=3,
                 activation_type='ReLU'):
        super(MNV4Tokenizer, self).__init__()

        # MobileNetV4 backbone
        self.conv_layers = MobileNetV4(input_channels, block_size=block_size, width_mult=width_mult, out_stage=out_stage, activation=activation_type)
        self.projection = Projection(
            self.conv_layers(torch.zeros((1, input_channels, imgH, imgW))).shape, 
            seq_len,
            proj_dropout
        )

    def forward(self, x):
        return self.projection(self.conv_layers(x))

class ResNet18Tokenizer(nn.Module):
    def __init__(self,
                 imgH, imgW,
                 input_channels,
                 seq_len,
                 proj_dropout=0.1,
                 width_mult=1.0,
                 out_stage=3,
                 activation_type='ReLU'):
        super(ResNet18Tokenizer, self).__init__()

        self.conv_layers = ResNet18(input_channels, width_mult=width_mult, out_stage=out_stage, activation=activation_type)
        self.projection = Projection(
            self.conv_layers(torch.zeros((1, input_channels, imgH, imgW))).shape, 
            seq_len,
            proj_dropout
        )

    def forward(self, x):
        return self.projection(self.conv_layers(x))

class VGGTokenizer(nn.Module):
    def __init__(self,
                 imgH, imgW,
                 input_channels,
                 seq_len,
                 proj_dropout=0.1,
                 output_channel=256,
                 activation_type='ReLU'):
        super(VGGTokenizer, self).__init__()

        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [64, 128, 256, 512]

        activation_fn = lambda: ACTIVATIONS[activation_type]
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, self.output_channel[0], 3, 1, 1),
            nn.BatchNorm2d(self.output_channel[0]),
            activation_fn(),
            nn.MaxPool2d((2, 2), (2, 2)),
            
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1),
            nn.Conv2d(self.output_channel[1], self.output_channel[1], 3, 1, 1),
            nn.BatchNorm2d(self.output_channel[1]),
            activation_fn(), 
            nn.MaxPool2d((2, 2), (2, 2)),
            
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1),
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1),
            nn.BatchNorm2d(self.output_channel[2]),
            activation_fn(),
            nn.MaxPool2d((2, 2), (2, 2)),
            
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1),
            nn.BatchNorm2d(self.output_channel[3]),
            activation_fn(),

            Rearrange('b c h w -> b (h w) c'),
        )
        self.projection = Projection(
            self.conv_layers(torch.zeros((1, input_channels, imgH, imgW))).shape, 
            seq_len,
            proj_dropout
        )

    def forward(self, x):
        return self.projection(self.conv_layers(x))

class Tokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 n_conv_layers=1,
                 n_input_channels=3,
                 n_output_channels=64,
                 in_planes=64,
                 activation=None,
                 max_pool=True,
                 conv_bias=False):
        super(Tokenizer, self).__init__()

        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(n_filter_list[i], n_filter_list[i + 1],
                          kernel_size=(kernel_size, kernel_size),
                          stride=(stride, stride),
                          padding=(padding, padding), bias=conv_bias),
                nn.Identity() if activation is None else activation(),
                nn.MaxPool2d(kernel_size=pooling_kernel_size,
                             stride=pooling_stride,
                             padding=pooling_padding) if max_pool else nn.Identity()
            )
                for i in range(n_conv_layers)
            ])

        self.flattener = nn.Flatten(2, 3)
        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        return self.flattener(self.conv_layers(x)).transpose(-2, -1)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

class TextTokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 embedding_dim=300,
                 n_output_channels=128,
                 activation=None,
                 max_pool=True,
                 *args, **kwargs):
        super(TextTokenizer, self).__init__()

        self.max_pool = max_pool
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, n_output_channels,
                      kernel_size=(kernel_size, embedding_dim),
                      stride=(stride, 1),
                      padding=(padding, 0), bias=False),
            nn.Identity() if activation is None else activation(),
            nn.MaxPool2d(
                kernel_size=(pooling_kernel_size, 1),
                stride=(pooling_stride, 1),
                padding=(pooling_padding, 0)
            ) if max_pool else nn.Identity()
        )

        self.apply(self.init_weight)

    def seq_len(self, seq_len=32, embed_dim=300):
        return self.forward(torch.zeros((1, seq_len, embed_dim)))[0].shape[1]

    def forward_mask(self, mask):
        new_mask = mask.unsqueeze(1).float()
        cnn_weight = torch.ones(
            (1, 1, self.conv_layers[0].kernel_size[0]),
            device=mask.device,
            dtype=torch.float)
        new_mask = F.conv1d(
            new_mask, cnn_weight, None,
            self.conv_layers[0].stride[0], self.conv_layers[0].padding[0], 1, 1)
        if self.max_pool:
            new_mask = F.max_pool1d(
                new_mask, self.conv_layers[2].kernel_size[0],
                self.conv_layers[2].stride[0], self.conv_layers[2].padding[0], 1, False, False)
        new_mask = new_mask.squeeze(1)
        new_mask = (new_mask > 0)
        return new_mask

    def forward(self, x, mask=None):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.transpose(1, 3).squeeze(1)
        if mask is not None:
            mask = self.forward_mask(mask).unsqueeze(-1).float()
            x = x * mask
        return x, mask

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
