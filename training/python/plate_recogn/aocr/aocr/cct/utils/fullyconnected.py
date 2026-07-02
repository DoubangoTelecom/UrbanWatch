import torch
import torch.nn.functional as F
import torch.nn as nn
from einops.layers.torch import Rearrange

""" 
    FullyConnected from Pytorch return E [ops/vsi_nn_op_fullconnect_relu.c:op_check:234]Inputs/Outputs data type not support: ASYM UINT8, ASYM UINT8,  INT8 
    on Debix A (VX Delegate)
"""

# matmul (@) is promoted to FullyConnected causing issues
# https://github.com/tensorflow/tensorflow/issues/62641
class FullyConnectedBmm(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(FullyConnectedBmm, self).__init__()
        self.bias = bias
        self.W = nn.Parameter(torch.empty(in_features, out_features))
        self.b = nn.Parameter(torch.empty(1, out_features)) if bias else None
        
        nn.init.trunc_normal_(self.W.data, std=.02)
        if self.bias:
            nn.init.constant_(self.b.data, 0)
        
    def forward(self, x):
        x = torch.matmul(x, self.W)
        if self.bias:
            x = x + self.b
        return x
    
# Same as FullyConnectedBmm but with extra-dim (0)
# to W to avoid promotion. 
# To be used with input shape len within [2,3]. Use 'FullyConnectedConv2d'
# or other shapes
class FullyConnected(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(FullyConnected, self).__init__()
        self.bias = bias
        self.W = nn.Parameter(torch.empty(1, in_features, out_features), requires_grad=True) # extra dim(0) for W
        self.b = nn.Parameter(torch.empty(1, out_features), requires_grad=True) if bias else None
        
        nn.init.trunc_normal_(self.W.data, std=.02)
        if self.bias:
            nn.init.constant_(self.b.data, 0)
        
    def forward(self, x):
        squeeze_needed = (len(x.shape) == 2)
        x = x @ self.W
        if self.bias:
            x = x + self.b
        return x.squeeze(0) if squeeze_needed else x
   
# FullyConnected using Conv2d. To be used at the least resort
class FullyConnectedConv2d(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(FullyConnectedConv2d, self).__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1, bias=bias)
        self.net3_in = Rearrange('b x y -> b y x ()')
        self.net3_out = Rearrange('b x y 1 -> b y x')
        self.net2_in = Rearrange('b x -> b x () ()')
        self.net2_out = Rearrange('b x 1 1 -> b x')
        nn.init.normal_(self.conv.weight, std=0.001)
        if self.conv.bias is not None:
            self.conv.bias.data.zero_()
    
    def forward(self, x):
        # [B,channels,in_features] -> [B,channels,out_features]
        # [B,in_features] -> [B,out_features]
        assert(len(x.shape) == 3 or len(x.shape) == 2)
        if len(x.shape) == 3:
            # transpose(2, 1) -> [B,in_features,channels]
            # unsqueeze(-1) -> [B,in_features,channels,1]
            # conv(in_features, out_features, 1) -> [B,out_features,channels,1]
            # transpose(1, 2) -> [B,channels,out_features, 1]
            # squeeze(-1) -> [B,channels,out_features]
            return self.net3_out(
                self.conv(
                    self.net3_in(x)
                )
            )
        elif len(x.shape) == 2:
            return self.net2_out(
                self.conv(
                    self.net2_in(x)
                )
            )
        else:
            raise NotImplementedError('Shape:'.format(x.shape))
