import torch
import torch.nn.functional as F
import torch.nn as nn

""" 
    FullyConnected from Pytorch return E [ops/vsi_nn_op_fullconnect_relu.c:op_check:234]Inputs/Outputs data type not support: ASYM UINT8, ASYM UINT8,  INT8 
    on Debix A (VX Delegate)
"""

# matmul (@) is promoted to FullyConnected causing issues
# https://github.com/tensorflow/tensorflow/issues/62641
class FullyConnectedDeprecated(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(FullyConnectedDeprecated, self).__init__()
        self.bias = bias
        self.W = nn.Parameter(torch.empty(in_features, out_features))
        self.b = nn.Parameter(torch.empty(1, out_features)) if bias else None
        
        nn.init.trunc_normal_(self.W.data, std=.02)
        if self.bias:
            nn.init.constant_(self.b.data, 0)
        
    def forward(self, x):
        x = x @ self.W
        if self.bias:
            x = x + self.b
        return x
    
class FullyConnected(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(FullyConnected, self).__init__()
        self.conv = torch.nn.Conv2d(in_features, out_features, kernel_size=1, bias=bias)
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
            return self.conv(x.transpose(2, 1).unsqueeze(-1)).transpose(1, 2).squeeze(-1)
        elif len(x.shape) == 2:
            return self.conv(x[:,:,None,None]).squeeze([-2,-1])
        else:
            raise NotImplementedError('Shape:'.format(x.shape))
