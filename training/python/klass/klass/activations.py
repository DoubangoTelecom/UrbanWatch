import torch
from .geffnet.activations import HardSwish as HS
   
ACTIVATIONS = { 
    'ReLU': torch.nn.ReLU(inplace=True), 
    'LeakyReLU': torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
    'GELU': torch.nn.GELU(approximate='tanh'),
    'SiLU': torch.nn.SiLU(),
    'Hardswish': HS(inplace=False), # 'inplace' must be False
    'Tanh': torch.nn.Tanh(),
    'Identity': torch.nn.Identity()
}