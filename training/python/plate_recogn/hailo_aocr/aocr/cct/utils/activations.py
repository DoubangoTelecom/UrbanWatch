import torch
   
ACTIVATIONS = { 
    'ReLU': torch.nn.ReLU(inplace=True), 
    'LeakyReLU': torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
    'GELU': torch.nn.GELU(approximate='tanh'),
    'SiLU': torch.nn.SiLU(),
    'PReLU': torch.nn.PReLU(),
    'Tanh': torch.nn.Tanh(),
    'Identity': torch.nn.Identity()
}