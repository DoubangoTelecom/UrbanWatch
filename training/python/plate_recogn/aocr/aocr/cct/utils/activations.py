import torch

class Linear(torch.nn.Module):
    def forward(self, output):
        return output
    
ACTIVATIONS = { 
    'ReLU': torch.nn.ReLU(inplace=True), 
    'LeakyReLU': torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
    'GELU': torch.nn.GELU(approximate='tanh'),
    'Linear': Linear
}