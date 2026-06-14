import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha :float=0.25, gamma :float=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none', reduce=False)
    
    def forward(self, inputs, targets) -> torch.Tensor:
        loss = self.criterion(inputs, targets)
        pt = torch.exp(-loss)
        return (self.alpha * (1-pt)**self.gamma * loss).mean()


        
        
    

