import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, type: str, alpha :float=0.25, gamma :float=2.0, blank :int=0):
        """https://github.com/jimitshah77/Focal-CTC-OMR"""
        super(FocalLoss, self).__init__()
        self.type = type
        self.alpha = alpha
        self.gamma = gamma
    
    def _focal(self, loss):
        pt = torch.exp(-loss)
        return (self.alpha * (1-pt)**self.gamma * loss).mean()
    
    @staticmethod
    def build(type: str, alpha :float=0.25, gamma :float=2.0, blank :int=0, label_smoothing:float=0.0) -> nn.Module:
        if type == 'ctc':
            return FocalLossCTC(alpha, gamma, blank)
        else:
            return FocalLossCE(alpha, gamma, label_smoothing)
        
class FocalLossCTC(FocalLoss):
    def __init__(self, alpha :float=0.25, gamma :float=2.0, blank :int=0):
        """https://github.com/jimitshah77/Focal-CTC-OMR"""
        super(FocalLossCTC, self).__init__('ctc', alpha, gamma, blank)
        self.criterion = torch.nn.CTCLoss(blank=blank, reduction='none', zero_infinity=True)
        
    def forward(self, log_probs: torch.Tensor, targets: torch.Tensor, input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        return self._focal(self.criterion(log_probs, targets, input_lengths, target_lengths))
    
class FocalLossCE(FocalLoss):
    def __init__(self, alpha :float=0.25, gamma :float=2.0, label_smoothing:float=0.0):
        """https://github.com/jimitshah77/Focal-CTC-OMR"""
        super(FocalLossCE, self).__init__('ce', alpha, gamma)
        self.criterion = torch.nn.CrossEntropyLoss(
            reduction='none', 
            reduce=False, 
            label_smoothing=label_smoothing
            )
        
    def forward(self, inputs, targets) -> torch.Tensor:
        return self._focal(self.criterion(inputs, targets))
        