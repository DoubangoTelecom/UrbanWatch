import torch.nn as nn
from klass.mobilenetv4 import MobileNetV4
from klass.fullyconnected import FullyConnected
import torch.nn.functional as F

class KlassModel(nn.Module):
    def __init__(self, cfg, training = True):
        super(KlassModel, self).__init__()
        self.export_mode = 'all'
        self.cfg = cfg
        self.is_training = training
        
        self.backbone = MobileNetV4(**cfg.model.backbone.MobileNetV4._asdict())
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.model.dropout),
            FullyConnected(self.backbone.output_channels, cfg.model.num_classes)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(-1, self.backbone.output_channels)
        x = self.classifier(x) 
        if not self.is_training:
            x = F.softmax(x, dim=-1)
        
        return x
        
