import torch, torch.nn as nn
from .mobilenetv4 import MobileNetV4
from .resnet import ResNet
from .vgg import VGG
from .fullyconnected import FullyConnected
from .activations import ACTIVATIONS
from .mobilenetv3 import mobilenet_v3_small, mobilenet_v3_large
from .mobilevit_v3_v2 import MobileViTv3_v2
from .mobilevit_v3_v1 import MobileViTv3_v1
from .colornet import ColorNet
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class FlattenLast(nn.Module):
    def forward(self, x):
        #return Rearrange('b c h w -> b (c h w)')(x) # ONNX:  translate to "Reshape" layer which doesn't work with Hailo and outpus WARN on VxDelegate
        return x.squeeze((-1, -2)) if (torch.__version__ >= "2.0.0") else x.squeeze(-1).squeeze(-1) # ONNX: translate to "Squeeze" and works fine with Hailo

# https://www.kaggle.com/code/nanguyen/arcface-loss
class FeatSeparator(nn.Module):
    def __init__(self, num_classes, embedding_size, margin, scale):
        """
        ArcFace: Additive Angular Margin Loss for Deep Face Recognition
        (https://arxiv.org/pdf/1801.07698.pdf)
        Args:
            num_classes: The number of classes in your training dataset
            embedding_size: The size of the embeddings that you pass into
            margin: m in the paper, the angular margin penalty in radians
            scale: s in the paper, feature scale
        """
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.margin = margin
        self.scale = scale
        
        self.W = torch.nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.xavier_normal_(self.W)
        
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (None, embedding_size)
            labels: (None,)
        Returns:
            loss: scalar
        """
        if self.margin > 0:
            cosine = self.get_cosine(embeddings) # (None, n_classes)
            mask = self.get_target_mask(labels) # (None, n_classes)
            cosine_of_target_classes = cosine[mask == 1] # (None, )
            modified_cosine_of_target_classes = self.modify_cosine_of_target_classes(
                cosine_of_target_classes
            ) # (None, )
            diff = (modified_cosine_of_target_classes - cosine_of_target_classes).unsqueeze(1) # (None,1)
            logits = cosine + (mask * diff) # (None, n_classes)
        else:
            logits = F.linear(embeddings.view((-1, self.embedding_size)), self.W)

        return self.scale_logits(logits)

    def get_cosine(self, embeddings):
        """
        Args:
            embeddings: (None, embedding_size)
        Returns:
            cosine: (None, n_classes)
        """
        cosine = F.linear(F.normalize(embeddings.view((-1, self.embedding_size))), F.normalize(self.W))
        return cosine
    
    def get_target_mask(self, labels):
        """
        Args:
            labels: (None,)
        Returns:
            mask: (None, n_classes)
        """
        batch_size = labels.size(0)
        onehot = torch.zeros(batch_size, self.num_classes, device=labels.device)
        onehot.scatter_(1, labels.unsqueeze(-1), 1)
        return onehot
        
    def modify_cosine_of_target_classes(self, cosine_of_target_classes):
        """
        Args:
            cosine_of_target_classes: (None,)
        Returns:
            modified_cosine_of_target_classes: (None,)
        """
        eps = 1e-6
        # theta in the paper
        angles = torch.acos(torch.clamp(cosine_of_target_classes, -1 + eps, 1 - eps))
        return torch.cos(angles + self.margin)
    
    def scale_logits(self, logits):
        """
        Args:
            logits: (None, n_classes)
        Returns:
            scaled_logits: (None, n_classes)
        """
        return logits * self.scale

class KlassModel(nn.Module):
    def __init__(self, cfg, training :bool=True):
        super(KlassModel, self).__init__()
        self.export_mode = 'all'
        self.cfg = cfg
        self.is_training = training
        
        # Create backbone
        if cfg.model.backbone.type == 'mnv4':
            self.backbone = MobileNetV4(**cfg.model.backbone.MobileNetV4._asdict())
        elif cfg.model.backbone.type == 'mnv3':
            self.backbone = mobilenet_v3_small(**cfg.model.backbone.MobileNetV3._asdict()) if cfg.model.backbone.MobileNetV3.block_size == 'small' else \
                mobilenet_v3_large(**cfg.model.backbone.MobileNetV3._asdict())
        elif cfg.model.backbone.type == 'mobilevitv3_v2':
            self.backbone = MobileViTv3_v2(**cfg.model.backbone.MobileViTv3_v2._asdict())
        elif cfg.model.backbone.type == 'mobilevitv3_v1':
            self.backbone = MobileViTv3_v1(**cfg.model.backbone.MobileViTv3_v1._asdict())
        elif cfg.model.backbone.type == 'resnet':
            self.backbone = ResNet(**cfg.model.backbone.ResNet._asdict())
        elif cfg.model.backbone.type == 'vgg':
            self.backbone = VGG(**cfg.model.backbone.Vgg._asdict())
        elif cfg.model.backbone.type == 'colornet':
            self.backbone = ColorNet(**cfg.model.backbone.ColorNet._asdict())
        else:
            assert False, f'{cfg.model.backbone.type} is not a valid backbone type'

        # Create classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if cfg.model.projection.fully_connected:
            self.classifier = nn.Sequential(
                nn.Dropout2d(cfg.model.projection.dropout),
                Rearrange('b c h w -> b (c h w)'),

                FullyConnected(self.backbone.output_channels, cfg.model.projection.hidden_channels, bias=False),
                nn.BatchNorm1d(cfg.model.projection.hidden_channels),
                ACTIVATIONS[cfg.model.projection.activation],

                FullyConnected(cfg.model.projection.hidden_channels, cfg.model.num_classes, bias=False)
            ) if cfg.model.projection.hidden_channels > 0 else \
            nn.Sequential(
                nn.Dropout2d(cfg.model.projection.dropout),
                Rearrange('b c h w -> b (c h w)'),
                FullyConnected(self.backbone.output_channels, cfg.model.num_classes, bias=False)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout2d(cfg.model.projection.dropout),

                nn.Conv2d(self.backbone.output_channels, cfg.model.projection.hidden_channels, 1),
                nn.BatchNorm2d(cfg.model.projection.hidden_channels),
                ACTIVATIONS[cfg.model.projection.activation],

                nn.Conv2d(cfg.model.projection.hidden_channels, cfg.model.num_classes, 1),
                FlattenLast()
            ) if cfg.model.projection.hidden_channels > 0 else \
            nn.Sequential(
                nn.Dropout2d(cfg.model.projection.dropout),
                nn.Conv2d(self.backbone.output_channels, cfg.model.num_classes, 1),
                FlattenLast()
            )

        # Features seperator
        self.featsep = lambda x,y: None
        if cfg.model.feat_separator.enabled:
            self.featsep = FeatSeparator(
                num_classes=cfg.model.num_classes, 
                embedding_size=self.backbone.output_channels,
                margin=cfg.model.feat_separator.margin,
                scale=cfg.model.feat_separator.scale
            )
        
    def forward(self, x, labels=None):
        x = self.backbone(x)
        x = self.avgpool(x)
        feat_logits = self.featsep(x, labels) if self.is_training else None
        x = self.classifier(x)
        if not self.is_training:
            x = F.softmax(x, dim=-1)
        
        return (x, feat_logits) if self.is_training else x
        
