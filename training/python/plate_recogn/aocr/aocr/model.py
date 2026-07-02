import torch.nn as nn
from aocr.cct.cct import CCT_AOCR
from aocr.cct.utils.stn import StnAffine, StnTPS
from aocr.cct.utils.fullyconnected import FullyConnected
from einops.layers.torch import Rearrange
import torch.nn.functional as F

class ReshapeProjection(nn.Module):
    def __init__(self, shape):
        super(ReshapeProjection, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)

class AOCR(nn.Module):
    def __init__(self, cfg, training = True):
        super(AOCR, self).__init__()
        self.export_mode = 'all'
        self.cfg = cfg
        self.is_training = training
        
        # Create STN object
        if self.cfg.model.stn_type == 'affine': 
            self.stn = StnAffine((self.cfg.model.imgH, self.cfg.model.imgW), 1 if self.cfg.model.grayscale else 3)
        elif self.cfg.model.stn_type == 'tps':
            self.stn = StnTPS(self.cfg.model.stn_tps_num_points, (self.cfg.model.imgH, self.cfg.model.imgW), (self.cfg.model.imgH, self.cfg.model.imgW), 1 if self.cfg.model.grayscale else 3)
        elif self.cfg.model.stn_type == 'none':
            self.stn = None
        else:
            raise NotImplementedError('Invalid stn_type ({})'.format(self.cfg.model.stn_type))
        
        # Create the Transformer
        self.cct = CCT_AOCR(
            imgh=cfg.model.imgH, imgw=cfg.model.imgW,
            n_input_channels=1 if cfg.model.grayscale else 3,
            **cfg.model.cct._asdict()
        )
        
        # Make sure sequence length not too short (at least 3 times max_len)
        if cfg.train.loss.type == 'ctc':
            assert self.cct.tokenizer.sequence_length > 3 * cfg.model.max_len, 'Sequence length is too short ({} < {})'.format(self.cct.tokenizer.sequence_length, 3 * cfg.model.max_len)
        
        # Classification
        self.alphabet_size = len(list(cfg.model.alphabet)) + 1 # +1 for CTC blank character
        self.sequence_length = self.cct.tokenizer.sequence_length
        if not self.cct.classifier.seq_pool:
            self.sequence_length += 1
        if cfg.train.loss.type == 'ce':
            if cfg.model.projection.fc:
                if False: # using Conv2d: faster, more accurate, lower weights size...
                    self.projection = nn.Sequential(
                        Rearrange('b c h -> b (c h) () ()'),
                        nn.Dropout2d(cfg.model.projection.dropout),
                        nn.Conv2d((self.sequence_length*self.cct.tokenizer.output_channels), (cfg.model.max_len*self.alphabet_size), 1), # Using Conv2d instead of FullyConnected (not NPU friendly)
                        ReshapeProjection((-1, cfg.model.max_len, self.alphabet_size)),
                    )
                else:
                    self.projection = nn.Sequential(
                        ReshapeProjection((-1, (self.sequence_length*self.cct.tokenizer.output_channels))),
                        nn.Dropout(cfg.model.projection.dropout),
                        FullyConnected((self.sequence_length*self.cct.tokenizer.output_channels), (cfg.model.max_len*self.alphabet_size)),
                        ReshapeProjection((-1, cfg.model.max_len, self.alphabet_size)),
                    )
                
            else:
                if True: # using Conv2d: faster, more accurate, lower weights size...
                    self.projection = nn.Sequential(
                        Rearrange('b c w -> b c w ()'),
                        nn.Dropout2d(cfg.model.projection.dropout),
                        nn.Conv2d(self.sequence_length, cfg.model.max_len, 1),
                        Rearrange('b h c w -> b c h w'),
                        nn.Dropout2d(cfg.model.projection.dropout),
                        nn.Conv2d(self.cct.tokenizer.output_channels, self.alphabet_size, 1),
                        Rearrange('b c h w -> b h (c w)'),
                    )
                else:
                    self.projection = nn.Sequential(
                        Rearrange('batch seqlen channels -> batch channels seqlen'),
                        nn.Dropout(cfg.model.projection.dropout),
                        FullyConnected(self.sequence_length, cfg.model.max_len),
                        Rearrange('batch channels seqlen -> batch seqlen channels'),
                        nn.Dropout(cfg.model.projection.dropout),
                        FullyConnected(self.cct.tokenizer.output_channels, self.alphabet_size)
                    )
        else:
            self.projection = nn.Sequential(
                nn.Dropout(cfg.model.projection.dropout),
                FullyConnected(self.cct.tokenizer.output_channels, self.alphabet_size)
            )
    
    def _set_export_mode(self, mode: str):
        # 'stn': export STN module only (with ablation)
        # '-stn': export everything except the STN module
        # 'all: export everything
        assert mode in ['stn', '-stn', 'all'], '{} is invalid export mode'
        if self.stn is None and mode == 'stn':
            raise Exception('No STN model to export')
        if not self.stn is None:
            self.stn._set_export_mode(mode == 'stn')
        self.export_mode = mode
    
    def _stn_set_trainable(self, yes :bool):
        self.stn._set_trainable(yes)
        
    def forward(self, x):
        if not self.stn is None:
            if self.export_mode != '-stn':
                x = self.stn(x)
            if self.export_mode == 'stn':
                return x
        x = self.cct(x)
        x = self.projection(x)      
        if not self.is_training:
            x = F.softmax(x, dim=-1)
        
        return x
        
