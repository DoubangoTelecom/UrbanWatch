import torch, torch.nn as nn
from aocr.cct.cct import CCT_MNV4
from aocr.cct.utils.stn import StnAffine, StnTPS
from aocr.cct.utils.fullyconnected import FullyConnected
import torch.nn.functional as F

class AOCR(nn.Module):
    def __init__(self, cfg, training = True):
        super(AOCR, self).__init__()
        self.export_mode = 'all'
        self.cfg = cfg
        self.is_training = training
        
        # Create STN object
        if self.cfg.model.stn_type == 'affine': 
            self.stn = StnAffine((self.cfg.model.imgH, self.cfg.model.imgW), 1 if self.cfg.model.grayscale else 3)
        else:
            self.stn = StnTPS(self.cfg.model.stn_tps_num_points, (self.cfg.model.imgH, self.cfg.model.imgW), (self.cfg.model.imgH, self.cfg.model.imgW), 1 if self.cfg.model.grayscale else 3)
        
        # Create the Transformer
        self.cct = CCT_MNV4(
            imgh=cfg.model.imgH, imgw=cfg.model.imgW,
            n_input_channels=1 if cfg.model.grayscale else 3,
            mnv4_block_size=cfg.model.mnv4.block_size,
            mnv4_width_mult=cfg.model.mnv4.width_mult,
            mnv4_out_stage=cfg.model.mnv4.out_stage,
            seq_pool=cfg.model.cct.seq_pool,
            dropout=cfg.model.cct.dropout,
            attention_dropout=cfg.model.cct.attention_dropout,
            stochastic_depth=cfg.model.cct.stochastic_depth,
            num_layers=cfg.model.cct.num_layers,
            num_heads=cfg.model.cct.num_heads,
            mlp_ratio=cfg.model.cct.mlp_ratio,
            positional_embedding=cfg.model.cct.positional_embedding
        )
        
        # Make sure sequence length not too short (at least 3 times max_len)
        assert self.cct.tokenizer.sequence_length > 3 * cfg.model.max_len, 'Sequence length is too short ({} < {})'.format(self.cct.tokenizer.sequence_length, 3 * cfg.model.max_len)
        
        # CTC projection
        alphabet_size = len(list(cfg.model.alphabet)) + 1 # +1 for CTC blank character
        self.projection = nn.Sequential(
            nn.Dropout(cfg.model.projection_dropout),
            FullyConnected(self.cct.tokenizer.output_channels, alphabet_size)
        )
    
    def _set_export_mode(self, mode: str):
        # 'stn': export STN module only (with ablation)
        # '-stn': export everything except the STN module
        # 'all: export everything
        assert mode in ['stn', '-stn', 'all'], '{} is invalid export mode'
        self.stn._set_export_mode(mode == 'stn')
        self.export_mode = mode
    
    def _stn_set_trainable(self, yes :bool):
        self.stn._set_trainable(yes)
        
    def forward(self, x):
        if self.export_mode != '-stn':
            x = self.stn(x)
        if self.export_mode == 'stn':
            return x
        x = self.cct(x)
        x = self.projection(x)
        if not self.is_training:
            x = F.softmax(x, dim=-1)
        
        return x
        
