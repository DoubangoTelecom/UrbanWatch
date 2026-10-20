import torch, torch.nn as nn
import torch.nn.functional as F

from aocr.cct.utils.dynamic_tanh import DynamicTanh
from aocr.cct.utils.stn import StnAffine, StnTPS
from aocr.cct.utils.activations import ACTIVATIONS
from einops.layers.torch import Rearrange
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

# Based on https://github.com/clovaai/deep-text-recognition-benchmark/blob/e2117f2fb882b3c6085030500a260c113be27a63/model.py#L25

# https://huggingface.co/docs/transformers/v4.17.0/en/model_doc/segformer
kModelsParams = {
    'MiT-b0': {
        'depths': (2, 2, 2, 2),
        'embed_dims': (32, 64, 160, 256),
        'decoder_embed_dim': 256
    },
    'MiT-b0.75': {
        'depths': (2, 2, 2, 2),
        'embed_dims': (48, 96, 240, 384),
        'decoder_embed_dim': 256
    },
    'MiT-b1': {
        'depths': (2, 2, 2, 2),
        'embed_dims': (64, 128, 320, 512),
        'decoder_embed_dim': 256
    },
}

def nlc_to_nchw(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    n, _, _ = x.shape
    return x.permute(0, 2, 1).reshape(n, -1, h, w)


def nchw_to_nlc(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(start_dim=2, end_dim=-1).transpose(1, 2)


class OverlappingPatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, downsampling_factor: int):
        super().__init__()
        # TODO: Factorize the following if clauses in order to automatically retrieve k,s,p
        if downsampling_factor == 4:
            k, s, p = 7, downsampling_factor, 3
        elif downsampling_factor == 2:
            k, s, p = 3, downsampling_factor, 1
        else:
            raise NotImplementedError(
                f"OverlappingPatchEmbedding for downsampling_factor={downsampling_factor} is not implemented."
            )
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=k, stride=s, padding=p
        )  # noqa
        self.norm = DynamicTanh(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = nchw_to_nlc(x)
        x = self.norm(x)
        return x, (H, W)


class MLP(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = nchw_to_nlc(x)
        x = self.proj(x)
        return x


class MixFFN(nn.Module):
    def __init__(self, mlp_in_dim: int, mlp_hidden_dim: int, dropout_prob: float = 0.0):
        super().__init__()
        fc1 = nn.Conv2d(
            in_channels=mlp_in_dim,
            out_channels=mlp_hidden_dim,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        depth_wise_conv = nn.Conv2d(
            in_channels=mlp_hidden_dim,
            out_channels=mlp_hidden_dim,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=mlp_hidden_dim,
        )  # By setting groups=mlp_hidden_dim we will learn one filter
        # for each dimension in the input embedding space.
        fc2 = nn.Conv2d(
            in_channels=mlp_hidden_dim,
            out_channels=mlp_in_dim,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        dropout = nn.Dropout(p=dropout_prob)
        gelu = nn.GELU(approximate='tanh')
        self.layers = nn.Sequential(fc1, depth_wise_conv, gelu, dropout, fc2, dropout)

    def forward(self, x_in: torch.Tensor, h: int, w: int):
        # From the paper x_out = MLP(GELU(Conv3×3(MLP(x_in)))) + x_in
        batch_size, seq_len, hidden_dim = x_in.shape
        assert h * w == seq_len
        x = nlc_to_nchw(x_in, h, w)
        x = self.layers(x)
        x = nchw_to_nlc(x)
        return x_in + x


class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        reduction_ratio: int,
        num_heads: int,
        dropout_prob: float = 0.0,
    ):
        """
        This block in the original paper isn't explained extremely well, refer to the SRA block introduced in
        section 3.3 of "Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions"
        (https://arxiv.org/pdf/2102.12122) for a more detailed explanation.
        """
        super().__init__()

        self.reduction_conv = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=reduction_ratio,
            stride=reduction_ratio,
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_prob,
            batch_first=True,
        )
        self.norm = DynamicTanh(normalized_shape=embed_dim)

    def forward(self, x_in: torch.Tensor, h: int, w: int):
        q = x_in  # q.shape: (N, HXW, embed_dim)
        x = nlc_to_nchw(x_in, h, w)
        x = self.reduction_conv(x)
        x = nchw_to_nlc(x)
        k = v = self.norm(x)  # k.shape: (N, (HXW)/(reduction_ratio**2), embed_dim )
        x = self.attn(query=q, key=k, value=v)[0]
        return x_in + x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        efficient_attn_reduction_ratio: int,
        mix_ffn_expansion_ratio: int,
        dropout_prob=0.0,
    ):
        super().__init__()
        self.norm1 = DynamicTanh(embed_dim)
        self.efficient_self_attn = EfficientSelfAttention(
            embed_dim=embed_dim,
            reduction_ratio=efficient_attn_reduction_ratio,
            num_heads=num_heads,
            dropout_prob=dropout_prob,
        )
        self.norm2 = DynamicTanh(embed_dim)
        mlp_hidden_dim = int(embed_dim * mix_ffn_expansion_ratio)
        self.mix_ffn = MixFFN(
            mlp_in_dim=embed_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout_prob=dropout_prob,
        )

    def forward(self, x: torch.Tensor, h: int, w: int):
        x = x + self.efficient_self_attn(self.norm1(x), h, w)
        x = x + self.mix_ffn(self.norm2(x), h, w)
        return x

class MixTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_encoding_blocks: List[int] = (2, 2, 2, 2),
        embed_dims: List[int] = (32, 64, 160, 256),
        efficient_self_attention_num_heads: List[int] = (1, 2, 5, 8),
        efficient_self_attention_reduction_ratios: List[int] = (8, 4, 2, 1),
        mix_ffn_expansion_ratios: List[int] = (8, 8, 4, 4),
        downsampling_factors: List[int] = (4, 2, 2, 2),
        dropout_prob: float = 0.0,
    ):
        """
        :param in_channels: Number of channels in the input image.
        :param n_encoding_blocks: Number of transformer blocks in each stage.
        :param embed_dims: Embedding dimensions for each stage (C_i in the paper).
        :param efficient_self_attention_num_heads: Number of attention heads in each stage.
        :param efficient_self_attention_reduction_ratios: Reduction ratio for each stage's Effic. Self Attention block
        :param mix_ffn_expansion_ratios: Expansion ratios for each stage's MixFFN block. (E_i in the paper)
        :param downsampling_factors: At each encoding stage we down-sample H,W by the specified
            down-sapling factors, we do that by merging overlapping patches in the input as explained
            in section 3.1 "Overlapped Patch Merging" of the paper. In the original implementation these factors are
            4, 2, 2, 2 for MiT-B0

        Reference implementation:
            https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/backbones/mit.py
        """
        super().__init__()
        if not (
            len(n_encoding_blocks)
            == len(embed_dims)
            == len(efficient_self_attention_num_heads)
            == len(downsampling_factors)
            == len(efficient_self_attention_reduction_ratios)
            == len(mix_ffn_expansion_ratios)
        ):
            logger.error(
                f"The parameters '{n_encoding_blocks}', '{embed_dims}', '{efficient_self_attention_num_heads}', "
                f"'{efficient_self_attention_reduction_ratios}', '{mix_ffn_expansion_ratios}' "
                f"and '{downsampling_factors}' "
                f"should be list of integers"
                f" with the same cardinality."
            )

        self.depths = n_encoding_blocks
        self.overlapping_patch_embed = nn.ModuleList()
        self.encoding_blocks = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        n_stages = len(n_encoding_blocks)
        for stage_idx in range(n_stages):
            self.overlapping_patch_embed.append(
                OverlappingPatchEmbedding(
                    in_channels, embed_dims[stage_idx], downsampling_factors[stage_idx]
                )
            )
            self.encoding_blocks.append(
                nn.ModuleList(
                    [
                        TransformerEncoderLayer(
                            embed_dim=embed_dims[stage_idx],
                            num_heads=efficient_self_attention_num_heads[stage_idx],
                            efficient_attn_reduction_ratio=efficient_self_attention_reduction_ratios[
                                stage_idx
                            ],
                            mix_ffn_expansion_ratio=mix_ffn_expansion_ratios[stage_idx],
                            dropout_prob=dropout_prob,
                        )
                        for _ in range(n_encoding_blocks[stage_idx])
                    ]
                )
            )
            self.norm_layers.append(DynamicTanh(embed_dims[stage_idx]))
            in_channels = embed_dims[stage_idx]

    def forward(self, x):
        features = []
        number_of_stages = len(self.depths)
        for stage_idx in range(number_of_stages):
            x, (h, w) = self.overlapping_patch_embed[stage_idx](x)
            for encoding_block in self.encoding_blocks[stage_idx]:
                x = encoding_block(x, h, w)
            x = self.norm_layers[stage_idx](x)
            x = nlc_to_nchw(x, h, w)
            features.append(x)
        return features


class MLPDecoder(nn.Module):
    def __init__(
        self,
        embed_dims: List[int],
        decoder_embed_dim: int,
        num_classes: int,
        dropout_prob: float = 0.0,
    ):
        """
        Reference implementation:
            https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/decode_heads/segformer_head.py
        :param embed_dims: [C_1, C_2, C_3, C_4] in the Paper
        :param decoder_embed_dim: C in the Paper.
        :param num_classes: Num classes to predict.
        :param dropout_prob: prob for Dropout layer just before the last linear layer.
        """
        super().__init__()
        self.linear_layers = nn.ModuleList(
            [MLP(embedding_dim, decoder_embed_dim) for embedding_dim in embed_dims]
        )
        self.relu = nn.ReLU()
        n_stages = len(embed_dims)
        self.linear_fuse = nn.Conv2d(
            in_channels=decoder_embed_dim * n_stages,
            out_channels=decoder_embed_dim,
            kernel_size=1,
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.linear_pred = nn.Conv2d(decoder_embed_dim, num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor]):
        new_features = []
        target_resolution = features[0].shape[2:]  # In the original implementation
        # this resolution corresponds to (H/4,W/4), where H and W are the height and width of the original image.
        for i in range(len(features)):
            encoder_output = features[i]
            n, _, h, w = encoder_output.shape
            out = self.linear_layers[i](encoder_output)
            out = nlc_to_nchw(out, h, w)
            up_sampled_out = F.interpolate(
                out, size=target_resolution, mode="bilinear", align_corners=False
            )
            new_features.append(up_sampled_out)
        x = torch.cat(new_features, 1)
        x = self.relu(self.linear_fuse(x))
        x = self.dropout(x)
        x = self.linear_pred(x)
        return x


class SegFormer(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        depths: List[int] = (2, 2, 2, 2),
        embed_dims: List[int] = (32, 64, 160, 256),
        encoder_efficient_self_attention_num_heads: List[int] = (1, 2, 5, 8),
        encoder_efficient_self_attention_reduction_ratios: List[int] = (8, 4, 2, 1),
        encoder_mix_ffn_expansion_ratios: List[int] = (8, 8, 4, 4),
        encoder_downsampling_factors: List[int] = (4, 2, 2, 2),
        decoder_embed_dim: int = 256,
        num_classes: int = 19,
        dropout_prob: float = 0.0,
    ):
        """
        Check Figure 2. of the paper (https://arxiv.org/pdf/2105.15203) and section "Details of MiT Series" to have
        a deeper understanding of the parameters presented in this implementation.
        Here, the default parameters for the Mix Transformer (MiT) encoder correspond to the ones of MiT-B0
        as detailed in Table 6 of the Paper.
        :param in_channels: Number of channels in the input image.
        :param depths: Number of encoder layers in each stage.  (L_i in the paper)
        :param embed_dims: Embedding dimensions for each stage (C_i in the paper).
        :param encoder_efficient_self_attention_num_heads: Number of attention heads in each stage. (N_i in the paper)
        :param encoder_efficient_self_attention_reduction_ratios: Reduction ratio in each stage to improve time
            complexity while computing the Efficient Self Attention block. (R_i in the paper)
        :param encoder_mix_ffn_expansion_ratios: Expansion ratios for each stage's MixFFN block. (E_i in the paper)
        :param encoder_downsampling_factors: At each encoding stage we down-sample H,W by the specified
            down-sampling factors, we do that by merging overlapping patches in the input as explained
            in section 3.1 "Overlapped Patch Merging" of the paper.
        :param decoder_embed_dim: Embedding dimension for decoder's MLP (C in the paper)
        :param num_classes: Number of classes to predict from the segmentation model.
        """
        super().__init__()
        self.encoder = MixTransformer(
            in_channels=in_channels,
            n_encoding_blocks=depths,
            embed_dims=embed_dims,
            efficient_self_attention_num_heads=encoder_efficient_self_attention_num_heads,
            efficient_self_attention_reduction_ratios=encoder_efficient_self_attention_reduction_ratios,
            mix_ffn_expansion_ratios=encoder_mix_ffn_expansion_ratios,
            downsampling_factors=encoder_downsampling_factors,
            dropout_prob=dropout_prob,
        )
        self.decoder = MLPDecoder(
            embed_dims=embed_dims,
            decoder_embed_dim=decoder_embed_dim,
            num_classes=num_classes,
            dropout_prob=dropout_prob,
        )

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        # TODO(dmi): next line commented by me
        #out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out

    def print_num_parameters(self):
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(
            f"Encoder parameters: {count_parameters(self.encoder) / 1e6} M, "
            f"Decoder parameters: {count_parameters(self.decoder) / 1e6} M"
        )

class AOCR(nn.Module):
    def __init__(self, cfg, training = True, onnx_export = False):
        super(AOCR, self).__init__()
        self.export_mode = 'all'
        self.cfg = cfg
        self.is_training = training

        self.alphabet_size = len(list(cfg.model.alphabet)) + 1 # +1 for CTC blank character

        if cfg.model.stn_type == 'affine': 
            self.stn = StnAffine((cfg.model.imgH, cfg.model.imgW), 1 if cfg.model.grayscale else 3)
        elif cfg.model.stn_type == 'tps':
            self.stn = StnTPS(cfg.model.stn_tps_num_points, (cfg.model.imgH, cfg.model.imgW), (cfg.model.imgH, cfg.model.imgW), 1 if cfg.model.grayscale else 3)
        elif cfg.model.stn_type == 'none':
            self.stn = nn.Identity()
        else:
            raise NotImplementedError(f'Invalid stn_type ({cfg.model.stn_type})')

        self.segf = SegFormer(
            in_channels = 1 if cfg.model.grayscale else 3,
            num_classes=self.alphabet_size,
            **kModelsParams[cfg.model.MiT]
        )
        self.Projection = nn.Sequential(
            Rearrange('n c h w -> n c w h'),
            nn.Conv2d(self.alphabet_size, self.alphabet_size, 3, (2, 1), (0, 1)),
            nn.BatchNorm2d(self.alphabet_size),
            ACTIVATIONS[cfg.model.projection.activation],
            nn.Conv2d(self.alphabet_size, self.alphabet_size, 3, (2, 1), (0, 1)),
            nn.BatchNorm2d(self.alphabet_size),
            ACTIVATIONS[cfg.model.projection.activation],
            nn.Conv2d(self.alphabet_size, self.alphabet_size, 3, (2, 1), (0, 1)),
            nn.BatchNorm2d(self.alphabet_size),
            ACTIVATIONS[cfg.model.projection.activation],
            nn.Conv2d(self.alphabet_size, self.alphabet_size, 2, (1, 1), (0, 0)),
            #nn.BatchNorm2d(self.alphabet_size),
            #ACTIVATIONS[cfg.model.projection.activation],
            
            # [h] must be equal to #1
            Rearrange('n c () w -> n w c') if cfg.train.loss.type == 'ctc' else nn.Identity()
        )
        '''self.Projection = nn.Sequential(
            nn.Conv2d(self.alphabet_size, self.alphabet_size, 3, 2, 0),
            nn.BatchNorm2d(self.alphabet_size),
            ACTIVATIONS[cfg.model.projection.activation],
            nn.Conv2d(self.alphabet_size, self.alphabet_size, 3, 2, 0),

            # The last part could be done using single command by fails with:
            #   UnsupportedShuffleLayerError in op /Reshape: Failed to determine type of layer to create in node /Reshape
            #Rearrange('n c h w -> n (w h) c')
            #   decomposition using 2 commands
            Rearrange('n c h w -> n c (w h)'),
            Rearrange('n c s -> n s c'),
        )'''

        self.seq_len = self.forward(torch.zeros((1, 1 if cfg.model.grayscale else 3, cfg.model.imgH, cfg.model.imgW))).shape[1]
        if cfg.train.loss.type == 'ctc':
            assert self.seq_len >= cfg.model.max_len * 2, f'Sequence length is too short: {self.seq_len}'
        else:
            assert cfg.train.loss.type == "ce", f'loss type ({cfg.train.loss.type}) is invalid'
            assert self.seq_len == cfg.model.max_len, f'Invalid sequence length {self.seq_len} != {cfg.model.max_len}'

        self.is_onnx_export = onnx_export
        
    def forward(self, x):
        x = self.stn(x)
        x = self.segf(x)
        x = self.Projection(x)
        if self.cfg.train.loss.type == 'ce':
            #x = F.interpolate(x, size=(1, self.cfg.model.max_len), mode="bilinear", align_corners=False)
            x = x.narrow(-1, 0, self.cfg.model.max_len)
            x = Rearrange('n c h w -> n (h w) c')(x)
        
        if getattr(self, 'is_onnx_export', False):
            x = x.view((-1, self.seq_len, self.alphabet_size)) # ONNX: should be useless, instead 2 last dims become dynamic if not set and lead to [-1,-1,-1]
        if not self.is_training:
            x = F.softmax(x, dim=-1)

        return x
