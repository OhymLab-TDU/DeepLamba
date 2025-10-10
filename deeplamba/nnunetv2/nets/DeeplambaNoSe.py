import re
import time
import math
import numpy as np
from functools import partial
from typing import Optional, Union, Type, List, Tuple, Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
# from torchsummary import summary
from fvcore.nn import FlopCountAnalysis, flop_count
from thop import profile

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from monai.networks.blocks import SubpixelUpsample


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        #         self.depthwise_conv = nn.Conv2d(4 * dim, 4 * dim, kernel_size=1, groups=4 * dim)
        #         self.pointwise_conv = nn.Conv2d(4 * dim, 2 * dim, kernel_size=1)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        # x = x.permute(0, 3, 1, 2)
        # x = self.depthwise_conv(x)
        # x = self.pointwise_conv(x)

        return x


class PatchExpand(nn.Module):
    """
    Reference: https://arxiv.org/pdf/2105.05537.pdf
    """

    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # B, C, H, W ==> B, H, W, C
        x = self.expand(x)
        B, H, W, C = x.shape

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        x = x.reshape(B, H * 2, W * 2, C // 4)

        return x


class FinalPatchExpand_X4(nn.Module):
    """
    Reference:
        - GitHub: https://github.com/HuCaoFighting/Swin-Unet/blob/main/networks/swin_transformer_unet_skip_expand_decoder_sys.py
        - Paper: https://arxiv.org/pdf/2105.05537.pdf
    """

    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        # self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # H, W = self.input_resolution
        x = x.permute(0, 2, 3, 1)  # B, C, H, W ==> B, H, W, C
        x = self.expand(x)
        B, H, W, C = x.shape
        # B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        # x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        x = x.reshape(B, H * self.dim_scale, W * self.dim_scale, self.output_dim)

        return x  #.permute(0, 3, 1, 2)


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class _ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate, norm_layer, norm_kwargs):
        super(_ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, norm_kwargs):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


class _ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, norm_layer, norm_kwargs=None):
        super(_ASPP, self).__init__()
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer, norm_kwargs)
        self.b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer, norm_kwargs)
        self.b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer, norm_kwargs)
        self.b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x


class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class SemanticAttention(nn.Module):

    def __init__(self, dim, n_cls):
        super().__init__()
        self.dim = dim
        self.n_cls = n_cls
        self.softmax = nn.Softmax(dim=-1)

        self.mlp_cls_q = nn.Linear(self.dim, self.n_cls)
        self.mlp_cls_k = nn.Linear(self.dim, self.n_cls)
        self.mlp_v = nn.Linear(self.dim, self.dim)

        self.mlp_res = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(0.1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        '''
        Args:
            x: input features with shape of (B, N, C)
        returns:
            class_seg_map: (B, N, K)
            gated feats: (B, N, C)
        '''

        seg_map = self.mlp_cls_q(x)
        seg_ft = self.mlp_cls_k(x)

        feats = self.mlp_v(x)

        seg_score = seg_map @ seg_ft.transpose(-2, -1)
        seg_score = self.softmax(seg_score)

        feats = seg_score @ feats
        feats = self.mlp_res(feats)
        feats = self.proj_drop(feats)

        feat_map = self.gamma * feats + x

        return seg_map, feat_map


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        if bn:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            self.relu = nn.ReLU(inplace=True) if relu else None
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=True)
            self.bn = None
            self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8, vision=1, groups=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision, dilation=vision,
                      relu=False, groups=groups)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 2,
                      dilation=vision + 2, relu=False, groups=groups)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1, groups=groups),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1,
                      groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 4,
                      dilation=vision + 4, relu=False, groups=groups)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


class DeeplambaNoSe(nn.Module):
    def __init__(
            self,
            in_chans=1,
            out_chans=13,
            feat_size=[48, 96, 192, 384, 768],
            depths=[2, 2, 7, 2],
            drop_path_rate=0.2,
            attn_drop_rate=0.,
            layer_scale_init_value=1e-6,
            hidden_size: int = 768,
            norm_name="instance",
            res_block: bool = True,
            spatial_dims=2,
    ) -> None:
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.feat_size = feat_size
        self.depth = depths
        self.drop_path_rate = drop_path_rate
        self.hidden_size = hidden_size
        self.resblock = res_block
        self.spatial_dims = spatial_dims
        self.norm_name = norm_name

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        # start with 0, end with drop_path_rate

        # self.stem = nn.Sequential(
        #     nn.Conv2d(in_chans, feat_size[0], kernel_size=7, stride=2, padding=3),
        #     nn.InstanceNorm2d(feat_size[0]),
        # )

        self.vssBlock1 = nn.ModuleList([VSSBlock(
            hidden_dim=feat_size[1],  # 96
            drop_path=dpr[i],
            norm_layer=nn.LayerNorm,
            attn_drop_rate=attn_drop_rate,
            d_state=16,
        ) for i in range(depths[0])
        ])

        self.vssBlock2 = nn.ModuleList([VSSBlock(
            hidden_dim=feat_size[2],  # 192
            drop_path=dpr[i + depths[0]],
            norm_layer=nn.LayerNorm,
            attn_drop_rate=attn_drop_rate,
            d_state=16,
        ) for i in range(depths[1])
        ])

        self.vssBlock3 = nn.ModuleList([VSSBlock(
            hidden_dim=feat_size[3],  # 384
            drop_path=dpr[i + depths[0] + depths[1]],
            norm_layer=nn.LayerNorm,
            attn_drop_rate=attn_drop_rate,
            d_state=16,
        ) for i in range(depths[2])
        ])

        self.vssBlock4 = nn.ModuleList([VSSBlock(
            hidden_dim=feat_size[4],  # 768
            drop_path=dpr[i + depths[0] + depths[1] + depths[2]],
            norm_layer=nn.LayerNorm,
            attn_drop_rate=attn_drop_rate,
            d_state=16,
        ) for i in range(depths[3])
        ])

        # self.seat1 = SemanticAttention(dim=96, n_cls=96)
        # self.seat2 = SemanticAttention(dim=192, n_cls=192)
        # self.seat3 = SemanticAttention(dim=384, n_cls=384)
        # self.seat4 = SemanticAttention(dim=768, n_cls=768)

        # self.conv1 = nn.Conv2d(in_channels=288, out_channels=feat_size[4], kernel_size=1,
        #                        stride=1)  # 96+192+384+768=1440 ----> 768
        self.conv1 = nn.Conv2d(in_channels=384, out_channels=self.out_chans, kernel_size=1,
                               stride=1)
        self.patchembed = PatchEmbed2D(in_chans=in_chans)  # first downsample
        self.patchmerge = nn.ModuleList([PatchMerging2D(
            dim=feat_size[i]
        ) for i in range(1, 4)
        ])  # 96, 192 ,768

        self.brfb = BasicRFB(288, 288)

        self.patchexpand4 = PatchExpand(input_resolution=None, dim=feat_size[4], dim_scale=2)
        self.patchexpand3 = PatchExpand(input_resolution=None, dim=feat_size[3], dim_scale=2)
        self.patchexpand2 = PatchExpand(input_resolution=None, dim=feat_size[2], dim_scale=2)
        self.finalpatchexpand = FinalPatchExpand_X4(input_resolution=None, dim=384)
        # self.upblock4 = UnetrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=self.feat_size[4],
        #     out_channels=self.feat_size[1],
        #     kernel_size=3,
        #     upsample_kernel_size=6,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )
        # self.upblock3 = UnetrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=self.feat_size[3],
        #     out_channels=self.feat_size[1],
        #     kernel_size=3,
        #     upsample_kernel_size=4,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )
        # self.upblock2 = UnetrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=self.feat_size[2],
        #     out_channels=self.feat_size[1],
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )
        # self.block1 = UnetrBasicBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=self.feat_size[1],
        #     out_channels=self.feat_size[1],
        #     kernel_size=3,
        #     stride=1,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )
        # self.upblock4 = SubpixelUpsample(
        #     scale_factor=8,
        #     in_channels=self.feat_size[4],
        #     out_channels=self.feat_size[4],
        #     spatial_dims=2
        # )
        # self.upblock3 = SubpixelUpsample(
        #     scale_factor=4,
        #     in_channels=self.feat_size[3],
        #     out_channels=self.feat_size[3],
        #     spatial_dims=2
        # )
        # self.upblock2 = SubpixelUpsample(
        #     scale_factor=2,
        #     in_channels=self.feat_size[2],
        #     out_channels=self.feat_size[2],
        #     spatial_dims=2
        # )
        #
        # self.upblock1 = SubpixelUpsample(
        #     scale_factor=4,
        #     in_channels=self.feat_size[1] + self.feat_size[4],
        #     out_channels=self.out_chans,
        #     spatial_dims=2
        # )

    def forward(self, x):
        # x1 = self.stem(x)
        x = self.patchembed(x)
        for vssblock1 in self.vssBlock1:
            x = vssblock1(x)
        enc2 = x.permute(0, 3, 1, 2).contiguous()
        x = self.patchmerge[0](x)
        for vssblock2 in self.vssBlock2:
            x = vssblock2(x)
        enc3 = x.permute(0, 3, 1, 2).contiguous()
        # enc3 = self.upblock2(enc3)
        enc3 = self.patchexpand2(enc3)
        enc3 = enc3.permute(0, 3, 1, 2).contiguous()
        x = self.patchmerge[1](x)
        for vssblock3 in self.vssBlock3:
            x = vssblock3(x)
        enc4 = x.permute(0, 3, 1, 2).contiguous()
        # enc4 = self.upblock3(enc4)
        enc4 = self.patchexpand3(enc4)
        enc4 = enc4.permute(0, 3, 1, 2).contiguous()
        enc4 = self.patchexpand2(enc4)
        enc4 = enc4.permute(0, 3, 1, 2).contiguous()
        x = self.patchmerge[2](x)
        for vssblock4 in self.vssBlock4:
            x = vssblock4(x)
        enc5 = x.permute(0, 3, 1, 2).contiguous()
        enc5 = self.patchexpand4(enc5)
        enc5 = enc5.permute(0, 3, 1, 2).contiguous()
        enc5 = self.patchexpand3(enc5)
        enc5 = enc5.permute(0, 3, 1, 2).contiguous()
        enc5 = self.patchexpand2(enc5)
        enc5 = enc5.permute(0, 3, 1, 2).contiguous()
        # enc5 = self.upblock4(enc5)
        enc_total = torch.cat([enc3, enc4, enc5], dim=1)
        # enc_total = self.conv1(enc_total)
        enc_total = self.brfb(enc_total)
        enc_total = torch.cat([enc_total, enc2], dim=1)
        # out = self.upblock1(enc_total)
        out = self.finalpatchexpand(enc_total)
        out = out.permute(0, 3, 1, 2).contiguous()
        out = self.conv1(out)

        return out


def get_DeeplambaNoSe_from_plans(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        deep_supervision: bool = False,
):
    label_manager = plans_manager.get_label_manager(dataset_json)

    model = DeeplambaNoSe(
        in_chans=num_input_channels,
        out_chans=label_manager.num_segmentation_heads,
        feat_size=[48, 96, 192, 384, 768],
        depths=[2, 2, 9, 2],
        hidden_size=768,
    )

    return model


def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    assert not with_complex
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops

def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try:
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)

def selective_scan_flop_jit(inputs, outputs, flops_fn=flops_selective_scan_fn, verbose=True):
    if verbose:
        print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops


if __name__ == '__main__':
    # model = VSSBlock(hidden_dim=96,drop_path=0,norm_layer=nn.LayerNorm,attn_drop_rate=0.,d_state=16,)
    # model = PatchEmbed2D()    # 4, 3, 224, 224 -----> 4, 56, 56, 96 follow by patchsize = 4
    # model = CBAMLayer(1024)
    # model = PatchExpand(input_resolution=None, dim=96, dim_scale=4) # 4, 96, 224, 224 ----> 4, 448, 448, 48
    # model = PatchMerging2D(dim=96) # 4,224,224,96 ----> 4,112,112,192
    # model = FinalPatchExpand_X4(input_resolution=None, dim=3) # 4, 3, 512, 512 ----> 4, 2048, 2048, 3
    model = DeeplambaNoSe(in_chans=3, out_chans=13, feat_size=[48, 96, 192, 384, 768], depths=[2, 2, 9, 2],
                      drop_path_rate=0.2)
    # model = SemanticAttention(dim=96, n_cls=13) # 4, 512 ,96 ----> 4, 512, 13
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params}')
    x = torch.rand(1, 3, 320, 320)
    if torch.cuda.is_available():
        x = x.cuda()
        model = model.cuda()
    print(model(x).shape)

    supported_ops = {
        "prim::PythonOp.SelectiveScanFn":  selective_scan_flop_jit,
    }

    Gflops, unsupported = flop_count(model=model, inputs=(x,), supported_ops=supported_ops)
    print(Gflops)


    # flops, params = profile(model, inputs=(x,))
    # flop_counter = FlopCountAnalysis(model, x)
    # print(f"FLOPs: {flop_counter.total()}")
    # print(f"FLOPs: {flops}")
    # print(f"Params: {params}")


# #    summary(model, (3, 512, 512))
#     state_dict = model.state_dict()
#     print(state_dict.keys())
