import math

import torch
import torch.nn as nn
import einops
import torch.nn.functional as F

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


class Upsample(nn.Module):

    def __init__(
        self, 
        upscale_type: str, 
        in_channels: int, 
        out_channels: int, 
        up_factor: int, 
        activation: nn.Module, 
        bias: bool = False
    ) -> None:
        """
        Build an upscaling block.
        Args:
            upscale_type (str): the type of upscaling to use
            in_channels (int): the input channel dimension
            out_channels (int): the output channel dimension
            up_factor (int): the upscaling factor
            activation (nn.Module): the type of activation to use
            bias (bool): whether to use bias in the convolution
        Returns:
            nn.Module: the upscaling block
        """
        super(Upsample, self).__init__()
        if upscale_type == 'nearest':
            upsample_block = nn.Sequential(
                nn.Upsample(scale_factor=up_factor, mode='nearest'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=bias),
                LayerNorm(out_channels, data_format="channels_first"),
                activation(),
            )
        elif upscale_type == 'bilinear':
            upsample_block = nn.Sequential(
                nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=bias),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=bias),
                LayerNorm(out_channels, data_format="channels_first"),
                activation(),
            )
        elif upscale_type == 'conv':
            upsample_block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=up_factor, stride=up_factor),
                LayerNorm(out_channels, data_format="channels_first"),
                activation(),
            )
        elif upscale_type == 'pixelshuffle':
            conv = nn.Conv2d(in_channels, out_channels * up_factor ** 2, kernel_size=1, bias=False)
            upsample_block = nn.Sequential(
                conv,
                LayerNorm(out_channels * up_factor ** 2, data_format="channels_first"),
                activation(),
                nn.PixelShuffle(up_factor),
            )
            self.init_shuffle_conv_(conv, up_factor)
        else:
            raise ValueError(f"Invalid upscaling type: {upscale_type}")
        
        self.upsample_block = upsample_block

    def init_shuffle_conv_(self, conv, up_factor):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // (up_factor ** 2), i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = einops.repeat(conv_weight, f'o ... -> (o {up_factor ** 2}) ...')

        conv.weight.data.copy_(conv_weight)
        if conv.bias is not None:
            nn.init.zeros_(conv.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample_block(x)
    

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: nn.Module = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ChanRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.scale * self.gamma

def dwt_init3d(x):

    x01 = x[:, :, :, 0::2, :] / 2
    x02 = x[:, :, :, 1::2, :] / 2
    x1 = x01[:, :, :, :, 0::2]
    x2 = x02[:, :, :, :, 0::2]
    x3 = x01[:, :, :, :, 1::2]
    x4 = x02[:, :, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2


    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class DWT3d(nn.Module):
    def __init__(self):
        super(DWT3d, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init3d(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)