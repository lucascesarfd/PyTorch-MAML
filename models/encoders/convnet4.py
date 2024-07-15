import math
from collections import OrderedDict

import torch.nn as nn

from .encoders import register
from ..modules import *


__all__ = ["convnet4", "wide_convnet4"]


class ConvBlock(Module):
    def __init__(self, in_channels, out_channels, bn_args):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.bn = BatchNorm2d(out_channels, **bn_args)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, params=None, episode=None):
        out = self.conv(x, get_child_dict(params, "conv"))
        out = self.bn(out, get_child_dict(params, "bn"), episode)
        out = self.pool(self.relu(out))
        return out


class ConvNet4(Module):
    def __init__(self, channels, hid_dim, out_dim, bn_args, in_dim):
        super(ConvNet4, self).__init__()
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.img_dim = in_dim

        episodic = bn_args.get("episodic") or []
        bn_args_ep, bn_args_no_ep = bn_args.copy(), bn_args.copy()
        bn_args_ep["episodic"] = True
        bn_args_no_ep["episodic"] = False
        bn_args_dict = dict()
        for i in [1, 2, 3, 4]:
            if "conv%d" % i in episodic:
                bn_args_dict[i] = bn_args_ep
            else:
                bn_args_dict[i] = bn_args_no_ep

        self.encoder = Sequential(
            OrderedDict(
                [
                    ("conv1", ConvBlock(channels, hid_dim, bn_args_dict[1])),
                    ("conv2", ConvBlock(hid_dim, hid_dim, bn_args_dict[2])),
                    ("conv3", ConvBlock(hid_dim, hid_dim, bn_args_dict[3])),
                    ("conv4", ConvBlock(hid_dim, out_dim, bn_args_dict[4])),
                ]
            )
        )

    def get_out_dim(self):
        return self.out_dim * math.floor(self.img_dim[0] / 2**4) * math.floor(self.img_dim[1] / 2**4)

    def forward(self, x, params=None, episode=None):
        out = self.encoder(x, get_child_dict(params, "encoder"), episode)
        out = out.view(out.shape[0], -1)
        return out


@register("convnet4")
def convnet4(bn_args=dict(), in_dims=[84, 84], channels=3):
    return ConvNet4(channels, 32, 32, bn_args, in_dims)


@register("wide-convnet4")
def wide_convnet4(bn_args=dict(), in_dims=[84, 84], channels=3):
    return ConvNet4(channels, 64, 64, bn_args, in_dims)
