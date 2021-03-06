# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 16:12:16 2022

@author: Abdul Qayyum
"""

from traitlets.traitlets import Bool
from typing import Dict

import SimpleITK
import tqdm
import json
from pathlib import Path
import tifffile
import numpy as np
import cv2
#import matplotlib.pyplot as plt
import torch


########### define model object for prediction ########
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        out=x
        #print(x1.shape)
        x = self.head(x)
        return x,out

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
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


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@register_model
def convnext_tiny(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_small(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model
import sys 
import os
import glob
import time
import random
import os
import glob
import time
import random
import torch
import numpy as np
import pandas as pd
import torch.nn as nn


class CFG:
    img_size = 256
    n_frames = 10
    
    cnn_features = 256
    lstm_hidden = 32
    
    n_fold = 5
    n_epochs = 15
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        #self.map = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.net =convnext_base(pretrained=False)
        #checkpoint = torch.load("../input/efficientnet-pytorch/efficientnet-b0-08094119.pth")
        #self.net.load_state_dict(checkpoint)
        #out = out.view(-1, self.in_planes)
        self.net.head = nn.Linear(in_features=1024, out_features=2, bias=True)
    
    def forward(self, x):
        #x = F.relu(self.map(x))
        out,out1 = self.net(x)
        return out,out1
    
model=CNN()

path1='C:\\Users\\Administrateur\\Desktop\\airgochallenegs2022\\folder_docker\\latest_docker_file\\airogs-example-algorithm-master\\2dmodelairgos.pth'


# import timm
# class efficient(nn.Module):
#     def __init__(self, n_classes, pretrained=False):

#         super(efficient, self).__init__()

#         self.model = timm.create_model("efficientnet_b4", pretrained=pretrained)
#         #if pretrained:
#         #   self.model = timm.create_model(CFG['model_arch'], pretrained=True)
#             #self.model.load_state_dict(torch.load(MODEL_PATH))
#         self.model.classifier = nn.Linear(1792, n_classes)
#         # self.model.head = nn.Sequential(nn.Linear(self.model.head.in_features, 512),
#         #                                 nn.Dropout(0.5),
#         #                                 nn.ReLU(True),
#         #                                 nn.Linear(512,n_classes),
#         #                                 )
#         #self.model.classifier = nn.Linear(self.model.classifier.in_features, n_classes)

#     def forward(self, x):
#         x = self.model(x)
#         return x
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #m
# model = efficient(n_classes=2, pretrained=False).to(device)



################### variational autoencoder model
import numpy as np
import torch
import torch.distributions as dist

#from example_algos.models.nets import BasicEncoder, BasicGenerator

import warnings

import numpy as np
import torch
import torch.nn as nn


class NoOp(nn.Module):
    def __init__(self, *args, **kwargs):
        """NoOp Pytorch Module.
        Forwards the given input as is.
        """
        super(NoOp, self).__init__()

    def forward(self, x, *args, **kwargs):
        return x


class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_op=nn.Conv2d,
        conv_params=None,
        normalization_op=None,
        normalization_params=None,
        activation_op=nn.LeakyReLU,
        activation_params=None,
    ):
        """Basic Conv Pytorch Conv Module
        Has can have a Conv Op, a Normlization Op and a Non Linearity:
        x = conv(x)
        x = some_norm(x)
        x = nonlin(x)

        Args:
            in_channels ([int]): [Number on input channels/ feature maps]
            out_channels ([int]): [Number of ouput channels/ feature maps]
            conv_op ([torch.nn.Module], optional): [Conv operation]. Defaults to nn.Conv2d.
            conv_params ([dict], optional): [Init parameters for the conv operation]. Defaults to None.
            normalization_op ([torch.nn.Module], optional): [Normalization Operation (e.g. BatchNorm, InstanceNorm,...)]. Defaults to None.
            normalization_params ([dict], optional): [Init parameters for the normalization operation]. Defaults to None.
            activation_op ([torch.nn.Module], optional): [Actiovation Operation/ Non-linearity (e.g. ReLU, Sigmoid,...)]. Defaults to nn.LeakyReLU.
            activation_params ([dict], optional): [Init parameters for the activation operation]. Defaults to None.
        """

        super(ConvModule, self).__init__()

        self.conv_params = conv_params
        if self.conv_params is None:
            self.conv_params = {}
        self.activation_params = activation_params
        if self.activation_params is None:
            self.activation_params = {}
        self.normalization_params = normalization_params
        if self.normalization_params is None:
            self.normalization_params = {}

        self.conv = None
        if conv_op is not None and not isinstance(conv_op, str):
            self.conv = conv_op(in_channels, out_channels, **self.conv_params)

        self.normalization = None
        if normalization_op is not None and not isinstance(normalization_op, str):
            self.normalization = normalization_op(out_channels, **self.normalization_params)

        self.activation = None
        if activation_op is not None and not isinstance(activation_op, str):
            self.activation = activation_op(**self.activation_params)

    def forward(self, input, conv_add_input=None, normalization_add_input=None, activation_add_input=None):

        x = input

        if self.conv is not None:
            if conv_add_input is None:
                x = self.conv(x)
            else:
                x = self.conv(x, **conv_add_input)

        if self.normalization is not None:
            if normalization_add_input is None:
                x = self.normalization(x)
            else:
                x = self.normalization(x, **normalization_add_input)

        if self.activation is not None:
            if activation_add_input is None:
                x = self.activation(x)
            else:
                x = self.activation(x, **activation_add_input)

        # nn.functional.dropout(x, p=0.95, training=True)

        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        n_convs: int,
        n_featmaps: int,
        conv_op=nn.Conv2d,
        conv_params=None,
        normalization_op=nn.BatchNorm2d,
        normalization_params=None,
        activation_op=nn.LeakyReLU,
        activation_params=None,
    ):
        """Basic Conv block with repeated conv, build up from repeated @ConvModules (with same/fixed feature map size)

        Args:
            n_convs ([type]): [Number of convolutions]
            n_featmaps ([type]): [Feature map size of the conv]
            conv_op ([torch.nn.Module], optional): [Convulioton operation -> see ConvModule ]. Defaults to nn.Conv2d.
            conv_params ([dict], optional): [Init parameters for the conv operation]. Defaults to None.
            normalization_op ([torch.nn.Module], optional): [Normalization Operation (e.g. BatchNorm, InstanceNorm,...) -> see ConvModule]. Defaults to nn.BatchNorm2d.
            normalization_params ([dict], optional): [Init parameters for the normalization operation]. Defaults to None.
            activation_op ([torch.nn.Module], optional): [Actiovation Operation/ Non-linearity (e.g. ReLU, Sigmoid,...) -> see ConvModule]. Defaults to nn.LeakyReLU.
            activation_params ([dict], optional): [Init parameters for the activation operation]. Defaults to None.
        """

        super(ConvBlock, self).__init__()

        self.n_featmaps = n_featmaps
        self.n_convs = n_convs
        self.conv_params = conv_params
        if self.conv_params is None:
            self.conv_params = {}

        self.conv_list = nn.ModuleList()

        for i in range(self.n_convs):
            conv_layer = ConvModule(
                n_featmaps,
                n_featmaps,
                conv_op=conv_op,
                conv_params=conv_params,
                normalization_op=normalization_op,
                normalization_params=normalization_params,
                activation_op=activation_op,
                activation_params=activation_params,
            )
            self.conv_list.append(conv_layer)

    def forward(self, input, **frwd_params):
        x = input
        for conv_layer in self.conv_list:
            x = conv_layer(x)

        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        n_convs,
        n_featmaps,
        conv_op=nn.Conv2d,
        conv_params=None,
        normalization_op=nn.BatchNorm2d,
        normalization_params=None,
        activation_op=nn.LeakyReLU,
        activation_params=None,
    ):
        """Basic Conv block with repeated conv, build up from repeated @ConvModules (with same/fixed feature map size) and a skip/ residual connection:
        x = input
        x = conv_block(x)
        out = x + input

        Args:
            n_convs ([type]): [Number of convolutions in the conv block]
            n_featmaps ([type]): [Feature map size of the conv block]
            conv_op ([torch.nn.Module], optional): [Convulioton operation -> see ConvModule ]. Defaults to nn.Conv2d.
            conv_params ([dict], optional): [Init parameters for the conv operation]. Defaults to None.
            normalization_op ([torch.nn.Module], optional): [Normalization Operation (e.g. BatchNorm, InstanceNorm,...) -> see ConvModule]. Defaults to nn.BatchNorm2d.
            normalization_params ([dict], optional): [Init parameters for the normalization operation]. Defaults to None.
            activation_op ([torch.nn.Module], optional): [Actiovation Operation/ Non-linearity (e.g. ReLU, Sigmoid,...) -> see ConvModule]. Defaults to nn.LeakyReLU.
            activation_params ([dict], optional): [Init parameters for the activation operation]. Defaults to None.
        """
        super(ResBlock, self).__init__()

        self.n_featmaps = n_featmaps
        self.n_convs = n_convs
        self.conv_params = conv_params
        if self.conv_params is None:
            self.conv_params = {}

        self.conv_block = ConvBlock(
            n_featmaps,
            n_convs,
            conv_op=conv_op,
            conv_params=conv_params,
            normalization_op=normalization_op,
            normalization_params=normalization_params,
            activation_op=activation_op,
            activation_params=activation_params,
        )

    def forward(self, input, **frwd_params):
        x = input
        x = self.conv_block(x)

        out = x + input

        return out


# Basic Generator
class BasicGenerator(nn.Module):
    def __init__(
        self,
        input_size,
        z_dim=256,
        fmap_sizes=(256, 128, 64),
        upsample_op=nn.ConvTranspose2d,
        conv_params=None,
        normalization_op=NoOp,
        normalization_params=None,
        activation_op=nn.LeakyReLU,
        activation_params=None,
        block_op=NoOp,
        block_params=None,
        to_1x1=True,
    ):
        """Basic configureable Generator/ Decoder.
        Allows for mutilple "feature-map" levels defined by the feature map size, where for each feature map size a conv operation + optional conv block is used.

        Args:
            input_size ((int, int, int): Size of the input in format CxHxW): 
            z_dim (int, optional): [description]. Dimension of the latent / Input dimension (C channel-dim).
            fmap_sizes (tuple, optional): [Defines the Upsampling-Levels of the generator, list/ tuple of ints, where each 
                                            int defines the number of feature maps in the layer]. Defaults to (256, 128, 64).
            upsample_op ([torch.nn.Module], optional): [Upsampling operation used, to upsample to a new level/ featuremap size]. Defaults to nn.ConvTranspose2d.
            conv_params ([dict], optional): [Init parameters for the conv operation]. Defaults to dict(kernel_size=3, stride=2, padding=1, bias=False).
            normalization_op ([torch.nn.Module], optional): [Normalization Operation (e.g. BatchNorm, InstanceNorm,...) -> see ConvModule]. Defaults to nn.BatchNorm2d.
            normalization_params ([dict], optional): [Init parameters for the normalization operation]. Defaults to None.
            activation_op ([torch.nn.Module], optional): [Actiovation Operation/ Non-linearity (e.g. ReLU, Sigmoid,...) -> see ConvModule]. Defaults to nn.LeakyReLU.
            activation_params ([dict], optional): [Init parameters for the activation operation]. Defaults to None.
            block_op ([torch.nn.Module], optional): [Block operation used for each feature map size after each upsample op of e.g. ConvBlock/ ResidualBlock]. Defaults to NoOp.
            block_params ([dict], optional): [Init parameters for the block operation]. Defaults to None.
            to_1x1 (bool, optional): [If Latent dimesion is a z_dim x 1 x 1 vector (True) or if allows spatial resolution not to be 1x1 (z_dim x H x W) (False) ]. Defaults to True.
        """

        super(BasicGenerator, self).__init__()

        if conv_params is None:
            conv_params = dict(kernel_size=4, stride=2, padding=1, bias=False)
        if block_op is None:
            block_op = NoOp
        if block_params is None:
            block_params = {}

        n_channels = input_size[0]
        input_size_ = np.array(input_size[1:])

        if not isinstance(fmap_sizes, list) and not isinstance(fmap_sizes, tuple):
            raise AttributeError("fmap_sizes has to be either a list or tuple or an int")
        elif len(fmap_sizes) < 2:
            raise AttributeError("fmap_sizes has to contain at least three elements")
        else:
            h_size_bot = fmap_sizes[0]

        # We need to know how many layers we will use at the beginning
        input_size_new = input_size_ // (2 ** len(fmap_sizes))
        if np.min(input_size_new) < 2 and z_dim is not None:
            raise AttributeError("fmap_sizes to long, one image dimension has already perished")

        ### Start block
        start_block = []

        if not to_1x1:
            kernel_size_start = [min(conv_params["kernel_size"], i) for i in input_size_new]
        else:
            kernel_size_start = input_size_new.tolist()

        if z_dim is not None:
            self.start = ConvModule(
                z_dim,
                h_size_bot,
                conv_op=upsample_op,
                conv_params=dict(kernel_size=kernel_size_start, stride=1, padding=0, bias=False),
                normalization_op=normalization_op,
                normalization_params=normalization_params,
                activation_op=activation_op,
                activation_params=activation_params,
            )

            input_size_new = input_size_new * 2
        else:
            self.start = NoOp()

        ### Middle block (Done until we reach ? x input_size/2 x input_size/2)
        self.middle_blocks = nn.ModuleList()

        for h_size_top in fmap_sizes[1:]:

            self.middle_blocks.append(block_op(h_size_bot, **block_params))

            self.middle_blocks.append(
                ConvModule(
                    h_size_bot,
                    h_size_top,
                    conv_op=upsample_op,
                    conv_params=conv_params,
                    normalization_op=normalization_op,
                    normalization_params={},
                    activation_op=activation_op,
                    activation_params=activation_params,
                )
            )

            h_size_bot = h_size_top
            input_size_new = input_size_new * 2

        ### End block
        self.end = ConvModule(
            h_size_bot,
            n_channels,
            conv_op=upsample_op,
            conv_params=conv_params,
            normalization_op=None,
            activation_op=None,
        )

    def forward(self, inpt, **kwargs):
        output = self.start(inpt, **kwargs)
        for middle in self.middle_blocks:
            output = middle(output, **kwargs)
        output = self.end(output, **kwargs)
        return output


# Basic Encoder
class BasicEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        z_dim=256,
        fmap_sizes=(64, 128, 256),
        conv_op=nn.Conv2d,
        conv_params=None,
        normalization_op=NoOp,
        normalization_params=None,
        activation_op=nn.LeakyReLU,
        activation_params=None,
        block_op=NoOp,
        block_params=None,
        to_1x1=True,
    ):
        """Basic configureable Encoder.
        Allows for mutilple "feature-map" levels defined by the feature map size, where for each feature map size a conv operation + optional conv block is used. 

        Args:
            z_dim (int, optional): [description]. Dimension of the latent / Input dimension (C channel-dim).
            fmap_sizes (tuple, optional): [Defines the Upsampling-Levels of the generator, list/ tuple of ints, where each 
                                            int defines the number of feature maps in the layer]. Defaults to (64, 128, 256).
            conv_op ([torch.nn.Module], optional): [Convolutioon operation used to downsample to a new level/ featuremap size]. Defaults to nn.Conv2d.
            conv_params ([dict], optional): [Init parameters for the conv operation]. Defaults to dict(kernel_size=3, stride=2, padding=1, bias=False).
            normalization_op ([torch.nn.Module], optional): [Normalization Operation (e.g. BatchNorm, InstanceNorm,...) -> see ConvModule]. Defaults to nn.BatchNorm2d.
            normalization_params ([dict], optional): [Init parameters for the normalization operation]. Defaults to None.
            activation_op ([torch.nn.Module], optional): [Actiovation Operation/ Non-linearity (e.g. ReLU, Sigmoid,...) -> see ConvModule]. Defaults to nn.LeakyReLU.
            activation_params ([dict], optional): [Init parameters for the activation operation]. Defaults to None.
            block_op ([torch.nn.Module], optional): [Block operation used for each feature map size after each upsample op of e.g. ConvBlock/ ResidualBlock]. Defaults to NoOp.
            block_params ([dict], optional): [Init parameters for the block operation]. Defaults to None.
            to_1x1 (bool, optional): [If True, then the last conv layer goes to a latent dimesion is a z_dim x 1 x 1 vector (similar to fully connected) or if False allows spatial resolution not to be 1x1 (z_dim x H x W, uses the in the conv_params given conv-kernel-size) ]. Defaults to True.
        """
        super(BasicEncoder, self).__init__()

        if conv_params is None:
            conv_params = dict(kernel_size=3, stride=2, padding=1, bias=False)
        if block_op is None:
            block_op = NoOp
        if block_params is None:
            block_params = {}

        n_channels = input_size[0]
        input_size_new = np.array(input_size[1:])

        if not isinstance(fmap_sizes, list) and not isinstance(fmap_sizes, tuple):
            raise AttributeError("fmap_sizes has to be either a list or tuple or an int")
        # elif len(fmap_sizes) < 2:
        #     raise AttributeError("fmap_sizes has to contain at least three elements")
        else:
            h_size_bot = fmap_sizes[0]

        ### Start block
        self.start = ConvModule(
            n_channels,
            h_size_bot,
            conv_op=conv_op,
            conv_params=conv_params,
            normalization_op=normalization_op,
            normalization_params={},
            activation_op=activation_op,
            activation_params=activation_params,
        )
        input_size_new = input_size_new // 2

        ### Middle block (Done until we reach ? x 4 x 4)
        self.middle_blocks = nn.ModuleList()

        for h_size_top in fmap_sizes[1:]:

            self.middle_blocks.append(block_op(h_size_bot, **block_params))

            self.middle_blocks.append(
                ConvModule(
                    h_size_bot,
                    h_size_top,
                    conv_op=conv_op,
                    conv_params=conv_params,
                    normalization_op=normalization_op,
                    normalization_params={},
                    activation_op=activation_op,
                    activation_params=activation_params,
                )
            )

            h_size_bot = h_size_top
            input_size_new = input_size_new // 2

            if np.min(input_size_new) < 2 and z_dim is not None:
                raise ("fmap_sizes to long, one image dimension has already perished")

        ### End block
        if not to_1x1:
            kernel_size_end = [min(conv_params["kernel_size"], i) for i in input_size_new]
        else:
            kernel_size_end = input_size_new.tolist()

        if z_dim is not None:
            self.end = ConvModule(
                h_size_bot,
                z_dim,
                conv_op=conv_op,
                conv_params=dict(kernel_size=kernel_size_end, stride=1, padding=0, bias=False),
                normalization_op=None,
                activation_op=None,
            )

            if to_1x1:
                self.output_size = (z_dim, 1, 1)
            else:
                self.output_size = (z_dim, *[i - (j - 1) for i, j in zip(input_size_new, kernel_size_end)])
        else:
            self.end = NoOp()
            self.output_size = input_size_new

    def forward(self, inpt, **kwargs):
        output = self.start(inpt, **kwargs)
        for middle in self.middle_blocks:
            output = middle(output, **kwargs)
        output = self.end(output, **kwargs)
        return output


class VAE(torch.nn.Module):
    def __init__(
        self,
        input_size,
        z_dim=256,
        fmap_sizes=(16, 64, 256, 1024),
        to_1x1=True,
        conv_op=torch.nn.Conv2d,
        conv_params=None,
        tconv_op=torch.nn.ConvTranspose2d,
        tconv_params=None,
        normalization_op=None,
        normalization_params=None,
        activation_op=torch.nn.LeakyReLU,
        activation_params=None,
        block_op=None,
        block_params=None,
        *args,
        **kwargs
    ):
        """Basic VAE build up of a symetric BasicEncoder (Encoder) and BasicGenerator (Decoder)

        Args:
            input_size ((int, int, int): Size of the input in format CxHxW): 
            z_dim (int, optional): [description]. Dimension of the latent / Input dimension (C channel-dim). Defaults to 256
            fmap_sizes (tuple, optional): [Defines the Upsampling-Levels of the generator, list/ tuple of ints, where each 
                                            int defines the number of feature maps in the layer]. Defaults to (16, 64, 256, 1024).
            to_1x1 (bool, optional): [If True, then the last conv layer goes to a latent dimesion is a z_dim x 1 x 1 vector (similar to fully connected) 
                                        or if False allows spatial resolution not to be 1x1 (z_dim x H x W, uses the in the conv_params given conv-kernel-size) ].
                                        Defaults to True.
            conv_op ([torch.nn.Module], optional): [Convolutioon operation used in the encoder to downsample to a new level/ featuremap size]. Defaults to nn.Conv2d.
            conv_params ([dict], optional): [Init parameters for the conv operation]. Defaults to dict(kernel_size=3, stride=2, padding=1, bias=False).
            tconv_op ([torch.nn.Module], optional): [Upsampling/ Transposed Conv operation used in the decoder to upsample to a new level/ featuremap size]. Defaults to nn.ConvTranspose2d.
            tconv_params ([dict], optional): [Init parameters for the conv operation]. Defaults to dict(kernel_size=3, stride=2, padding=1, bias=False).
            normalization_op ([torch.nn.Module], optional): [Normalization Operation (e.g. BatchNorm, InstanceNorm,...) -> see ConvModule]. Defaults to nn.BatchNorm2d.
            normalization_params ([dict], optional): [Init parameters for the normalization operation]. Defaults to None.
            activation_op ([torch.nn.Module], optional): [Actiovation Operation/ Non-linearity (e.g. ReLU, Sigmoid,...) -> see ConvModule]. Defaults to nn.LeakyReLU.
            activation_params ([dict], optional): [Init parameters for the activation operation]. Defaults to None.
            block_op ([torch.nn.Module], optional): [Block operation used for each feature map size after each upsample op of e.g. ConvBlock/ ResidualBlock]. Defaults to NoOp.
            block_params ([dict], optional): [Init parameters for the block operation]. Defaults to None.
        """

        super(VAE, self).__init__()

        input_size_enc = list(input_size)
        input_size_dec = list(input_size)

        self.enc = BasicEncoder(
            input_size=input_size_enc,
            fmap_sizes=fmap_sizes,
            z_dim=z_dim * 2,
            conv_op=conv_op,
            conv_params=conv_params,
            normalization_op=normalization_op,
            normalization_params=normalization_params,
            activation_op=activation_op,
            activation_params=activation_params,
            block_op=block_op,
            block_params=block_params,
            to_1x1=to_1x1,
        )
        self.dec = BasicGenerator(
            input_size=input_size_dec,
            fmap_sizes=fmap_sizes[::-1],
            z_dim=z_dim,
            upsample_op=tconv_op,
            conv_params=tconv_params,
            normalization_op=normalization_op,
            normalization_params=normalization_params,
            activation_op=activation_op,
            activation_params=activation_params,
            block_op=block_op,
            block_params=block_params,
            to_1x1=to_1x1,
        )

        self.hidden_size = self.enc.output_size

    def forward(self, inpt, sample=True, no_dist=False, **kwargs):
        y1 = self.enc(inpt, **kwargs)

        mu, log_std = torch.chunk(y1, 2, dim=1)
        std = torch.exp(log_std)
        z_dist = dist.Normal(mu, std)
        if sample:
            z_sample = z_dist.rsample()
        else:
            z_sample = mu

        x_rec = self.dec(z_sample)

        if no_dist:
            return x_rec
        else:
            return x_rec, z_dist

    def encode(self, inpt, **kwargs):
        """Encodes a sample and returns the paramters for the approx inference dist. (Normal)

        Args:
            inpt ([tensor]): The input to encode

        Returns:
            mu : The mean used to parameterized a Normal distribution
            std: The standard deviation used to parameterized a Normal distribution
        """
        enc = self.enc(inpt, **kwargs)
        mu, log_std = torch.chunk(enc, 2, dim=1)
        std = torch.exp(log_std)
        return mu, std

    def decode(self, inpt, **kwargs):
        """Decodes a latent space sample, used the generative model (decode = mu_{gen}(z) as used in p(x|z) = N(x | mu_{gen}(z), 1) ).

        Args:
            inpt ([type]): A sample from the latent space to decode

        Returns:
            [type]: [description]
        """
        x_rec = self.dec(inpt, **kwargs)
        return x_rec


class AE(torch.nn.Module):
    def __init__(
        self,
        input_size,
        z_dim=1024,
        fmap_sizes=(16, 64, 256, 1024),
        to_1x1=True,
        conv_op=torch.nn.Conv2d,
        conv_params=None,
        tconv_op=torch.nn.ConvTranspose2d,
        tconv_params=None,
        normalization_op=None,
        normalization_params=None,
        activation_op=torch.nn.LeakyReLU,
        activation_params=None,
        block_op=None,
        block_params=None,
        *args,
        **kwargs
    ):
        """Basic AE build up of a symetric BasicEncoder (Encoder) and BasicGenerator (Decoder)

        Args:
            input_size ((int, int, int): Size of the input in format CxHxW): 
            z_dim (int, optional): [description]. Dimension of the latent / Input dimension (C channel-dim). Defaults to 256
            fmap_sizes (tuple, optional): [Defines the Upsampling-Levels of the generator, list/ tuple of ints, where each 
                                            int defines the number of feature maps in the layer]. Defaults to (16, 64, 256, 1024).
            to_1x1 (bool, optional): [If True, then the last conv layer goes to a latent dimesion is a z_dim x 1 x 1 vector (similar to fully connected) 
                                        or if False allows spatial resolution not to be 1x1 (z_dim x H x W, uses the in the conv_params given conv-kernel-size) ].
                                        Defaults to True.
            conv_op ([torch.nn.Module], optional): [Convolutioon operation used in the encoder to downsample to a new level/ featuremap size]. Defaults to nn.Conv2d.
            conv_params ([dict], optional): [Init parameters for the conv operation]. Defaults to dict(kernel_size=3, stride=2, padding=1, bias=False).
            tconv_op ([torch.nn.Module], optional): [Upsampling/ Transposed Conv operation used in the decoder to upsample to a new level/ featuremap size]. Defaults to nn.ConvTranspose2d.
            tconv_params ([dict], optional): [Init parameters for the conv operation]. Defaults to dict(kernel_size=3, stride=2, padding=1, bias=False).
            normalization_op ([torch.nn.Module], optional): [Normalization Operation (e.g. BatchNorm, InstanceNorm,...) -> see ConvModule]. Defaults to nn.BatchNorm2d.
            normalization_params ([dict], optional): [Init parameters for the normalization operation]. Defaults to None.
            activation_op ([torch.nn.Module], optional): [Actiovation Operation/ Non-linearity (e.g. ReLU, Sigmoid,...) -> see ConvModule]. Defaults to nn.LeakyReLU.
            activation_params ([dict], optional): [Init parameters for the activation operation]. Defaults to None.
            block_op ([torch.nn.Module], optional): [Block operation used for each feature map size after each upsample op of e.g. ConvBlock/ ResidualBlock]. Defaults to NoOp.
            block_params ([dict], optional): [Init parameters for the block operation]. Defaults to None.
        """
        super(AE, self).__init__()

        input_size_enc = list(input_size)
        input_size_dec = list(input_size)

        self.enc = BasicEncoder(
            input_size=input_size_enc,
            fmap_sizes=fmap_sizes,
            z_dim=z_dim,
            conv_op=conv_op,
            conv_params=conv_params,
            normalization_op=normalization_op,
            normalization_params=normalization_params,
            activation_op=activation_op,
            activation_params=activation_params,
            block_op=block_op,
            block_params=block_params,
            to_1x1=to_1x1,
        )
        self.dec = BasicGenerator(
            input_size=input_size_dec,
            fmap_sizes=fmap_sizes[::-1],
            z_dim=z_dim,
            upsample_op=tconv_op,
            conv_params=tconv_params,
            normalization_op=normalization_op,
            normalization_params=normalization_params,
            activation_op=activation_op,
            activation_params=activation_params,
            block_op=block_op,
            block_params=block_params,
            to_1x1=to_1x1,
        )

        self.hidden_size = self.enc.output_size

    def forward(self, inpt, **kwargs):

        y1 = self.enc(inpt, **kwargs)

        x_rec = self.dec(y1)

        return x_rec

    def encode(self, inpt, **kwargs):
        """Encodes a input sample to a latent space sample

        Args:
            inpt ([tensor]): Input sample

        Returns:
            enc: Encoded input sample in the latent space
        """
        enc = self.enc(inpt, **kwargs)
        return enc

    def decode(self, inpt, **kwargs):
        """Decodes a latent space sample back to the input space

        Args:
            inpt ([tensor]): [Latent space sample]

        Returns:
            [rec]: [Encoded latent sample back in the input space]
        """
        rec = self.dec(inpt, **kwargs)
        return rec

z_dim=512
model_feature_map_sizes=(16, 64, 256, 1024)
    
import torch
input_shape=((1, 1, 128, 128))
#input_shape(0).shape
#c,h,y=input_shape.size()[0],input_shape.size()[1],input_shape.size()[2]

model1 = AE(input_size=(3,128,128), z_dim=z_dim, fmap_sizes=model_feature_map_sizes)
patha='C:\\Users\\Administrateur\\Desktop\\airgochallenegs2022\\folder_docker\\latest_docker_file\\airogs-example-algorithm-master\\model_weights1.pth'
#trained=torch.load(patha,map_location=torch.device('cpu'))
#model1.load_state_dict(trained)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model=nn.DataParallel(model)
#model=model.to(device)
from evalutils import ClassificationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
from evalutils.io import ImageLoader


class DummyLoader(ImageLoader):
    @staticmethod
    def load_image(fname):
        return str(fname)


    @staticmethod
    def hash_image(image):
        return hash(image)


class airogs_algorithm(ClassificationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )

        self._file_loaders = dict(input_image=DummyLoader())

        self.output_keys = ["multiple-referable-glaucoma-likelihoods", 
                            "multiple-referable-glaucoma-binary",
                            "multiple-ungradability-scores",
                            "multiple-ungradability-binary"]
    
    def load(self):
        for key, file_loader in self._file_loaders.items():
            fltr = (
                self._file_filters[key] if key in self._file_filters else None
            )
            self._cases[key] = self._load_cases(
                folder=Path("C:\\Users\\Administrateur\\Desktop\\airgochallenegs2022\\folder_docker\\latest_docker_file\\airogs-example-algorithm-master\\test\\images/color-fundus/"),
                file_loader=file_loader,
                file_filter=fltr,
            )

        pass
    
    def combine_dicts(self, dicts):
        out = {}
        for d in dicts:
            for k, v in d.items():
                if k not in out:
                    out[k] = []
                out[k].append(v)
        return out
    
    def process_case(self, *, idx, case):
        # Load and test the image(s) for this case
        if case.path.suffix == '.tiff':
            results = []
            #predc=[]
            with tifffile.TiffFile(case.path) as stack:
                for page in tqdm.tqdm(stack.pages):
                    input_image_array = page.asarray()
                    
                    #print(input_image_array.shape)
                    input_image_array= cv2.resize(input_image_array, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
                    input_image_array=((input_image_array - input_image_array.min()) / (input_image_array.max() - input_image_array.min()))
                    #np_img=input_image_array
                    #torch_array=torch.from_numpy(np_img).unsqueeze(axis=0)
                    #torch_array=torch_array.permute(0, 3, 1, 2)
                    #print(torch_array.shape)
                    #plt.imshow(input_image_array)
                    #print(input_image_array.min())
                    #print(input_image_array.max())
                    #p1,p2=self.predict(input_image_array=input_image_array)
                    results.append(self.predict(input_image_array=input_image_array))
                    #predc.append(p2)
        else:
            input_image = SimpleITK.ReadImage(str(case.path))
            input_image_array = SimpleITK.GetArrayFromImage(input_image)
            #print(input_image_array.shape)
            results = [self.predict(input_image_array=input_image_array)]
        
        results = self.combine_dicts(results)

        # Test classification output
        if not isinstance(results, dict):
            raise ValueError("Expected a dictionary as output")

        return results
    
    #pc11=[]

    def predict(self, *, input_image_array: np.ndarray) -> Dict:
        # From here, use the input_image to predict the output
        # We are using a not-so-smart algorithm to predict the output, you'll want to do your model inference here

        # Replace starting here
        
        #rg_likelihood = ((input_image_array - input_image_array.min()) / (input_image_array.max() - input_image_array.min())).mean()
        #rg_binary = bool(rg_likelihood > .2)
        #ungradability_score = rg_likelihood * 15
        #ungradability_binary = bool(rg_likelihood < .2)
        
        np_img=input_image_array
        torch_array=torch.from_numpy(np_img).unsqueeze(axis=0)
        torch_array=torch_array.permute(0, 3, 1, 2)
        torch_array=torch_array.float().to(device)
        ####### load pytorch model for prediction
        model=CNN()
        #model= efficient(n_classes=2, pretrained=False)
        model=nn.DataParallel(model)
        model=model.to(device)
        pretrained=torch.load(path1,map_location=torch.device('cpu'))
        #model=nn.DataParallel(model)
        #model=model.to(device)
        model.load_state_dict(pretrained)
        model1 = AE(input_size=(3,128,128), z_dim=z_dim, fmap_sizes=model_feature_map_sizes)
        #patha='/content/drive/MyDrive/Irgos_challenege2022/model_weights.pth'
        model1=nn.DataParallel(model1)
        trained=torch.load(patha,map_location=torch.device('cpu'))
        model1.load_state_dict(trained)
        with torch.no_grad():
          y_pred,_=model(torch_array)
          pc1= torch.softmax(y_pred, dim=1)
          pc1=pc1.squeeze(axis=0)
          predic0,predic1=pc1.detach().cpu().numpy()
          #p2=predic1
          #predict11.append(p2)
          #predic00.append(predic0)
          input_image_array1= cv2.resize(input_image_array, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
          input_image_array1=((input_image_array1 - input_image_array1.min()) / (input_image_array1.max() - input_image_array1.min()))
          #print(input_image_array1.shape)
          np_img1=input_image_array1
          torch_array1=torch.from_numpy(np_img1).unsqueeze(axis=0)
          torch_array1=torch_array1.permute(0, 3, 1, 2)
          torch_array1=torch_array1.float().to(device)
          model1=model1.to(device)
          #torch_array1=torch_array1.to(device)
          batch_rec = model1(torch_array1)
          loss = torch.mean(torch.pow(torch_array1 - batch_rec, 2), dim=(1, 2, 3))
          #slice_scores += loss.cpu().tolist()
          #batch_rec1=batch_rec.squeeze(axis=0).detach().cpu().numpy()
          ###### output probabilties computed here
          # # Compute the training data empirical correlation matrix inverse
          # x_t_x_inv = torch.linalg.inv(features.T @ features)
          # #print(x_t_x_inv.shape)
          # #probs = torch.softmax(model.classifier(features), dim=-1)
          # probs=pc1

          # # Normalize
          # norm = torch.linalg.norm(features, dim=-1, keepdim=True)
          # features = features / norm

          # # Calc projection
          # x_proj = features @ x_t_x_inv @ features.T
          # xt_g = x_proj / (1 + x_proj)

          # # Equation 20
          # n_classes = 2
          # nf = torch.sum(probs / (probs + (1 - probs) * (probs ** xt_g)), dim=-1)
          # regrets = torch.log(nf) / torch.log(torch.tensor(n_classes))
          # print(regrets.numpy())
          
          
          rg_likelihood=float(predic1)
          rg_binary=bool(rg_likelihood>0.5)
            #rg_likelihood=str(rg_likelihood1)
            #rg_binary=str(rg_binary1)
          un_gradable=predic0
          predb0=(un_gradable>0.5)
          #predb0=(rg_likelihood<0.5)
          w=loss.cpu().numpy()
          #w=(w>0.2)
          
          #ungradability_score=np.array(regrets)
          #ungradability_score=2*np.array(w or predb0)
            #ungradability_score=str(ungradability_score1)
          #ungradability_score=float(np.squeeze(ungradability_score))
          #ungradability_binary=bool(predb0<0.5 and rg_likelihood<0.5)
            #ungradability_score=str(ungradability_binary1)
          #s=[]
          if predic1<0.5 and predic1>=0:
              s=2*predic1
            
          else:
             s=-2*(predic1-1)
          ungradability_score=float(rg_likelihood*2)
          ungradability_binary=bool(rg_likelihood<0.5)
        
          
          # import torch.nn.functional as F  
          # to_np = lambda x: x.data.cpu().numpy()
          # # #p=F.softmax(output, dim=1)
          # output=y_pred
          # p1=to_np(F.softmax(output, dim=1))                                  

          # energy=-to_np((1*torch.logsumexp(output /1, dim=1)))
          # threshold=np.percentile(np_img1, 95)
          # print(energy)
          # print(threshold)
          # if energy>=threshold:
          #     rg_likelihood=float(predic1)
          #     rg_binary=bool(rg_likelihood>0.5)
          # elif energy<threshold:
          #     w=loss.cpu().numpy()
          #     w=(w>0.2)
          #     predb0=(rg_likelihood<0.5)
          #     ungradability_score=np.array(w or predb0)
          #     ungradability_binary=bool(ungradability_score)
          # else:
          #     print('wrong score')
           #ungradability_binary=bool(ungradability_score>0)
           #rg_binary=bool(rg_likelihood>0.5)

          #print(sucess)
          # print(rg_binary)
          # print(ungradability_score)
          # print(ungradability_binary)
          # print(ungradability_score)
          # if energy<threshold:
          #     ungradability_score=(rg_likelihood<0.5)
          #     #ungradability_score=np.array(loss.cpu().numpy()*predb0)
          #     #ungradability_score=str(ungradability_score1)
          #     #ungradability_score=float(np.squeeze(ungradability_score))
          #     ungradability_binary=bool(ungradability_score>0)
          # else:
          #     ungradability_score=(rg_likelihood<0.5)
          #     #ungradability_score=np.array(loss.cpu().numpy()*predb0)
          #     #ungradability_score=str(ungradability_score1)
          #     #ungradability_score=float(np.squeeze(ungradability_score))
          #     ungradability_binary=bool(ungradability_score)
             

          #likelihood1.append(likelihood)
          #rg_binary1.append(rg_binary)
          #ungradability_score1.append(ungradability_score)
          #ungradability_binary1.append(ungradability_binary)
        
        
        # to here with your inference algorithm

        out = {
            "multiple-referable-glaucoma-likelihoods": rg_likelihood,
            "multiple-referable-glaucoma-binary": rg_binary,
            "multiple-ungradability-scores": ungradability_score,
            "multiple-ungradability-binary": ungradability_binary
        }

        return out

    def save(self):
        for key in self.output_keys:
            with open(f"C:\\Users\\Administrateur\\Desktop\\airgochallenegs2022\\folder_docker\\latest_docker_file\\airogs-example-algorithm-master\\new_output/{key}.json", "w") as f:
                out = []
                for case_result in self._case_results:
                    out += case_result[key]
                json.dump(out, f)


if __name__ == "__main__":
    airogs_algorithm().process()