# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 14:20:14 2022

@author: Abdul Qayyum
"""
#%% 2d autoencoder
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7" 
import cv2
import matplotlib.pyplot as plt
import skimage 
from torchvision.io import read_image
import torchvision.transforms
import albumentations as A
import albumentations.augmentations.functional as F
from torchvision import transforms
from PIL import Image
trans=transforms.Compose([transforms.ToTensor(), ])


trans2=transforms.Compose([
        #transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),])

class eye_img(Dataset):
  def __init__(self, images_dir, labels_dir, transform= None):
    self.images_dir=images_dir
    self.labels_dir=labels_dir
    self.transform= transform

    # Image Links
    self.lis_images=os.listdir(self.images_dir)
    #read_label=self.labels_dir
    self.Image_link=[]
    for i in self.labels_dir['challenge_id']:
      pathfull=os.path.join(self.images_dir,i+'.jpg')
      self.Image_link.append(pathfull)
    
    # Label Links
    #read_label=self.labels_dir
    
    self.Label_link=self.labels_dir['label']
    
  def __getitem__(self,idx):
    im=self.Image_link[idx]
    lb=self.Label_link[idx]
    lb=torch.from_numpy(np.asarray(lb))

    read_im=Image.open(im)
    #resize_im=cv2.resize(read_im, (224,224))
    img = read_im.resize((128,128), Image.ANTIALIAS)
    #img = np.array(img)
    #normalized = (img-min(img))/(max(img)-min(img))
    #img = np.array(img)
    #img=normalized
    #swap_im_axis=np.swapaxes(resize_im, 0, 2).astype(float)
    
    if self.transform is not None:
      read_im=self.transform(img)

    #swap_im_axis=np.swapaxes(read_im=, 0, 2).astype(float)
    

    return read_im


  def __len__(self):
    return len(self.labels_dir['label'])  


path_images="/home/imranr/irgos_screen/images"
path_label="/home/imranr/irgos_screen/folddataset.csv"

data=pd.read_csv(path_label)
fold=0
train_df = data[data.fold != fold].reset_index(drop=True)
val_df = data[data.fold == fold].reset_index(drop=True)

# Dataset object
valid_transform=transforms.Compose([#transforms.Resize(224),
                                   #transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   ])
train_dataset=eye_img(path_images,train_df,trans2)
valid_dataset=eye_img(path_images,val_df,valid_transform)
print(len(train_dataset))
print(len(valid_dataset))

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Data Loader

batch=128

train_dataloader = DataLoader(train_dataset, batch_size= batch,num_workers=12, shuffle=True)

valid_dataloader = DataLoader(valid_dataset, batch_size= batch,num_workers=12, shuffle=False)




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

model = AE(input_size=(3,128,128), z_dim=z_dim, fmap_sizes=model_feature_map_sizes)
model=nn.DataParallel(model)
#inp=torch.rand(1,3,128,128)

n_epochs=35
from tqdm import tqdm
import torch.optim as optim
optimizer=optim.Adam(model.parameters(),lr=0.0001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for epoch in range(n_epochs):
    ### Train
    model.train()
    model.to(device)
    train_loss = 0
    print("\nStart epoch ", epoch)
    data_loader_ = tqdm(enumerate(train_dataloader))
    for batch_idx, data in data_loader_:
        inpt = data.to(device)
        optimizer.zero_grad()
        inpt_rec = model(inpt)
        loss = torch.mean(torch.pow(inpt - inpt_rec, 2))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    print(f"====> Epoch: {epoch} Average loss: {train_loss / len(train_dataloader):.4f}")

    ### Validate
    model.eval()

    val_loss = 0
    with torch.no_grad():
        data_loader_ = tqdm(enumerate(valid_dataloader))
        data_loader_.set_description_str("Validating")
        for i, data in data_loader_:
            inpt = data.to(device)
            inpt_rec = model(inpt)

            loss = torch.mean(torch.pow(inpt - inpt_rec, 2))
            val_loss += loss.item()

    print(f"====> Epoch: {epoch} Validation loss: {val_loss / len(valid_dataloader):.4f}")
PATH='/home/imranr/irgos_screen/model_weights.pth'
torch.save(model.state_dict(), PATH)