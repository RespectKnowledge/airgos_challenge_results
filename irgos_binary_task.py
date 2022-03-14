# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 14:06:03 2022

@author: Administrateur
"""

import numpy as np
import pandas as pd
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
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
# trans1=transforms.Compose([
#         #transforms.RandomCrop(256),
#         #transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

trans1=transforms.Compose([
        #transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.ToTensor()])

trans2=transforms.Compose([
        #transforms.RandomCrop(256),
        #transforms.RandomHorizontalFlip(0.5),
        #transforms.RandomVerticalFlip(0.5),
        transforms.ToTensor()])

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
    img = read_im.resize((224,224), Image.ANTIALIAS)
    #swap_im_axis=np.swapaxes(resize_im, 0, 2).astype(float)
    
    if self.transform is not None:
      read_im=self.transform(img)

    #swap_im_axis=np.swapaxes(read_im=, 0, 2).astype(float)
    

    return read_im,lb


  def __len__(self):
    return len(self.labels_dir['label'])  


path_images="/home/imranr/irgos_screen/images"
path_label="/home/imranr/irgos_screen/folddataset.csv"

data=pd.read_csv(path_label)
fold=0
train_df = data[data.fold != fold].reset_index(drop=True)
val_df = data[data.fold == fold].reset_index(drop=True)

# Dataset object

train_dataset=eye_img(path_images,train_df,trans1)
valid_dataset=eye_img(path_images,val_df,trans2)
print(len(train_dataset))
print(len(valid_dataset))
#x,y=ob_dst[0]

#print(len(ob_dst))

# Show image
#plt.imshow(x)


# Check 0's and 1's in label
#y_zeroes=[]

#for i in range(0,len(train_dataset)):
  #x,y=train_dataset[i]
  #if y==0:
   # y_zeroes.append(0)

#print(len(y_zeroes))

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Data Loader

batch=64

train_dataloader = DataLoader(train_dataset, batch_size= batch,pin_memory=True,num_workers=6, shuffle=True)

valid_dataloader = DataLoader(valid_dataset, batch_size= batch,pin_memory=True,num_workers=6, shuffle=False)

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
# def seed_everything(seed):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#seed = 123
#seed_everything(seed)

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
        self.net =convnext_base(pretrained=True)
        #checkpoint = torch.load("../input/efficientnet-pytorch/efficientnet-b0-08094119.pth")
        #self.net.load_state_dict(checkpoint)
        #out = out.view(-1, self.in_planes)
        self.net.head = nn.Linear(in_features=1024, out_features=2, bias=True)
    
    def forward(self, x):
        #x = F.relu(self.map(x))
        out,out1 = self.net(x)
        return out,out1
    
#model=CNN()

##### transformer

import timm
print("Available Vision Transformer Models: ")
print(timm.list_models("vit*"))
import timm
#################################################### transformer model #############################
class ViTBase16(nn.Module):
    def __init__(self, n_classes, pretrained=False):

        super(ViTBase16, self).__init__()

        self.model = timm.create_model("vit_base_patch16_224_miil_in21k", pretrained=pretrained)
        #if pretrained:
        #   self.model = timm.create_model(CFG['model_arch'], pretrained=True)
            #self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.head = nn.Linear(self.model.head.in_features, n_classes)
        # self.model.head = nn.Sequential(nn.Linear(self.model.head.in_features, 512),
        #                                 nn.Dropout(0.5),
        #                                 nn.ReLU(True),
        #                                 nn.Linear(512,n_classes),
        #                                 )
        #self.model.classifier = nn.Linear(self.model.classifier.in_features, n_classes)

    def forward(self, x):
        x = self.model(x)
        return x
    
model = ViTBase16(n_classes=2, pretrained=True).to(device)

model=nn.DataParallel(model)
model=model.to(device)
#loss_func=nn.BCEWithLogitsLoss()
from tqdm import tqdm
import torch.optim as optim
optimizer=optim.Adam(model.parameters(),lr=0.0001)

classes=['NRG','RG']
c1=data['label'].value_counts()[0]
c2=data['label'].value_counts()[1]
my_distribution=np.array([c1,c2])
class_weights = torch.from_numpy(np.divide(1, my_distribution)).float().to(device)
class_weights = class_weights / class_weights.sum()
for i, c in enumerate(classes):
  print('Weight for class %s: %f' % (c, class_weights.cpu().numpy()[i]))
loss_func = nn.CrossEntropyLoss(weight=class_weights)
#label=label.to(torch.int64)
################################ training functions ###################
def train_fn(model,train_loader):
    model.train()
    counter=0
    training_run_loss=0.0
    train_running_correct=0.0
    for i, data in tqdm(enumerate(train_loader),total=int(len(train_dataset)/train_loader.batch_size)):
        counter+=1
        # extract dataset
        imge,label=data
        imge=imge.float()
        #label=label.float()
        label.to(torch.int64)
        imge=imge.to(device)
        label=label.to(device)
        #imge=imge.cuda()
        #label=label.cuda()
        # zero_out the gradient
        optimizer.zero_grad()
        output=model(imge)
        loss=loss_func(output,label)
        training_run_loss+=loss.item()
        _,preds=torch.max(output.data,1)
        train_running_correct+=(preds==label).sum().item()
        loss.backward()
        optimizer.step()
    ###################### state computation ###################
    train_loss=training_run_loss/len(train_loader.dataset)
    train_loss_ep.append(train_loss)
    train_accuracy=100.* train_running_correct/len(train_loader.dataset)
    train_accuracy_ep.append(train_accuracy)
    print(f"Train Loss:{train_loss:.4f}, Train Acc:{train_accuracy:0.2f}")
    return train_loss_ep,train_accuracy_ep

########################## validation function ##################
def validation_fn(model,valid_loader):
  # evluation start
    print("validation start")
    
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    with torch.no_grad():
        for i,data in tqdm(enumerate(valid_loader),total=int(len(valid_loader)/valid_loader.batch_size)):
            imge,label=data
            imge=imge.float()
            #label=label.float()
            label.to(torch.int64)
            imge=imge.to(device)
            label=label.to(device)
            #imge=imge.cuda()
            #label=label.cuda()
            output=model(imge)
            loss=loss_func(output,label)
            val_running_loss+=loss.item()
            _,pred=torch.max(output.data,1)
            val_running_correct+=(pred==label).sum().item()
        val_loss=val_running_loss/len(valid_loader.dataset)
        val_loss_ep.append(val_loss)
        val_accuracy=100.* val_running_correct/(len(valid_loader.dataset))
        val_accuracy_ep.append(val_accuracy)
        print(f"Val Loss:{val_loss:0.4f}, Val_Acc:{val_accuracy:0.2f}")
        return val_loss_ep,val_accuracy_ep

import torch.optim as optim
optimizer=optim.Adam(model.parameters(),lr=0.0001)
train_loss_ep=[]
train_accuracy_ep=[]
val_loss_ep=[]
val_accuracy_ep=[]
lr = 3e-4
log = pd.DataFrame(index=[], columns=['epoch', 'lr', 'loss', 'accu', 'val_loss', 'val_accu'])
early_stop=20
epochs=500
best_acc = 0
name='transformer2d'
trigger = 0
for epoch in range(epochs):
    print('Epoch [%d/%d]' %(epoch, epochs))
    # train for one epoch
    train_loss_ep,train_accuracy_ep=train_fn(model,train_dataloader)
    train_loss_ep1=np.mean(train_loss_ep)
    train_accuracy_ep1=np.mean(train_accuracy_ep)
    #y_pred,labels=Prediciton_fn(model,valid_loader)

    val_loss_ep,val_accuracy_ep=validation_fn(model,valid_dataloader)
    val_loss_ep1=np.mean(val_loss_ep)
    val_accuracy_ep1=np.mean(val_accuracy_ep)
    
    print('loss %.4f - accu %.4f - val_loss %.4f - val_accu %.4f'%(train_loss_ep1, train_accuracy_ep1, val_loss_ep1, val_accuracy_ep1))

    tmp = pd.Series([epoch,lr,train_loss_ep1,train_accuracy_ep1,val_loss_ep1,val_accuracy_ep1], index=['epoch', 'lr', 'loss', 'accu', 'val_loss', 'val_accu'])

    log = log.append(tmp, ignore_index=True)
    log.to_csv('models/%s/log.csv' %name, index=False)

    trigger += 1

    if val_accuracy_ep1 > best_acc:
        torch.save(model.state_dict(), 'models/%s/2dmodelairgos.pth' %name)
        best_acc = val_accuracy_ep1
        print("=> saved best model")
        trigger = 0

    # early stopping
    if not early_stop is None:
        if trigger >= early_stop:
            print("=> early stopping")
            break

    torch.cuda.empty_cache() 