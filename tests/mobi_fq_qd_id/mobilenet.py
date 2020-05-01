#
# mobilenet.py
# Manuele Rusci <manuele.rusci@unibo.it>
#
# Copyright (C) 2019 University of Bologna
# All rights reserved.
#
# This is an implementation of the quantized mobilenet built from 
# https://github.com/marvis/pytorch-mobilenet/blob/master/main.py
#

import PIL
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torchvision.transforms as transforms
import torch.nn.functional as F

import nemo


###### Full Precision Blocks #############
def conv_dw(inp, oup, stride, pad1=0, bias_ena=False):
    padding = (1,1,1,1) if stride==1 else (0,1,0,1)
    return nn.Sequential(
        nn.ConstantPad2d(padding, value=0.),
        nn.Conv2d(inp, inp, 3, stride, 0, groups=inp, bias=bias_ena),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=False)
    )

def conv_pw(inp, oup, stride,bias_ena=False):
    padding = (0,0,0,0)
    return nn.Sequential(
        nn.ConstantPad2d(padding, value=0.),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=bias_ena),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=False)
    )

def conv_bn(inp, oup, stride):
    padding = (1,1,1,1) if stride==1 else (0,1,0,1)
    return nn.Sequential(
        nn.ConstantPad2d(padding, value=0.),
        nn.Conv2d(inp, oup, 3, stride, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=False)
    )

class mobilenet_real(nn.Module):
    def __init__(self, width_mult=1.0, input_dim = 224):
        super(mobilenet_real, self).__init__()
        print(width_mult, input_dim)

        if input_dim == 224:
            avg_size = 7
            crop_size = 256
        elif input_dim == 192:
            avg_size = 6
            crop_size = 220
        elif input_dim == 160:
            avg_size = 5
            crop_size = 180
        elif input_dim == 128:
            avg_size = 4
            crop_size = 146
        else:
            return -1
        self.width_mult = width_mult
        self.model = nn.Sequential(
            conv_bn(                  3,  int(width_mult* 32), 2), 
            conv_dw( int(width_mult* 32), int(width_mult* 64), 1),
            conv_pw( int(width_mult* 32), int(width_mult* 64), 1),
            conv_dw( int(width_mult* 64), int(width_mult*128), 2),
            conv_pw( int(width_mult* 64), int(width_mult*128), 2),
            conv_dw( int(width_mult*128), int(width_mult*128), 1),           
            conv_pw( int(width_mult*128), int(width_mult*128), 1),       
            conv_dw( int(width_mult*128), int(width_mult*256), 2),
            conv_pw( int(width_mult*128), int(width_mult*256), 2),
            conv_dw( int(width_mult*256), int(width_mult*256), 1),           
            conv_pw( int(width_mult*256), int(width_mult*256), 1),
            conv_dw( int(width_mult*256), int(width_mult*512), 2),
            conv_pw( int(width_mult*256), int(width_mult*512), 2),
            conv_dw( int(width_mult*512), int(width_mult*512), 1),
            conv_pw( int(width_mult*512), int(width_mult*512), 1),           
            conv_dw( int(width_mult*512), int(width_mult*512), 1),
            conv_pw( int(width_mult*512), int(width_mult*512), 1),           
            conv_dw( int(width_mult*512), int(width_mult*512), 1),           
            conv_pw( int(width_mult*512), int(width_mult*512), 1),           
            conv_dw( int(width_mult*512), int(width_mult*512), 1),           
            conv_pw( int(width_mult*512), int(width_mult*512), 1),           
            conv_dw( int(width_mult*512), int(width_mult*512), 1),           
            conv_pw( int(width_mult*512), int(width_mult*512), 1),            
            conv_dw( int(width_mult*512), int(width_mult*1024), 2),            
            conv_pw( int(width_mult*512), int(width_mult*1024), 2),
            conv_dw( int(width_mult*1024), int(width_mult*1024), 1),            
            conv_pw( int(width_mult*1024), int(width_mult*1024), 1),
            nn.AvgPool2d(avg_size),
        )
        self.fc = nn.Linear( int(width_mult*1024), 1000) 

        self.regime = {
            0: {'optimizer': 'Adam', 'lr': 1e-4 },
            5: {'lr': 5e-5},
            8: {'lr': 1e-5 }
        }

        #prepocessing
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.input_transform = {
            'train': transforms.Compose([
                transforms.Scale(crop_size), #, interpolation=PIL.Image.BILINEAR),
                transforms.RandomCrop(input_dim),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
            'eval': transforms.Compose([
                transforms.Scale(crop_size), #, interpolation=PIL.Image.BILINEAR),
                transforms.CenterCrop(input_dim),
                transforms.ToTensor(),
                normalize
            ])
        }

    def forward(self, x):
        x = self.model(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


def mobilenet(activ_bits =None, weight_bits= None, width_mult=1.0, input_dim = 224,**kwargs):
    print(','.join('{0}={1!r}'.format(k,v) for k,v in kwargs.items()))

    print(activ_bits, weight_bits)

    model =  mobilenet_real(width_mult, input_dim)

    return model

