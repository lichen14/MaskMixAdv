# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
import copy
import PIL
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
import matplotlib.pyplot as plt

def RResize(x,row,col):
    resize=Compose([
            # ToTensor(),
            Resize((row,col),interpolation=PIL.Image.NEAREST),
        ])
    return resize(x)

class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        # print(x.shape,x0.shape,x1.shape,x2.shape,x3.shape,x4.shape)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output


class Decoder_DS(nn.Module):
    def __init__(self, params):
        super(Decoder_DS, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,
                                      kernel_size=3, padding=1)

    def forward(self, feature, shape):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x = self.up1(x4, x3)
        dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        x = self.up2(x, x2)
        dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


class Decoder_URDS(nn.Module):
    def __init__(self, params):
        super(Decoder_URDS, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,
                                      kernel_size=3, padding=1)
        self.feature_noise = FeatureNoise()

    def forward(self, feature, shape):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x = self.up1(x4, x3)
        if self.training:
            dp3_out_seg = self.out_conv_dp3(Dropout(x, p=0.5))
        else:
            dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        x = self.up2(x, x2)
        if self.training:
            dp2_out_seg = self.out_conv_dp2(FeatureDropout(x))
        else:
            dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        if self.training:
            dp1_out_seg = self.out_conv_dp1(self.feature_noise(x))
        else:
            dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


def Dropout(x, p=0.5):
    x = torch.nn.functional.dropout2d(x, p)
    return x


def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x

def FeatureMask(x):
    # attention = torch.mean(x, dim=1, keepdim=True)
    # max_val, _ = torch.max(attention.view(
    #     x.size(0), -1), dim=1, keepdim=True)
    # threshold = max_val * np.random.uniform(0.7, 0.9)
    # threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    # drop_mask = (attention < threshold).float()

    masked_image1 = x#copy.deepcopy(x)
    # print(x.shape)  #torch.Size([3, 16, 256, 256])
    img_deps,_,img_rows, img_cols = masked_image1.shape  #(3, 512, 512)
    mask_rows = 32
    mask_cols = 32
    num_patches = mask_rows* mask_cols 
    num_mask = int(0.5 * num_patches)
    # print("mask generating: total patches {}, mask patches {}".format(num_patches,num_mask))
    mask = torch.zeros([mask_rows, mask_cols])#np.random.rand(mask_rows, mask_cols)
    # print(mask.shape)
    # np.hstack(
    #     [np.ones(num_patches-num_mask),
    #      np.zeros(num_mask),]
    # ).reshape(mask_rows,mask_cols)#.astype(bool)
    # print('mask',mask)
    # np.random.shuffle(mask)
    mask=torch.reshape(mask,(num_patches,1))
    # print(mask.shape)
    mask[:num_mask] =1
    mask[num_mask:] = 0
    shuffle_index=torch.randperm(num_patches)
    # torch.random.shuffle(mask)
    # print(mask)
    mask=torch.reshape(mask[shuffle_index],(1,mask_rows, mask_cols))
    # print('mask',mask.shape)  #mask torch.Size([3, 16, 256, 256])
    
    # print(mask.shape)
    mask1 = RResize(mask,img_rows, img_cols)
    # mask = cv2.resize(mask, dsize=None,fx=16,fy=16,interpolation=cv2.INTER_LINEAR)
    # mask = np.resize(mask, (img_rows, img_cols))
    # print(mask1)
    mask1[mask1>0.5] =1
    mask1[mask1<0.5]=0
    # print('mask1',mask1.shape)  #mask torch.Size([3, 16, 256, 256])
    # print(x)
    x = x.mul(mask1.cuda())
    # print(x)
    return x

def FeatureMask_0117(x):
    # attention = torch.mean(x, dim=1, keepdim=True)
    # max_val, _ = torch.max(attention.view(
    #     x.size(0), -1), dim=1, keepdim=True)
    # threshold = max_val * np.random.uniform(0.7, 0.9)
    # threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    # drop_mask = (attention < threshold).float()

    masked_image1 = x#copy.deepcopy(x)
    # print(x.shape)  #torch.Size([3, 16, 256, 256])
    img_deps,_,img_rows, img_cols = masked_image1.shape  #(3, 512, 512)
    mask_rows = 32
    mask_cols = 32
    num_patches = mask_rows* mask_cols 
    num_mask = int(0.5 * num_patches)
    # print("mask generating: total patches {}, mask patches {}".format(num_patches,num_mask))
    mask = torch.zeros([mask_rows, mask_cols])#np.random.rand(mask_rows, mask_cols)
    # print(mask.shape)
    # np.hstack(
    #     [np.ones(num_patches-num_mask),
    #      np.zeros(num_mask),]
    # ).reshape(mask_rows,mask_cols)#.astype(bool)
    # print('mask',mask)
    # np.random.shuffle(mask)
    mask=torch.reshape(mask,(num_patches,1))
    # print(mask.shape)
    mask[:num_mask] =1
    mask[num_mask:] = 0
    shuffle_index=torch.randperm(num_patches)
    # torch.random.shuffle(mask)
    # print(mask)
    mask=torch.reshape(mask[shuffle_index],(1,mask_rows, mask_cols))
    # print('mask',mask.shape)  #mask torch.Size([3, 16, 256, 256])
    
    # print(mask.shape)
    mask1 = RResize(mask,img_rows, img_cols)
    # mask = cv2.resize(mask, dsize=None,fx=16,fy=16,interpolation=cv2.INTER_LINEAR)
    # mask = np.resize(mask, (img_rows, img_cols))
    # print(mask1)
    mask1[mask1>0.5] =1
    mask1[mask1<0.5]=0
    mask2=1-mask1
    # print('mask1',mask1.shape)  #mask torch.Size([3, 16, 256, 256])
    # print(x)
    y = x.mul(mask2.cuda())
    x = x.mul(mask1.cuda())
    # print(x)
    return x,y

class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(
            x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output


class UNet_DS(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_DS, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.decoder = Decoder_DS(params)

    def forward(self, x):
        shape = x.shape[2:]
        feature = self.encoder(x)
        dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg = self.decoder(
            feature, shape)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


class UNet_CCT(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_CCT, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.main_decoder = Decoder(params)
        self.aux_decoder1 = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        main_seg = self.main_decoder(feature)
        aux1_feature = [FeatureDropout(i) for i in feature]
        aux_seg1 = self.aux_decoder1(aux1_feature)
        return main_seg, aux_seg1


class UNet_CCT_3H(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_CCT_3H, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.main_decoder = Decoder(params)
        self.aux_decoder1 = Decoder(params)
        self.aux_decoder2 = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        main_seg = self.main_decoder(feature)
        aux1_feature = [Dropout(i) for i in feature]
        aux_seg1 = self.aux_decoder1(aux1_feature)
        aux2_feature = [FeatureNoise()(i) for i in feature]
        aux_seg2 = self.aux_decoder1(aux2_feature)
        return main_seg, aux_seg1, aux_seg2

class UNet_MAE(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_MAE, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        params_aux = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': in_chns,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.seg_decoder = Decoder(params)
        self.mae_decoder = Decoder(params_aux)

    def forward(self, x):
        feature = self.encoder(x)
        main_seg = self.seg_decoder(feature)
        # aux1_feature = [Dropout(i) for i in feature]
        restore_img = self.mae_decoder(feature)
        return main_seg, restore_img

class UNet_MAE_CCT(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_MAE_CCT, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        params_aux = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': in_chns,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.seg_decoder = Decoder(params)
        self.mae_decoder = Decoder(params_aux)
        self.aux_decoder = Decoder(params)


    # def forward(self, x):
    #     feature = self.encoder(x)
    #     main_seg = self.seg_decoder(feature)
    #     aux1_feature = [Dropout(i) for i in feature]
    #     aux_seg1 = self.aux_decoder(aux1_feature)
    #     restore_img = self.mae_decoder(feature)
    #     return main_seg, aux_seg1,restore_img
    def forward(self, x1,x2):
        feature1 = self.encoder(x1)
        main_seg = self.seg_decoder(feature1)
        feature2 = self.encoder(x2)
        aux_feature = [Dropout(i) for i in feature2]
        aux_seg = self.aux_decoder(aux_feature)
        restore_img = self.mae_decoder(feature1)
        return main_seg, aux_seg,restore_img

class UNet_MAE_CCT_0109(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_MAE_CCT_0109, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        params_aux = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': in_chns,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.seg_decoder = Decoder(params)
        self.mae_decoder = Decoder(params_aux)
        self.aux_decoder = Decoder(params)


    # def forward(self, x):
    #     feature = self.encoder(x)
    #     main_seg = self.seg_decoder(feature)
    #     aux1_feature = [Dropout(i) for i in feature]
    #     aux_seg1 = self.aux_decoder(aux1_feature)
    #     restore_img = self.mae_decoder(feature)
    #     return main_seg, aux_seg1,restore_img
    def forward(self, x):
        feature1 = self.encoder(x)

        main_seg = self.seg_decoder(feature1)
        # feature2 = self.encoder(x2)
        aux_feature = [FeatureDropout(i) for i in feature1]
        # aux_feature = [FeatureMask(i) for i in feature1]
        aux_seg = self.aux_decoder(aux_feature)
        restore_img = self.mae_decoder(aux_feature)
        return main_seg, aux_seg,restore_img

class UNet_MAE_CCT_0110(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_MAE_CCT_0110, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        params_aux = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': in_chns,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.seg_decoder = Decoder(params)
        self.mae_decoder = Decoder(params_aux)
        self.aux_decoder = Decoder(params)

    #  可视化特征图
    def show_feature_map(self,feature_map):
        print(feature_map.shape)
        feature_map = feature_map.squeeze(0)
        feature_map = feature_map.cpu().numpy()
        feature_map_num = feature_map.shape[0]
        # print(feature_map_num.shape)
        row_num = np.ceil(np.sqrt(feature_map_num))
        plt.figure()
        for index in range(1, feature_map_num+1):
            plt.subplot(row_num, row_num, index)
            plt.imshow(feature_map[index-1], cmap='gray')
            plt.axis('off')
            # scipy.misc.imsave(str(index)+".png", feature_map[index-1])
        plt.show()

    # def forward(self, x):
    #     feature = self.encoder(x)
    #     main_seg = self.seg_decoder(feature)
    #     aux1_feature = [Dropout(i) for i in feature]
    #     aux_seg1 = self.aux_decoder(aux1_feature)
    #     restore_img = self.mae_decoder(feature)
    #     return main_seg, aux_seg1,restore_img
    def forward(self, x):
        feature1 = self.encoder(x)
        main_seg = self.seg_decoder(feature1)
        # print(main_seg[-1].shape)
        # self.show_feature_map(main_seg[-1])
        # feature2 = self.encoder(x2)
        # aux_feature = [FeatureDropout(i) for i in feature1]
        aux_feature = [FeatureMask(i) for i in feature1]
        aux_seg = self.aux_decoder(aux_feature)
        restore_img = self.mae_decoder(aux_feature)
        return main_seg, aux_seg,restore_img

class UNet_MAE_CCT_0111(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_MAE_CCT_0111, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        params_aux = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': in_chns,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.seg_decoder = Decoder(params)
        self.mae_decoder = Decoder(params_aux)
        self.aux_decoder = Decoder(params)


    # def forward(self, x):
    #     feature = self.encoder(x)
    #     main_seg = self.seg_decoder(feature)
    #     aux1_feature = [Dropout(i) for i in feature]
    #     aux_seg1 = self.aux_decoder(aux1_feature)
    #     restore_img = self.mae_decoder(feature)
    #     return main_seg, aux_seg1,restore_img
    def forward(self, x):
        feature1 = self.encoder(x)

        main_seg = self.seg_decoder(feature1)
        # feature2 = self.encoder(x2)
        # aux_feature = [FeatureDropout(i) for i in feature1]
        # aux_feature = [FeatureMask(i) for i in feature1]
        aux_feature = [FeatureNoise()(i) for i in feature1]
        aux_seg = self.aux_decoder(aux_feature)
        restore_img = self.mae_decoder(aux_feature)
        return main_seg, aux_seg,restore_img
        
class UNet_MAE_CCT_0117(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_MAE_CCT_0117, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        params_aux = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': in_chns,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.seg_decoder = Decoder(params)
        self.mae_decoder = Decoder(params_aux)
        self.aux_decoder = Decoder(params)


    # def forward(self, x):
    #     feature = self.encoder(x)
    #     main_seg = self.seg_decoder(feature)
    #     aux1_feature = [Dropout(i) for i in feature]
    #     aux_seg1 = self.aux_decoder(aux1_feature)
    #     restore_img = self.mae_decoder(feature)
    #     return main_seg, aux_seg1,restore_img
    def forward(self, x):
        feature1 = self.encoder(x)
        main_feature=[]
        aux_feature=[]
        for i in feature1:
            main,aux= FeatureMask_0117(i)
            main_feature.append(main),aux_feature.append(aux)
        # feature = [FeatureMask_0117(i) for i in feature1]
        # print(feature.shape)
        # main_feature,aux_feature=feature[0],feature[1]
        # print(main_feature.shape)
        main_seg = self.seg_decoder(main_feature)
        aux_seg = self.aux_decoder(aux_feature)
        restore_img = self.mae_decoder(main_feature)
        return main_seg, aux_seg,restore_img
