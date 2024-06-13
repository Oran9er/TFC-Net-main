# -*- coding: utf-8 -*-
from __future__ import division, print_function

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions.uniform import Uniform

def Dropout(x, p=0.5):
    x = torch.nn.functional.dropout(x, p)
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


class SpatialAttention(nn.Module):
    def __init__(self, in_channels_up):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels_up, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att = self.conv(x)
        att = self.sigmoid(att)
        return att * x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels_down):
        super(ChannelAttention, self).__init__()
        self.att_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels_down, in_channels_down // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels_down // 2, in_channels_down),
            nn.Sigmoid()
        )

    def forward(self, x):
        att = self.att_pool(x).view(x.size(0), -1)
        att = self.fc(att).view(x.size(0), x.size(1), 1, 1)
        return att * x


class AttentionBlock(nn.Module):
    def __init__(self, in_channels_up, in_channels_down):
        super(AttentionBlock, self).__init__()

        self.spatial_attention = SpatialAttention(in_channels_up)
        self.channel_attention = ChannelAttention(in_channels_down)
        self.convadjus = nn.Conv2d(in_channels=in_channels_down, out_channels=in_channels_up, kernel_size=1)

    def forward(self, x_encoder, x_decoder):
        x_encoder_att = self.spatial_attention(x_encoder)
        x_decoder_att = self.channel_attention(x_decoder)
        x_decoder_att = F.interpolate(x_decoder_att, size=x_encoder_att.shape[2:], mode='bilinear', align_corners=False)
        x_decoder_att = self.convadjus(x_decoder_att)

        x_decoder_att += x_encoder_att

        return x_decoder_att



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

class DiaConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dia):
        super(DiaConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dia, dilation=dia),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


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
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, mode_upsampling=1):
        super(UpBlock, self).__init__()
        self.mode_upsampling = mode_upsampling
        if mode_upsampling==0:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        elif mode_upsampling==1:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif mode_upsampling==2:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        elif mode_upsampling==3:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.mode_upsampling != 0:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]

class MainDecoder(nn.Module):
    def __init__(self, params):
        super(MainDecoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        assert (len(self.ft_chns) == 5)

        self.att1 = AttentionBlock(in_channels_up=self.ft_chns[3], in_channels_down=self.ft_chns[4])
        self.att2 = AttentionBlock(in_channels_up=self.ft_chns[2], in_channels_down=self.ft_chns[3])
        self.att3 = AttentionBlock(in_channels_up=self.ft_chns[1], in_channels_down=self.ft_chns[2])

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, mode_upsampling=self.up_type)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x3_att = self.att1(x3, x4)
        x = self.up1(x4, x3_att)

        x2_att = self.att2(x2, x)
        x = self.up2(x, x2_att)

        x1_att = self.att3(x1, x)
        x = self.up3(x, x1_att)

        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output

class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, mode_upsampling=self.up_type)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

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

    
class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params1)
        self.decoder1 = Decoder(params1)

    def forward(self, x):
        feature = self.encoder(x)
        output1 = self.decoder1(feature)
        return output1



class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()
        self.conv1 = ConvBlock(64, 128, 0.1)
        self.conv2 = ConvBlock(128, 128, 0.1)
        self.conv3 = ConvBlock(256, 128, 0.1)

    def forward(self, x2, x3, x4):
        x2 = self.conv1(x2)
        x3 = self.conv2(x3)
        x4 = self.conv3(x4)
        upsample = nn.Upsample(size=x2.size()[2:], mode='bilinear', align_corners=True)
        x3 = upsample(x3)
        x4 = upsample(x4)
        out = torch.cat([x2, x3, x4], dim=1)
        return out

class GetPseudoLabel(nn.Module):
    def __init__(self):
        super(GetPseudoLabel, self).__init__()

    def forward(self, out0, out1, out2, out3, is_train):
        label0 = F.softmax(out0, dim=1)
        label1 = F.softmax(out1, dim=1)
        label2 = F.softmax(out2, dim=1)
        label3 = F.softmax(out3, dim=1)

        if is_train == 1:
            lbl_weight = np.random.dirichlet(np.ones(4), size=1)[0]
            un_lbl_pseudo = torch.argmax((lbl_weight[0] * label0.detach() \
                                          + lbl_weight[1] * label1.detach() \
                                          + lbl_weight[2] * label2.detach() \
                                          + lbl_weight[3] * label3.detach()), dim=1, keepdim=False).cuda()
        elif is_train == 0:
            un_lbl_pseudo = torch.argmax((1 * label0.detach()), dim=1, keepdim=False).cuda()

        return un_lbl_pseudo

class GetRefinePseudoLabel(nn.Module):
    def __init__(self):
        super(GetRefinePseudoLabel, self).__init__()
        self.concat = Concat()
        self.diaconv1 = DiaConvBlock(in_channels=384, out_channels=128, dia=1)
        self.diaconv2 = DiaConvBlock(in_channels=384, out_channels=128, dia=3)
        self.diaconv3 = DiaConvBlock(in_channels=384, out_channels=128, dia=5)

        self.att = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2, stride=2))
        self.deconv = nn.Sequential(nn.ConvTranspose2d(16, 16, kernel_size=4, stride=4, padding=0),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU())

        self.out_conv = nn.Conv2d(16, 2, kernel_size=3, padding=1)

    def forward(self, pseudo, x2, x3, x4):

        x = self.concat(x2, x3, x4)
        out1 = self.diaconv1(x)
        out2 = self.diaconv2(x)
        out3 = self.diaconv3(x)
        out = out1 + out2 + out3
        att_map = self.att(out)
        pseudo = pseudo.unsqueeze(1).float()
        pseudo = self.conv1(pseudo)
        pseudo = self.conv2(pseudo)
        pseudo_label_refine = pseudo * att_map
        pseudo_label_refine = self.deconv(pseudo_label_refine)
        pseudo_label_refine = self.out_conv(pseudo_label_refine)
        return pseudo_label_refine

class TFCNetv2(nn.Module):
    def __init__(self, in_chns, class_num):
        super(TFCNetv2, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.main_decoder = MainDecoder(params)
        self.aux_decoder1 = Decoder(params)
        self.aux_decoder2 = Decoder(params)
        self.aux_decoder3 = Decoder(params)

        self.pseudo = GetPseudoLabel()
        self.pseudo_refine = GetRefinePseudoLabel()

    def forward(self, x, n):
        feature = self.encoder(x)
        main_seg = self.main_decoder(feature)
        aux1_feature = [FeatureNoise()(i) for i in feature]
        aux_seg1 = self.aux_decoder1(aux1_feature)
        aux2_feature = [Dropout(i) for i in feature]
        aux_seg2 = self.aux_decoder2(aux2_feature)
        aux3_feature = [FeatureDropout(i) for i in feature]
        aux_seg3 = self.aux_decoder3(aux3_feature)

        x2 = feature[2]  #torch.Size([24, 64, 64, 64])
        x3 = feature[3]  #torch.Size([24, 128, 32, 32])
        x4 = feature[4]  #torch.Size([24, 256, 16, 16])

        is_train = n

        pseudo_label = self.pseudo(main_seg, aux_seg1, aux_seg2, aux_seg3, is_train)

        pseudo_label_refine = self.pseudo_refine(pseudo_label, x2, x3, x4)

        return main_seg, aux_seg1, aux_seg2, aux_seg3, pseudo_label_refine, pseudo_label


if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from ptflops import get_model_complexity_info
    model = UNet(in_chns=1, class_num=4).cuda()
    with torch.cuda.device(0):
      macs, params = get_model_complexity_info(model, (1, 256, 256), as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
      print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
      print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    import ipdb; ipdb.set_trace()
