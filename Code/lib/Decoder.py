import torch
import torch.nn as nn
import torch.nn.functional as F
from Code.lib.BaseBlock import BaseConv2d, ChannelAttention


class Deconv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Deconv, self).__init__()
        self.conv1 = BaseConv2d(in_channels * 2, out_channels, kernel_size=3, padding=1)
        self.conv2 = BaseConv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, fea_before, fea_vgg):
        fea_mix = self.conv1(torch.cat((fea_before, fea_vgg), dim=1))
        fea_out = self.conv2(fea_mix)

        return fea_out


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        channels = [64, 128, 256, 512, 512]

        self.r1 = Deconv(channels[4], channels[3])
        self.r2 = Deconv(channels[3], channels[2])
        self.r3 = Deconv(channels[2], channels[1])
        self.r4 = Deconv(channels[1], channels[0])
        self.r5 = Deconv(channels[0], 3)

        self.d1 = Deconv(channels[4], channels[3])
        self.d2 = Deconv(channels[3], channels[2])
        self.d3 = Deconv(channels[2], channels[1])
        self.d4 = Deconv(channels[1], channels[0])
        self.d5 = Deconv(channels[0], 3)


    def forward(self, rgb_list, depth_list):
        rgb_de_list = []
        depth_de_list = []
        # rgb deconv stream
        rgb_block5 = self.r1(rgb_list[5], rgb_list[4])
        rgb_block5_up = F.interpolate(rgb_block5, scale_factor=2, mode='bilinear')
        rgb_block4 = self.r2(rgb_block5_up, rgb_list[3])
        rgb_block4_up = F.interpolate(rgb_block4, scale_factor=2, mode='bilinear')
        rgb_block3 = self.r3(rgb_block4_up, rgb_list[2])
        rgb_block3_up = F.interpolate(rgb_block3, scale_factor=2, mode='bilinear')
        rgb_block2 = self.r4(rgb_block3_up, rgb_list[1])
        rgb_block2_up = F.interpolate(rgb_block2, scale_factor=2, mode='bilinear')
        rgb_block1 = self.r5(rgb_block2_up, rgb_list[0])
        rgb_de_list.extend([rgb_block5, rgb_block4, rgb_block3, rgb_block2, rgb_block1])

        # depth deconv stream
        depth_block5 = self.d1(depth_list[5], depth_list[4])
        depth_block5_up = F.interpolate(depth_block5, scale_factor=2, mode='bilinear')
        depth_block4 = self.d2(depth_block5_up, depth_list[3])
        depth_block4_up = F.interpolate(depth_block4, scale_factor=2, mode='bilinear')
        depth_block3 = self.d3(depth_block4_up, depth_list[2])
        depth_block3_up = F.interpolate(depth_block3, scale_factor=2, mode='bilinear')
        depth_block2 = self.d4(depth_block3_up, depth_list[1])
        depth_block2_up = F.interpolate(depth_block2, scale_factor=2, mode='bilinear')
        depth_block1 = self.d5(depth_block2_up, depth_list[0])
        depth_de_list.extend([depth_block5, depth_block4, depth_block3, depth_block2, depth_block1])





        return rgb_de_list,depth_de_list



