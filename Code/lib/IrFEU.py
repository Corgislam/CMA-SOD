import torch
import torch.nn as nn
import torch.nn.functional as F
from Code.lib.BaseBlock import *


class IrFEU(nn.Module):
    def __init__(self, input_channels, squeeze_ratio=2):
        super(IrFEU, self).__init__()
        intermediate_channels = input_channels // squeeze_ratio

        self.rgb_conv = nn.Conv2d(input_channels, intermediate_channels, kernel_size=1)
        self.depth_conv = nn.Conv2d(input_channels, intermediate_channels, kernel_size=1)
        self.rgbd_conv1 = nn.Conv2d(input_channels, intermediate_channels, kernel_size=1)
        self.rgbd_conv2 = nn.Conv2d(input_channels, intermediate_channels, kernel_size=1)

    def forward(self, rgb_input, depth_input, rgbd_input):
        B, C, H, W = rgb_input.size()
        P = H * W

        rgb_features = self.rgb_conv(rgb_input).view(B, -1, P).permute(0, 2, 1)  # [B, HW, C]
        depth_features = self.depth_conv(depth_input).view(B, -1, P)  # [B, C, HW]
        relevance_matrix = F.softmax(torch.bmm(rgb_features, depth_features), dim=-1)  # [B, HW, HW]

        rgbd_features1 = self.rgbd_conv1(rgbd_input).view(B, -1, P).permute(0, 2, 1)  # [B, HW, C]
        rgbd_features2 = self.rgbd_conv2(rgbd_input).view(B, -1, P)  # [B, C, HW]
        rgbd_relevance_matrix = F.softmax(torch.bmm(rgbd_features1, rgbd_features2), dim=-1)

        combined_weights = F.softmax(torch.mul(relevance_matrix, rgbd_relevance_matrix), dim=-1)  # [B, HW, HW]

        # Process RGB
        rgb_flattened = rgb_input.view(B, -1, P)  # [B, C, HW]
        rgb_refined = torch.bmm(rgb_flattened, combined_weights).view(B, C, H, W)
        rgb_output = rgb_input + rgb_refined

        # Process Depth
        depth_flattened = depth_input.view(B, -1, P)  # [B, C, HW]
        depth_refined = torch.bmm(depth_flattened, combined_weights).view(B, C, H, W)
        depth_output = depth_input + depth_refined

        return rgb_output, depth_output

