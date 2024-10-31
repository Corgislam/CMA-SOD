from einops import rearrange
from Code.lib.ResNet import Backbone_ResNet50
from Code.lib.tensor_ops import *
from Code.lib.Decoder import *
from Code.lib.IrFEU import *


def _get_act_fn(act_name, inplace=True):
    if act_name == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_name == "leaklyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    else:
        raise NotImplementedError

class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        act_name="relu",
    ):
        super().__init__()
        self.add_module(
            name="conv",
            module=nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
        )
        self.add_module(name="bn", module=nn.BatchNorm2d(out_planes))
        if act_name is not None:
            self.add_module(name=act_name, module=_get_act_fn(act_name=act_name, inplace=False))


class StackedCBRBlock(nn.Sequential):
    def __init__(self, in_c, out_c, num_blocks=1, kernel_size=3):
        assert num_blocks >= 1
        super().__init__()

        if kernel_size == 3:
            kernel_setting = dict(kernel_size=3, stride=1, padding=1)
        elif kernel_size == 1:
            kernel_setting = dict(kernel_size=1)
        else:
            raise NotImplementedError

        cs = [in_c] + [out_c] * num_blocks
        self.channel_pairs = self.slide_win_select(cs, win_size=2, win_stride=1, drop_last=True)
        self.kernel_setting = kernel_setting

        for i, (i_c, o_c) in enumerate(self.channel_pairs):
            self.add_module(name=f"cbr_{i}", module=ConvBNReLU(i_c, o_c, **self.kernel_setting))

    @staticmethod
    def slide_win_select(items, win_size=1, win_stride=1, drop_last=False):
        num_items = len(items)
        i = 0
        while i + win_size <= num_items:
            yield items[i : i + win_size]
            i += win_stride

        if not drop_last:
            # 对于最后不满一个win_size的切片，保留
            yield items[i : i + win_size]

class ConvFFN(nn.Module):
    def __init__(self, dim, out_dim=None, ffn_expand=4):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.net = nn.Sequential(
            StackedCBRBlock(dim, dim * ffn_expand, num_blocks=2, kernel_size=3),
            nn.Conv2d(dim * ffn_expand, out_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class Tokenreembedding:
    @staticmethod
    def encode(x, nh, ph, pw):
        return rearrange(x, "b (nh hd) (nhp ph) (nwp pw) -> b nh (hd ph pw) (nhp nwp)", nh=nh, ph=ph, pw=pw)

    @staticmethod
    def decode(x, nhp, ph, pw):
        return rearrange(x, "b nh (hd ph pw) (nhp nwp) -> b (nh hd) (nhp ph) (nwp pw)", nhp=nhp, ph=ph, pw=pw)

####################################################

##TRPMMH
class Spatial_TRPM(nn.Module):
    def __init__(self, dim, p, nh=2):
        super().__init__()
        self.p = p
        self.nh = nh
        self.scale = (dim // nh * self.p ** 2) ** -0.5

        self.to_q = nn.Conv2d(dim, dim, 1)
        self.to_kv = nn.Conv2d(dim, dim * 2, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, q, kv=None, need_weights: bool = False):
        if kv is None:
            kv = q
        N, C, H, W = q.shape

        q = self.to_q(q)
        k, v = torch.chunk(self.to_kv(kv), 2, dim=1)

        # TRPM
        q = Tokenreembedding.encode(q, nh=self.nh, ph=self.p, pw=self.p)
        k = Tokenreembedding.encode(k, nh=self.nh, ph=self.p, pw=self.p)
        v = Tokenreembedding.encode(v, nh=self.nh, ph=self.p, pw=self.p)

        qk = torch.einsum("bndx, bndy -> bnxy", q, k) * self.scale
        qk = qk.softmax(-1)
        qkv = torch.einsum("bnxy, bndy -> bndx", qk, v)

        qkv = Tokenreembedding.decode(qkv, nhp=H // self.p, ph=self.p, pw=self.p)

        x = self.proj(qkv)
        if not need_weights:
            return x
        else:

            return x, qk.mean(dim=1)


class Channel_TRPM(nn.Module):
    def __init__(self, dim, nh):
        super().__init__()
        self.nh = nh
        self.to_q = nn.Conv2d(dim, dim, 1)
        self.to_kv = nn.Conv2d(dim, dim * 2, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, q, kv=None):
        if kv is None:
            kv = q
        B, C, H, W = q.shape

        q = self.to_q(q)
        k, v = torch.chunk(self.to_kv(kv), 2, dim=1)
        q = q.reshape(B, self.nh, C // self.nh, H * W)
        k = k.reshape(B, self.nh, C // self.nh, H * W)
        v = v.reshape(B, self.nh, C // self.nh, H * W)

        q = q * (q.shape[-1] ** (-0.5))
        qk = q @ k.transpose(-2, -1)
        qk = qk.softmax(dim=-1)
        qkv = qk @ v

        qkv = qkv.reshape(B, C, H, W)
        x = self.proj(qkv)
        return x


class MultiScaleSelfAttention(nn.Module):
    def __init__(self, dim, p, nh, ffn_expand):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.sa = Spatial_TRPM(dim, p=p, nh=nh)
        self.ca = Channel_TRPM(dim, nh=nh)
        self.alpha = nn.Parameter(data=torch.zeros(1))
        self.beta = nn.Parameter(data=torch.zeros(1))

        self.norm2 = nn.BatchNorm2d(dim)
        self.ffn = ConvFFN(dim=dim, ffn_expand=ffn_expand, out_dim=dim)

    def forward(self, x):
        normed_x = self.norm1(x)
        x = x + self.alpha.sigmoid() * self.sa(normed_x) + self.beta.sigmoid() * self.ca(normed_x)#残差连接
        x = x + self.ffn(self.norm2(x))
        return x


class CrossModalCrossAttention(nn.Module):
    def __init__(self, dim, p, nh=4, ffn_expand=1):
        super().__init__()
        self.rgb_norm2 = nn.BatchNorm2d(dim)
        self.depth_norm2 = nn.BatchNorm2d(dim)
        self.depth_to_rgb_sa = Spatial_TRPM(dim, p=p, nh=nh)


        self.depth_to_rgb_ca = Channel_TRPM(dim, nh=nh)
        self.rgb_alpha = nn.Parameter(data=torch.zeros(1))
        self.rgb_beta = nn.Parameter(data=torch.zeros(1))

        self.rgb_to_depth_sa = Spatial_TRPM(dim, p=p, nh=nh)


        self.rgb_to_depth_ca = Channel_TRPM(dim, nh=nh)
        self.depth_alpha = nn.Parameter(data=torch.zeros(1))
        self.depth_beta = nn.Parameter(data=torch.zeros(1))

        self.norm3 = nn.BatchNorm2d(2 * dim)
        self.ffn = ConvFFN(dim=2 * dim, ffn_expand=ffn_expand, out_dim=2 * dim)

    def forward(self, rgb, depth):
        normed_rgb = self.rgb_norm2(rgb)
        normed_depth = self.depth_norm2(depth)
        transd_rgb = self.rgb_alpha.sigmoid() * self.depth_to_rgb_sa(
            normed_rgb, normed_depth
        ) + self.rgb_beta.sigmoid() * self.depth_to_rgb_ca(normed_rgb, normed_depth)
        rgb_rgbd = rgb + transd_rgb
        transd_depth = self.depth_alpha.sigmoid() * self.rgb_to_depth_sa(
            normed_depth, normed_rgb
        ) + self.depth_beta.sigmoid() * self.rgb_to_depth_ca(normed_depth, normed_rgb)
        depth_rgbd = depth + transd_depth

        rgbd = torch.cat([rgb_rgbd, depth_rgbd], dim=1)
        rgbd = rgbd + self.ffn(self.norm3(rgbd))
        return rgbd

    #decoder

class CAMoudle(nn.Module):
    def __init__(self, in_dim, embed_dim, p, nh, ffn_expand):
        super().__init__()
        self.p = p
        self.rgb_cnn_proj = nn.Sequential(
            StackedCBRBlock(in_c=in_dim, out_c=embed_dim), nn.Conv2d(embed_dim, embed_dim, 1)
        )
        self.depth_cnn_proj = nn.Sequential(
            StackedCBRBlock(in_c=in_dim, out_c=embed_dim), nn.Conv2d(embed_dim, embed_dim, 1)
        )


        self.cmca = CrossModalCrossAttention(embed_dim, nh=nh, p=p, ffn_expand=ffn_expand)
        self.mssa = MultiScaleSelfAttention(2 * embed_dim, nh=nh, p=p, ffn_expand=ffn_expand)#2*embed_dim

    def forward(self, rgb, depth, before_rgbd=None):
        """NCHW"""
        rgb = self.rgb_cnn_proj(rgb)
        depth = self.depth_cnn_proj(depth)

        rgbd = self.cmca(rgb, depth)
        if before_rgbd is not None:
            rgbd = rgbd + before_rgbd


        rgbd = self.mssa(rgbd)
        return rgbd


class CANet(nn.Module):
    def __init__(self, ps=(8, 8, 8, 8, 8), embed_dim=64, pretrained=None):
        #Resnet fea
        super(CANet, self).__init__()
        (
            self.rgb_block1,
            self.rgb_block2,
            self.rgb_block3,
            self.rgb_block4,
            self.rgb_block5,
        ) = Backbone_ResNet50(pretrained=True)

        (
            self.depth_block1,
            self.depth_block2,
            self.depth_block3,
            self.depth_block4,
            self.depth_block5,
        ) = Backbone_ResNet50(pretrained=True)
        #dennet fea


        self.layer_dep = nn.Conv2d(1, 3, kernel_size=1)


        self.CAM = nn.ModuleList(
            [
                CAMoudle(in_dim=c, embed_dim=embed_dim, p=p, nh=2, ffn_expand=1)
                for i, (p, c) in enumerate(zip(ps, (512, 256, 128, 64, 3)))
            ]
        )

        res_channels = [64, 256, 512, 1024, 2048]
        #
        channels = [64, 128, 256, 512, 512]

        # layer 1
        self.re1_r = BaseConv2d(res_channels[0], channels[0], kernel_size=1)
        self.re1_d = BaseConv2d(res_channels[0], channels[0], kernel_size=1)

        # layer 2
        self.re2_r = BaseConv2d(res_channels[1], channels[1], kernel_size=1)
        self.re2_d = BaseConv2d(res_channels[1], channels[1], kernel_size=1)

        # layer 3
        self.re3_r = BaseConv2d(res_channels[2], channels[2], kernel_size=1)
        self.re3_d = BaseConv2d(res_channels[2], channels[2], kernel_size=1)
        self.conv1 = BaseConv2d(2 * channels[2], channels[2], kernel_size=1)
        self.SA1 = SpatialAttention(kernel_size=7)

        # layer 4
        self.re4_r = BaseConv2d(res_channels[3], channels[3], kernel_size=1)
        self.re4_d = BaseConv2d(res_channels[3], channels[3], kernel_size=1)
        self.conv2 = BaseConv2d(2 * channels[3], channels[3], kernel_size=1)
        self.SA2 = SpatialAttention(kernel_size=7)

        # layer 5
        self.re5_r = BaseConv2d(res_channels[4], channels[4], kernel_size=1)
        self.re5_d = BaseConv2d(res_channels[4], channels[4], kernel_size=1)
        self.conv3 = BaseConv2d(2 * channels[4], channels[4], kernel_size=1)

        # Intra-Fearture Enhancement Unit
        self.ca_rgb = ChannelAttention(channels[4])
        self.ca_depth = ChannelAttention(channels[4])
        self.ca_rgbd = ChannelAttention(channels[4])

        self.sa_rgb = SpatialAttention(kernel_size=7)
        self.sa_depth = SpatialAttention(kernel_size=7)
        self.sa_rgbd = SpatialAttention(kernel_size=7)

        # Inter-Feature Enhancement Unit
        self.IrFEU = IrFEU(channels[4], squeeze_ratio=1)
        self.decoder = Decoder()

        #deconv
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

        self.conv_r_map = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.conv_d_map = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.predictor = nn.ModuleList()
        self.predictor.append(StackedCBRBlock(embed_dim * 2, embed_dim))
        self.predictor.append(StackedCBRBlock(embed_dim, 32))
        self.predictor.append(nn.Conv2d(32, 1, 1))



    def forward(self, rgb, depth):
        Deconv_rgb_list = []
        Deconv_depth_list = []
        #backbone
        depth = self.layer_dep(depth)
        #RGB
        rgb_feas_1 = self.rgb_block1(rgb)
        rgb_feas_2 = self.rgb_block2(rgb_feas_1)
        rgb_feas_3 = self.rgb_block3(rgb_feas_2)
        rgb_feas_4 = self.rgb_block4(rgb_feas_3)
        rgb_feas_5 = self.rgb_block5(rgb_feas_4)#1 1024 8 8

        rgb_feas1 = self.re1_r(rgb_feas_1)
        rgb_feas2 = self.re2_r(rgb_feas_2)
        rgb_feas3 = self.re3_r(rgb_feas_3)
        rgb_feas4 = self.re4_r(rgb_feas_4)
        rgb_feas5 = self.re5_r(rgb_feas_5) # 1 512 8 8


        #depth
        depth_feas_1 = self.depth_block1(depth)
        depth_feas_2 = self.depth_block2(depth_feas_1)
        depth_feas_3 = self.depth_block3(depth_feas_2)
        depth_feas_4 = self.depth_block4(depth_feas_3)
        depth_feas_5 = self.depth_block5(depth_feas_4)

        depth_feas1 = self.re1_d(depth_feas_1)
        depth_feas2 = self.re2_d(depth_feas_2)
        depth_feas3 = self.re3_d(depth_feas_3)
        depth_feas4 = self.re4_d(depth_feas_4)
        depth_feas5 = self.re5_d(depth_feas_5)  # 1 512 8 8




        #SAIU
        conv3_rgbd = self.conv1(torch.cat([rgb_feas3, depth_feas3], dim=1))
        conv3_rgbd = F.interpolate(conv3_rgbd, scale_factor=0.5, mode='bilinear', align_corners=True)
        conv3_rgbd_map = self.SA1(conv3_rgbd)

        conv4_rgbd = self.conv2(torch.cat([rgb_feas4, depth_feas4], dim=1))
        conv4_rgbd = conv4_rgbd * conv3_rgbd_map + conv4_rgbd
        conv4_rgbd = F.interpolate(conv4_rgbd, scale_factor=0.5, mode='bilinear', align_corners=True)
        conv4_rgbd_map = self.SA2(conv4_rgbd)

        conv5_rgbd = self.conv3(torch.cat([rgb_feas5, depth_feas5], dim=1))
        conv5_rgbd = conv5_rgbd * conv4_rgbd_map + conv5_rgbd

        # IaFEU
        B, C, H, W = rgb_feas5.size()
        P = H * W

        rgb_SA = self.sa_rgb(rgb_feas5).view(B, -1, P)  # (2,1,64)
        depth_SA = self.sa_depth(depth_feas5).view(B, -1, P)
        rgbd_SA = self.sa_rgbd(conv5_rgbd).view(B, -1, P)

        rgb_CA = self.ca_rgb(rgb_feas5).view(B, C, -1)  # (2,512,1)
        depth_CA = self.ca_depth(depth_feas5).view(B, C, -1)
        rgbd_CA = self.ca_rgbd(conv5_rgbd).view(B, C, -1)


        rgb_M = torch.bmm(rgb_CA, rgb_SA).view(B, C, H, W)
        depth_M = torch.bmm(depth_CA, depth_SA).view(B, C, H, W)
        rgbd_M = torch.bmm(rgbd_CA, rgbd_SA).view(B, C, H, W)

        rgb_IaFEU = rgb_feas5 * rgb_M + rgb_feas5
        depth_IaFEU = depth_feas5 * depth_M + depth_feas5
        rgbd_IaFEU = conv5_rgbd * rgbd_M + conv5_rgbd


        #IrFEU
        rgb_IrFEU,depth_IrFEU = self.IrFEU(rgb_IaFEU,depth_IaFEU,rgbd_IaFEU)
        Deconv_rgb_list.extend([rgb_feas1, rgb_feas2, rgb_feas3, rgb_feas4, rgb_feas5, rgb_IrFEU])
        Deconv_depth_list.extend([depth_feas1, depth_feas2, depth_feas3, depth_feas4, depth_feas5, depth_IrFEU])

        #deconv
        deconv_rgb,deconv_depth = self.decoder(Deconv_rgb_list,Deconv_depth_list)

        #cross attention fusion
        rgbd_ca_fusion1 = self.CAM[0](rgb=deconv_rgb[0],depth=deconv_depth[0])
        rgbd_ca_fusion2 = self.CAM[1](rgb=deconv_rgb[1], depth=deconv_depth[1], before_rgbd=cus_sample(rgbd_ca_fusion1, scale_factor=2))
        rgbd_ca_fusion3 = self.CAM[2](rgb=deconv_rgb[2], depth=deconv_depth[2], before_rgbd=cus_sample(rgbd_ca_fusion2, scale_factor=2))
        rgbd_ca_fusion4 = self.CAM[3](rgb=deconv_rgb[3], depth=deconv_depth[3], before_rgbd=cus_sample(rgbd_ca_fusion3, scale_factor=2))

        #prediction

        pre_map1 = self.predictor[0](cus_sample(rgbd_ca_fusion4, scale_factor=2))
        pre_map2 = self.predictor[1](cus_sample(pre_map1, scale_factor=2))
        pre_map = self.predictor[2](pre_map2)

        rgb_block1_up = F.interpolate(deconv_rgb[4], scale_factor=2, mode='bilinear')
        rgb_map = self.conv_r_map(rgb_block1_up)

        depth_block1_up = F.interpolate(deconv_depth[4], scale_factor=2, mode='bilinear')
        depth_map = self.conv_d_map(depth_block1_up)


        return rgb_map,depth_map,pre_map


    


# Test code 

if __name__ == '__main__':
    rgb = torch.rand((2, 3, 256, 256)).cuda()
    depth = torch.rand((2, 1, 256, 256)).cuda()
    model = CANet().cuda()
    # l = model(rgb,depth)
    rgb_map, depth_map, pre_map = model(rgb, depth)


