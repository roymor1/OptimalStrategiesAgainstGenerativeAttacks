# imports
import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(project_root)


########################################################################################################################
# Utility Functions
########################################################################################################################
def weights_init(init_type='kaiming'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0.2)
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    return init_fun


def custom_std(x):
    batch_size = x.size(0)
    sample_size = x.size(1)
    if sample_size > 1:
        s = torch.sqrt(x.var(1) + 1e-8)
    else:
        s = torch.zeros((batch_size, *x.size()[2:]), device=x.device)
    return s


########################################################################################################################
# Custom Utility Layers & Modules
########################################################################################################################
class Flatten(nn.Module):
    """"""
    def __init__(self):
        """"""
        super().__init__()

    def forward(self, x):
        """"""
        batch_size = x.size(0)
        return x.view(batch_size, -1)


class Identity(nn.Module):
    """"""
    def __init__(self):
        """"""
        super().__init__()

    def forward(self, x):
        """"""
        return x


class MLP(nn.Module):
    """"""
    def __init__(self, layer_dims):
        """"""
        super().__init__()
        assert len(layer_dims) >= 2
        layers = []
        inp_dim = layer_dims[0]
        for out_dim in layer_dims[1:-1]:
            layers.append(nn.Linear(inp_dim, out_dim))
            layers.append(nn.LeakyReLU(0.2))
            inp_dim = out_dim
        layers.append(nn.Linear(inp_dim, layer_dims[-1]))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """"""
        return self.model(x)


class ResMLP(nn.Module):
    """"""
    def __init__(self, layer_dims):
        """"""
        super().__init__()
        assert len(layer_dims) >= 2
        layers = []
        inp_dim = layer_dims[0]
        for out_dim in layer_dims[1:-1]:
            layers.append(nn.Linear(inp_dim, out_dim))
            layers.append(nn.LeakyReLU(0.2))
            inp_dim = out_dim
        layers.append(nn.Linear(inp_dim, layer_dims[-1]))
        self.model = nn.Sequential(*layers)
        self.linear = nn.Linear(layer_dims[0], layer_dims[-1])
        self.out_linear = nn.Linear(2 * layer_dims[-1], layer_dims[-1])

    def forward(self, x):
        """"""
        x1 = self.linear(x)
        x2 = self.model(x)
        x = torch.cat((x1, x2), dim=-1)
        return self.out_linear(x)

    def init_to_replay(self, style_dim):
        """"""
        # out linear
        nn.init.normal_(self.out_linear.weight, 0.0, 0.0001)
        # nn.init.constant_(self.out_linear.weight.data, 0.)
        nn.init.constant_(self.out_linear.bias.data, 0.)
        for i in range(style_dim):
            self.out_linear.weight.data[i, i] = 1.

        # linear
        # nn.init.constant_(self.linear.weight.data, 0.)
        nn.init.normal_(self.linear.weight, 0.0, 0.0001)
        nn.init.constant_(self.linear.bias.data, 0.)
        for i in range(style_dim):
            self.linear.weight.data[i, i] = 1.

        # mlp
        self.model.apply(weights_init('kaiming'))


class ResMLP2(nn.Module):
    """"""
    def __init__(self, layer_dims):
        """"""
        super().__init__()
        assert len(layer_dims) >= 2
        layers = []
        inp_dim = layer_dims[0]
        for out_dim in layer_dims[1:-1]:
            layers.append(nn.Linear(inp_dim, out_dim))
            layers.append(nn.LeakyReLU(0.2))
            inp_dim = out_dim
        layers.append(nn.Linear(inp_dim, layer_dims[-1]))
        self.model = nn.Sequential(*layers)
        self.linear = nn.Linear(inp_dim + layer_dims[-1], layer_dims[-1])

    def forward(self, x):
        """"""
        x = torch.cat((x, self.model(x)), dim=-1)
        return self.linear(x)

    def init_to_replay(self, style_dim):
        """"""
        # out linear
        nn.init.normal_(self.out_linear.weight, 0.0, 0.0001)
        # nn.init.constant_(self.out_linear.weight.data, 0.)
        nn.init.constant_(self.out_linear.bias.data, 0.)
        for i in range(style_dim):
            self.linear.weight.data[i, i] = 1.
        # mlp
        self.model.apply(weights_init('kaiming'))


########################################################################################################################
# Custom SG Layers
########################################################################################################################
def upscale2d(x, factor=2, gain=1):
    assert x.dim() == 4
    if gain != 1:
        x = x * gain
    if factor != 1:
        shape = x.shape
        x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, factor, -1, factor)
        x = x.contiguous().view(shape[0], shape[1], factor * shape[2], factor * shape[3])
    return x


class Upscale2d(nn.Module):
    def __init__(self, factor=2, gain=1):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.gain = gain
        self.factor = factor
    def forward(self, x):
        return upscale2d(x, factor=self.factor, gain=self.gain)


class NoiseLayer(nn.Module):
    """adds noise. noise is per pixel (constant over channels) with per-channel weight"""

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))
        self.noise = None

    def forward(self, x, noise=None):
        if noise is None and self.noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        elif noise is None:
            # here is a little trick: if you get all the noiselayers and set each
            # modules .noise attribute, you can have pre-defined noise.
            # Very useful for analysis
            noise = self.noise
        x = x + self.weight.view(1, -1, 1, 1) * noise
        return x


class PixelNormLayer(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + self.eps)


class BlurLayer(nn.Module):
    def __init__(self, kernel=(1, 2, 1), normalize=True, flip=False, stride=1):
        super(BlurLayer, self).__init__()
        kernel = [1, 2, 1]
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel[None, None]
        if normalize:
            kernel = kernel / kernel.sum()
        if flip:
            kernel = kernel[:, :, ::-1, ::-1]
        self.register_buffer('kernel', kernel)
        self.stride = stride

    def forward(self, x):
        # expand kernel channels
        kernel = self.kernel.expand(x.size(1), -1, -1, -1)
        x = F.conv2d(
            x,
            kernel,
            stride=self.stride,
            padding=int((self.kernel.size(2) - 1) / 2),
            groups=x.size(1)
        )
        return x


class StyleMod(nn.Module):
    def __init__(self, style_dim, channels):
        super().__init__()
        self.lin = nn.Linear(style_dim, channels * 2)

    def forward(self, x, style):
        style = self.lin(style)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1)] + (x.dim() - 2) * [1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x


class StyleEstimator(nn.Module):
    def __init__(self, style_dim, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 2 * style_dim, kernel_size=1, padding=0)
        self.lrelu = nn.LeakyReLU(0.2)
        self.lin = nn.Linear(2 * style_dim, style_dim)

    def forward(self, x):
        batch_size, img_channels, img_size, _ = x.size()
        x = self.conv(x)
        x = self.lrelu(x)
        x = F.avg_pool2d(x, img_size).view(batch_size, -1)
        return self.lin(x)


########################################################################################################################
# Custom SG Model Blocks
########################################################################################################################
class SGLayerEpilogue(nn.Module):
    """Things to do at the end of each layer."""

    def __init__(self,
                 channels, style_dim, activation_layer,
                 use_pixel_norm, use_instance_norm, use_noise):
        super().__init__()
        self.noise = NoiseLayer(channels) if use_noise else Identity()
        self.activation = activation_layer
        self.pixel_norm = PixelNormLayer() if use_pixel_norm else Identity()
        self.instance_norm = nn.InstanceNorm2d(channels) if use_instance_norm else Identity()
        self.style_mod = StyleMod(style_dim, channels)

    def forward(self, x, style):
        x = self.noise(x)
        x = self.activation(x)
        x = self.pixel_norm(x)
        x = self.instance_norm(x)
        x = self.style_mod(x, style)
        return x


class SGInputBlock(nn.Module):
    def __init__(self,
                 channels, style_dim, activation_layer,
                 use_pixel_norm, use_instance_norm, use_noise):
        super().__init__()
        self.channels = channels
        self.epi1 = SGLayerEpilogue(
            channels=self.channels,
            style_dim=style_dim,
            activation_layer=activation_layer,
            use_pixel_norm=use_pixel_norm,
            use_instance_norm=use_instance_norm,
            use_noise=use_noise
        )
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.epi2 = SGLayerEpilogue(
            channels=self.channels,
            style_dim=style_dim,
            activation_layer=activation_layer,
            use_pixel_norm=use_pixel_norm,
            use_instance_norm=use_instance_norm,
            use_noise=use_noise
        )

    def forward(self, x, style1, style2):
        """"""
        x = self.epi1(x, style1)
        x = self.conv(x)
        x = self.epi2(x, style2)
        return x


class SGConstInputBlock(nn.Module):
    def __init__(self,
                 channels, style_dim, init_img_size, activation_layer,
                 use_pixel_norm, use_instance_norm, use_noise):
        super().__init__()
        self.channels = channels
        self.init_img = nn.Parameter(torch.ones(1, self.channels, init_img_size, init_img_size))
        self.bias = nn.Parameter(torch.ones(self.channels))
        self.model = SGInputBlock(
            channels=self.channels, style_dim=style_dim, activation_layer=activation_layer,
            use_pixel_norm=use_pixel_norm, use_instance_norm=use_instance_norm, use_noise=use_noise
        )

    def forward(self, style1, style2):
        batch_size = style1.size(0)
        x = self.init_img.expand(batch_size, -1, -1, -1)
        x = x + self.bias.view(1, -1, 1, 1)
        x = self.model(x, style1, style2)
        return x


class SGToImgBlock(nn.Module):
    """"""
    def __init__(self, in_channels, img_channels, init_type='kaiming'):
        """"""
        super().__init__()
        self.model = nn.Conv2d(in_channels, img_channels, kernel_size=1, padding=0)
        self.apply(weights_init(init_type))

    def forward(self, x):
        """"""
        return self.model(x)


class SGFromImgBlock(nn.Module):
    """"""
    def __init__(self, in_channels, out_channels, init_type='kaiming'):
        """"""
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.LeakyReLU(0.2),
        )
        self.apply(weights_init(init_type))

    def forward(self, x):
        """"""
        return self.model(x)


class SGDecoderBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels, style_dim,
                 activation_layer, use_pixel_norm, use_instance_norm, use_noise,
                 init_type='kaiming'):
        """"""
        super().__init__()
        self.upscale2d = Upscale2d()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.blur = BlurLayer()
        self.epi1 = SGLayerEpilogue(
            channels=out_channels,
            style_dim=style_dim,
            activation_layer=activation_layer,
            use_pixel_norm=use_pixel_norm,
            use_instance_norm=use_instance_norm,
            use_noise=use_noise

        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.epi2 = SGLayerEpilogue(
            channels=out_channels,
            style_dim=style_dim,
            activation_layer=activation_layer,
            use_pixel_norm=use_pixel_norm,
            use_instance_norm=use_instance_norm,
            use_noise=use_noise
        )
        # initialize weights
        self.apply(weights_init(init_type))

    def forward(self, x, style1, style2):
        """"""
        x = self.upscale2d(x)
        x = self.conv1(x)
        x = self.blur(x)
        x = self.epi1(x, style1)
        x = self.conv2(x)
        x = self.epi2(x, style2)
        return x


class SGEncoderBlock(nn.Module):
    """"""
    def __init__(self, in_channels, out_channels1, out_channels2, style_dim, pool=True, init_type='kaiming'):
        """"""
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels1, kernel_size=3, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.style_est1 =  StyleEstimator(style_dim=style_dim, channels=out_channels1)

        self.conv2 = nn.Conv2d(out_channels1, out_channels2, kernel_size=3, padding=1)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.style_est2 = StyleEstimator(style_dim=style_dim, channels=out_channels2)
        self.pool = nn.AvgPool2d(2) if pool else Identity()
        self.apply(weights_init(init_type))

    def forward(self, x):
        """"""
        x = self.conv1(x)
        x = self.lrelu1(x)
        style1 = self.style_est1(x)

        x = self.conv2(x)
        x = self.lrelu2(x)
        style2 = self.style_est2(x)

        x = self.pool(x)
        return x, style1, style2


class SGDisBlock(nn.Module):
    """"""
    def __init__(self, in_channels, out_channels1, out_channels2, pool=True, init_type='kaiming'):
        """"""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels2
        self.conv1 = nn.Conv2d(in_channels, out_channels1, kernel_size=3, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(out_channels1, out_channels2, kernel_size=3, padding=1)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.pool = nn.AvgPool2d(2) if pool else Identity()
        self.apply(weights_init(init_type))

    def forward(self, x):
        """"""
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)
        x = self.pool(x)
        return x

########################################################################################################################
# GIM Face Models Blocks
########################################################################################################################
class ResBlockDown(nn.Module):
    def __init__(self, in_channel, out_channel, conv_size=3, padding_size=1):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2)
        self.avg_pool2d = nn.AvgPool2d(2)
        # left
        self.conv_l1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1, ))
        # right
        self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, conv_size, padding=padding_size))
        self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, conv_size, padding=padding_size))

    def forward(self, x):
        res = x

        # left
        out_res = self.conv_l1(res)
        out_res = self.avg_pool2d(out_res)

        # right
        out = self.lrelu(x)
        out = self.conv_r1(out)
        out = self.lrelu(out)
        out = self.conv_r2(out)
        out = self.avg_pool2d(out)

        # merge
        out = out_res + out

        return out


class SelfAttention(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        # conv f
        self.conv_f = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel // 8, 1))
        # conv_g
        self.conv_g = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel // 8, 1))
        # conv_h
        self.conv_h = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 1))

        self.softmax = nn.Softmax(-2)  # sum in column j = 1
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        f_projection = self.conv_f(x)  # BxC'xHxW, C'=C//8
        g_projection = self.conv_g(x)  # BxC'xHxW
        h_projection = self.conv_h(x)  # BxCxHxW

        f_projection = torch.transpose(f_projection.view(B, -1, H * W), 1, 2)  # BxNxC', N=H*W
        g_projection = g_projection.view(B, -1, H * W)  # BxC'xN
        h_projection = h_projection.view(B, -1, H * W)  # BxCxN

        attention_map = torch.bmm(f_projection, g_projection)  # BxNxN
        attention_map = self.softmax(attention_map)  # sum_i_N (A i,j) = 1

        # sum_i_N (A i,j) = 1 hence oj = (HxAj) is a weighted sum of input columns
        out = torch.bmm(h_projection, attention_map)  # BxCxN
        out = out.view(B, C, H, W)

        out = self.gamma * out + x
        return out

class ImgAttConvBlock(nn.Module):
    """"""
    def __init__(self, in_channels, out_channels):
        """"""
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2)
        # left
        self.conv_l1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 1, ))
        # right
        self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 9, padding=4))
        self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(out_channels, out_channels, 3, padding=1))

    def forward(self, x):
        res = x

        # left
        out_res = self.conv_l1(res)

        # right
        out = self.lrelu(x)
        out = self.conv_r1(out)
        out = self.lrelu(out)
        out = self.conv_r2(out)

        # merge
        out = out_res + out

        return out


class ImgAttention(nn.Module):
    """"""
    def __init__(self, img1_channels, img2_channels):
        """"""
        super().__init__()
        self.q1conv = ImgAttConvBlock(img1_channels + img2_channels, img1_channels)
        self.q2conv = ImgAttConvBlock(img1_channels + img2_channels, img1_channels)
        self.k1conv = ImgAttConvBlock(img1_channels, img1_channels)
        self.k2conv = ImgAttConvBlock(img2_channels, img1_channels)
        self.v2conv = ImgAttConvBlock(img2_channels, img1_channels)

    def forward(self, x1, x2):
        """"""
        x = torch.cat((x1, x2), dim=1)
        q1 = self.q1conv(x)
        q2 = self.q2conv(x)
        k1 = self.k1conv(x1)
        k2 = self.k2conv(x2)
        v2 = self.v2conv(x2)

        scores1 =  torch.mul(q1, k1).sum(1) # [b, h, w]
        scores2 =  torch.mul(q2, k2).sum(1) # [b, h, w]
        scores = torch.stack((scores1, scores2), dim=1)
        attention = F.softmax(scores, dim=1) # [b, 2, h, w]

        out1 = torch.mul(x1, attention[:, 0].unsqueeze(1))
        out2 = torch.mul(v2, attention[:, 1].unsqueeze(1))
        return out1 + out2


def ada_in(feature, mean_style, std_style, eps=1e-5):
    """

    :param feature: [batch_size, channels, img_size, img_size]
    :param mean_style: [batch_size, channels, 1]
    :param std_style: [batch_size, channels, 1]
    :param eps:
    :return:
    """
    B, C, H, W = feature.shape

    feature = feature.view(B, C, -1)

    std_feat = (torch.std(feature, dim=2) + eps).view(B, C, 1)
    mean_feat = torch.mean(feature, dim=2).view(B, C, 1)

    adain = std_style * (feature - mean_feat) / std_feat + mean_style

    adain = adain.view(B, C, H, W)
    return adain


class AdaResBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        # using no ReLU method

        # general
        self.lrelu = nn.LeakyReLU(0.2)

        # left
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 3, padding=1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 3, padding=1))

    def forward(self, x, style1, style2):
        """

        :param x: [batch_size, img_channels, img_size, img_size]
        :param style1: [batch_size, 2 * img_channels]
        :param style2: [batch_size, 2 * img_channels]
        :return:
        """
        res = x
        x = self.conv1(x)
        x = ada_in(feature=x, mean_style=style1[:, :x.size(1)], std_style=style1[:, x.size(1):])
        x = self.lrelu(x)
        x = self.conv2(x)
        x = ada_in(feature=x, mean_style=style2[:, :x.size(1)], std_style=style2[:,x.size(1):])
        x = x + res

        return x


class ResBlockD(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        # using no ReLU method

        # general
        self.lrelu = nn.LeakyReLU(0.2)

        # left
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 3, padding=1))
        self.in1 = nn.InstanceNorm2d(in_channel, affine=True)
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 3, padding=1))
        self.in2 = nn.InstanceNorm2d(in_channel, affine=True)

    def forward(self, x):
        res = x

        out = self.conv1(x)
        out = self.in1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.in2(out)

        out = out + res

        return out


class AdaResBlockUp(nn.Module):
    def __init__(self, in_channel, out_channel, out_size=None, scale=2, conv_size=3, padding_size=1):
        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.upsample = nn.Upsample(size=out_size, scale_factor=scale)
        self.lrelu = nn.LeakyReLU(0.2)

        # left
        self.conv_l1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1))

        # right
        self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, conv_size, padding=padding_size))
        self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, conv_size, padding=padding_size))

    def forward(self, x, style1, style2):
        """"""
        res = x

        # left
        out_res = self.upsample(res)
        out_res = self.conv_l1(out_res)

        # right
        out = ada_in(feature=x, mean_style=style1[:, :x.size(1)], std_style=style1[:, x.size(1):])
        out = self.lrelu(out)
        out = self.upsample(out)
        out = self.conv_r1(out)
        out = ada_in(feature=out, mean_style=style2[:, :out.size(1)], std_style=style2[:, out.size(1):])
        out = self.lrelu(out)
        out = self.conv_r2(out)

        out = out + out_res

        return out


class ResBlockUp(nn.Module):
    def __init__(self, in_channel, out_channel, out_size=None, scale=2, conv_size=3, padding_size=1, use_norm=True):
        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.upsample = nn.Upsample(size=out_size, scale_factor=scale)
        self.lrelu = nn.LeakyReLU(0.2)

        # left
        self.conv_l1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1))

        # right
        self.in1 = nn.InstanceNorm2d(in_channel, affine=True)
        self.in2 = nn.InstanceNorm2d(out_channel, affine=True)
        self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, conv_size, padding=padding_size))
        self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, conv_size, padding=padding_size))

    def forward(self, x):
        """"""
        batch_size, channels, img_size, _ = x.size()

        res = x

        # left
        out_res = self.upsample(res)
        out_res = self.conv_l1(out_res)

        # right
        out = self.in1(x)
        out = self.lrelu(out)
        out = self.upsample(out)
        out = self.conv_r1(out)
        out = self.in2(out)
        out = self.lrelu(out)
        out = self.conv_r2(out)

        out = out + out_res

        return out


class AdaResBlock2(nn.Module):
    def __init__(self, channels, style_dim):
        super().__init__()
        self.style_dim = style_dim
        self.channels = channels

        # general
        self.lrelu = nn.LeakyReLU(0.2)

        # linear for style
        self.lin1_mean = nn.Linear(style_dim, channels)
        self.lin1_std = nn.Linear(style_dim, channels)
        self.lin2_mean = nn.Linear(style_dim, channels)
        self.lin2_std = nn.Linear(style_dim, channels)

        # left
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(channels, channels, 3, padding=1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x, style):
        """
        :param x: [batch_size, channels, img_size, img_size]
        :param style1: [batch_size, style_dim]
        :param style2: [batch_size, style_dim]
        :return:
        """
        res = x
        mean_st1 = self.lin1_mean(style).unsqueeze(2)
        std_st1 = self.lin1_std(style).unsqueeze(2)
        mean_st2 = self.lin2_mean(style).unsqueeze(2)
        std_st2 = self.lin2_std(style).unsqueeze(2)

        x = self.conv1(x)
        x = ada_in(feature=x,mean_style=mean_st1, std_style=std_st1)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = ada_in(feature=x,mean_style=mean_st2, std_style=std_st2)
        x = x + res
        return x


class AdaResBlockUp2(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim, out_size=None, scale=2, conv_size=3, padding_size=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.style_dim = style_dim

        # general
        self.upsample = nn.Upsample(size=out_size, scale_factor=scale)
        self.lrelu = nn.LeakyReLU(0.2)

        # linear for style
        self.lin1_mean = nn.Linear(style_dim, in_channels)
        self.lin1_std = nn.Linear(style_dim, in_channels)
        self.lin2_mean = nn.Linear(style_dim, out_channels)
        self.lin2_std = nn.Linear(style_dim, out_channels)

        # left
        self.conv_l1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 1))

        # right
        self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, conv_size, padding=padding_size))
        self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(out_channels, out_channels, conv_size, padding=padding_size))

    def forward(self, x, style):
        """"""
        res = x
        mean_st1 = self.lin1_mean(style).unsqueeze(2)
        std_st1 = self.lin1_std(style).unsqueeze(2)
        mean_st2 = self.lin2_mean(style).unsqueeze(2)
        std_st2 = self.lin2_std(style).unsqueeze(2)

        # left
        out_res = self.upsample(res)
        out_res = self.conv_l1(out_res)

        # right
        out = ada_in(feature=x, mean_style=mean_st1, std_style=std_st1)
        out = self.lrelu(out)
        out = self.upsample(out)
        out = self.conv_r1(out)
        out = ada_in(feature=out, mean_style=mean_st2, std_style=std_st2)
        out = self.lrelu(out)
        out = self.conv_r2(out)

        out = out + out_res

        return out


########################################################################################################################
# UNIT TEST
########################################################################################################################
if __name__ == '__main__':
    def test_ada_in():
        x = torch.randn((2, 3, 32, 32))
        e1 =  torch.randn((2, 3, 1))
        e2 =  torch.randn((2, 3, 1))
        y = ada_in(x, e1, e2, eps=1e-5)
        print(y.size())

    def test_res_block_up():
        x = torch.randn((2, 512))
        x = x.view(x.size(0), x.size(1), 1, 1)
        model = ResBlockUp(in_channel=512, out_channel=256)
        y = model(x)
        print(y.size())

    def test_ada_res_block_up():
        x = torch.randn((2, 512, 1, 1))
        style1 =  torch.randn((2, 1024, 1))
        style2 =  torch.randn((2, 512, 1))
        model = AdaResBlockUp(in_channel=512, out_channel=256)
        y = model(x=x, style1=style1, style2=style2)
        print(y.size())

    def test_mlp():
        mlp = MLP([512 for _ in range(5)] + [4])
        x = torch.torch.randn((2, 5, 512))
        y = mlp(x)
        print(y.size())

    def test_img_att_conv_block():
        img_size = 64
        in_channels = 6
        out_channels = 3
        x = torch.randn((2, in_channels, img_size, img_size))
        model = ImgAttConvBlock(in_channels, out_channels)
        y = model(x)
        print(y.size())

    def test_img_attention():
        img_size = 64
        channels1 = 3
        channels2 = 512
        x1 = torch.randn((2, channels1, img_size, img_size))
        x2 = torch.randn((2, channels2, img_size, img_size))
        model = ImgAttention(channels1, channels2)
        y = model(x1, x2)
        print(y.size())

    # test_ada_in()
    # test_res_block_up()
    # test_ada_res_block_up()
    # test_mlp()
    # test_img_att_conv_block()
    test_img_attention()