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
import models.model_blocks as mb
from models.gim_basic_models import GIMMeanStdFcStat

########################################################################################################################
# ID Encoder
########################################################################################################################
class Encoder(nn.Module):
    """"""
    def __init__(self, img_size, img_channels, style_dim=512, min_n_channels=64, use_out_lrelu=True):
        """"""
        super().__init__()
        self.img_size = img_size
        self.img_channels = img_channels
        self.style_dim = style_dim
        self.use_out_lrelu = use_out_lrelu

        # calc channel sizes
        self.n_down_blocks = int(math.log2(img_size)) - 2
        self.min_n_channels = int(max(min_n_channels, style_dim / (2 ** (self.n_down_blocks - 1))))
        self.channel_sizes = [img_channels] + [min(style_dim, int(self.min_n_channels * (2 ** i))) for i in range(self.n_down_blocks)]
        self.att_loc = int(math.ceil(self.n_down_blocks / 2))

        # layers:
        self.lrelu = nn.LeakyReLU(0.2)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.down_blocks = nn.ModuleList()
        for i in range(self.n_down_blocks):
            self.down_blocks.append(mb.ResBlockDown(self.channel_sizes[i], self.channel_sizes[i + 1]))
        self.att = mb.SelfAttention(self.channel_sizes[self.att_loc])

    def forward(self, x):
        """
        :param x: [batch_size, img_channels, img_size, img_size]
        :return: [batch_size, style_dim]
        """
        for i in range(self.n_down_blocks):
            if i == self.att_loc:
                x = self.att(x)
            x = self.down_blocks[i](x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        if self.use_out_lrelu:
            x = self.lrelu(x)  # out 512*1*1
        return x


########################################################################################################################
# Env Decoder
########################################################################################################################
class EnvDecoder(nn.Module):
    def __init__(self, img_size, img_channels, style_dim=512, min_n_channels=64):
        super().__init__()
        self.img_size = img_size
        self.img_channels = img_channels
        self.style_dim = style_dim
        self.min_n_channels = min_n_channels

        # calc channel sizes
        self.n_up_blocks = int(math.log2(img_size))
        self.channel_sizes = list(
            reversed([min(style_dim, int(self.min_n_channels * (2 ** i))) for i in range(self.n_up_blocks)])
        ) + [img_channels]
        self.att_loc = int(math.ceil(self.n_up_blocks / 2))

        # layers:
        self.lrelu = nn.LeakyReLU(0.2)
        self.up_blocks = nn.ModuleList()
        for i in range(self.n_up_blocks):
            self.up_blocks.append(mb.ResBlockUp(self.channel_sizes[i], self.channel_sizes[i + 1]))
        self.att = mb.SelfAttention(self.channel_sizes[self.att_loc])

    def forward(self, x):
        """
        :param x: [batch_size, style_dim]
        :return:
        """
        x = x.view(x.size(0), x.size(1), 1, 1)
        for i in range(self.n_up_blocks):
            if i == self.att_loc:
                x = self.att(x)
            x = self.up_blocks[i](x)
        return x


########################################################################################################################
# Image to Image
########################################################################################################################
class Img2ImgDownModule(nn.Module):
    """"""
    def __init__(self, img_size, img_channels, style_dim=512, min_n_channels=64):
        """"""
        super().__init__()
        self.img_size = img_size
        self.img_channels = img_channels
        self.style_dim = style_dim

        # calc channel sizes
        self.n_down_blocks = int(math.log2(img_size)) - 2
        self.min_n_channels = int(max(min_n_channels, style_dim / (2 ** (self.n_down_blocks - 1))))
        self.channel_sizes = [img_channels] + [min(style_dim, int(self.min_n_channels * (2 ** i))) for i in range(self.n_down_blocks)]
        self.att_loc = int(math.ceil(self.n_down_blocks / 2))

        # layers:
        self.lrelu = nn.LeakyReLU(0.2)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.down_blocks = nn.ModuleList()
        self.in_layers = nn.ModuleList()
        for i in range(self.n_down_blocks):
            if i == 0:
                self.down_blocks.append(mb.ResBlockDown(self.channel_sizes[i], self.channel_sizes[i + 1], conv_size=9, padding_size=4))
            else:
                self.down_blocks.append(mb.ResBlockDown(self.channel_sizes[i], self.channel_sizes[i + 1]))
            self.in_layers.append(nn.InstanceNorm2d(self.channel_sizes[i + 1], affine=True))
        self.att = mb.SelfAttention(self.channel_sizes[self.att_loc])

    def forward(self, x):
        """
        :param x: [batch_size, img_channels, img_size, img_size]
        :return: [batch_size, style_dim]
        """
        for i in range(self.n_down_blocks):
            if i == self.att_loc:
                x = self.att(x)
            x = self.down_blocks[i](x)
            x = self.in_layers[i](x)
        return x


class Img2ImgAdaInResModule(nn.Module):
    """"""
    def __init__(self, style_dim=512, n_blocks=5):
        """"""
        super().__init__()
        self.style_dim = style_dim
        self.n_blocks = n_blocks

        # layers:
        self.res_blocks = nn.ModuleList()
        for i in range(self.n_blocks):
            self.res_blocks.append(mb.AdaResBlock2(channels=style_dim, style_dim=style_dim))

    def forward(self, x, style):
        """
        :param x: [batch_size, img_channels, img_size, img_size]
        :return: [batch_size, style_dim]
        """
        for i in range(self.n_blocks):
            x = self.res_blocks[i](x=x, style=style)
        return x


class Img2ImgAdaInUpModule(nn.Module):
    """"""
    def __init__(self, img_size, img_channels, style_dim=512, min_n_channels=64):
        """"""
        super().__init__()
        self.img_size = img_size
        self.img_channels = img_channels
        self.style_dim = style_dim

        # calc channel sizes
        self.n_up_blocks = int(math.log2(img_size)) - 2
        self.min_n_channels = int(max(min_n_channels, style_dim / (2 ** (self.n_up_blocks - 1))))
        self.channel_sizes = list(
            reversed([min(style_dim, int(self.min_n_channels * (2 ** i))) for i in range(self.n_up_blocks)])
        ) + [img_channels]
        self.att_loc = int(math.ceil(self.n_up_blocks / 2))

        # layers:
        self.up_blocks = nn.ModuleList()
        for i in range(self.n_up_blocks):
            if i == (self.n_up_blocks - 1):
                self.up_blocks.append(
                    mb.AdaResBlockUp2(
                        in_channels=self.channel_sizes[i],
                        out_channels=self.channel_sizes[i + 1],
                        style_dim=style_dim,
                        conv_size=9,
                        padding_size=4
                    )
                )
            else:
                self.up_blocks.append(
                    mb.AdaResBlockUp2(
                        in_channels=self.channel_sizes[i],
                        out_channels=self.channel_sizes[i + 1],
                        style_dim=style_dim
                    )
                )
        self.att = mb.SelfAttention(self.channel_sizes[self.att_loc])

    def forward(self, x, style):
        """
        :param x:
        :param styles:
        :return:
        """
        for i in range(self.n_up_blocks):
            if i == self.att_loc:
                x = self.att(x)
            x = self.up_blocks[i](x=x, style=style)
        return F.tanh(x)


class AdaInImage2Image(nn.Module):
    """"""
    def __init__(self, img_size, in_channels, out_channels, style_dim, n_adain_res_blocks=5, min_n_channels=64):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.style_dim = style_dim
        self.n_adain_res_blocks = n_adain_res_blocks
        self.min_n_channels = min_n_channels

        # activations
        self.lrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

        # Down
        self.down_block = Img2ImgDownModule(
            img_size=img_size,
            img_channels=in_channels,
            style_dim=style_dim,
            min_n_channels=min_n_channels
        )
        # AdaIn Res
        self.adain_res_block = Img2ImgAdaInResModule(style_dim=style_dim, n_blocks=n_adain_res_blocks)

        # AdaIn Up
        self.adain_up_block = Img2ImgAdaInUpModule(
            img_size=img_size, img_channels=out_channels, style_dim=style_dim, min_n_channels=min_n_channels
        )

    def forward(self, x, style):
        """
        :param x: [batch_size, in_channels, img_size, img_size]
        :param style: [batch_size, style_dim]
        :return:
        """
        x = self.down_block(x)
        x = self.adain_res_block(x=x, style=style)
        x = self.adain_up_block(x=x, style=style)
        return x # [batch_size, img_channels, img_size, img_size]


########################################################################################################################
# GIM Discriminators
########################################################################################################################
class GIMFaceDis(nn.Module):
    """"""
    def __init__(self, src_dim, env_dim, stat):
        """"""
        super().__init__()
        self.src_dim = src_dim
        self.env_dim = env_dim

        self.stat = stat
        self.n_stats = stat.n_stats
        mlp_input_dim = 2 * (self.n_stats * env_dim + src_dim)
        self.mlp = mb.MLP((mlp_input_dim, env_dim + src_dim, 2 * (env_dim + src_dim), 1))

        # initialization
        self.mlp.apply(mb.weights_init('kaiming'))

    def forward(self, test_src, test_env, si_src, si_env):
        """

        :param test_src: [batch_size, n, src_dim]
        :param test_env: [batch_size, n, env_dim]
        :param si_src: [batch_size, k, src_dim]
        :param si_env: [batch_size, k, env_dim]
        :return: [batch, 1]
        """
        # comp src stats
        test_src_mean = test_src.mean(1) # [batch_size, src_dim]
        si_src_mean = si_src.mean(1) # [batch_size, src_dim]

        # comp env stats
        test_env_stat = self.stat(test_env) # [batch_size, env_dim * n_stats]
        si_env_stat = self.stat(si_env) # [batch_size, env_dim * n_stats]

        # output
        x = torch.cat((test_src_mean, si_src_mean, test_env_stat, si_env_stat), dim=-1)  # [batch_size, 2 * (src_dim + env_dim * n_stats)]
        x = self.mlp(x) # [batch, 1]
        return x

########################################################################################################################
# Authenticator
########################################################################################################################
class GIMFaceAuthenticator(nn.Module):
    """"""
    def __init__(self, src_encoder, env_encoder, dis):
        """"""
        super().__init__()
        self.src_encoder = src_encoder
        self.env_encoder = env_encoder
        self.dis = dis

    def forward(self, test_sample, si_sample):
        """"""
        test_src = self.src_encode_sample(test_sample)
        si_src = self.src_encode_sample(si_sample)

        test_env = self.env_encode_sample(test_sample)
        si_env = self.env_encode_sample(si_sample)
        x = self.dis(
            test_src=test_src,
            test_env=test_env,
            si_src=si_src,
            si_env=si_env
        )
        return x

    def src_encode_sample(self, sample):
        batch_size = sample.size(0)
        sample_size = sample.size(1)
        x = self.src_encoder(sample.view(batch_size*sample_size,*sample.size()[2:]))
        x = x.view(batch_size, sample_size, *x.size()[1:])
        return x

    def env_encode_sample(self, sample):
        batch_size = sample.size(0)
        sample_size = sample.size(1)
        x = self.env_encoder(sample.view(batch_size*sample_size,*sample.size()[2:]))
        x = x.view(batch_size, sample_size, *x.size()[1:])
        return x


########################################################################################################################
# Impersonator
########################################################################################################################
class GIMFaceImpersonator(nn.Module):
    """"""
    def __init__(self, src_encoder, env_encoder, env_decoder, img2img, env_noise_mapper, use_img_att=False):
        """"""
        super().__init__()
        self.src_encoder = src_encoder
        self.env_encoder = env_encoder
        self.env_decoder = env_decoder
        self.img2img = img2img
        self.env_noise_mapper = env_noise_mapper
        self.style_dim = src_encoder.style_dim
        assert src_encoder.style_dim == env_encoder.style_dim == env_decoder.style_dim == img2img.style_dim

        self.use_img_att = use_img_att
        self.img_att = mb.ImgAttention(
            img1_channels=self.src_encoder.img_channels, img2_channels=self.img2img.out_channels
        )

    def forward(self, leaked_sample, n, remove_noise_mean=True):
        """"""
        batch_size, m, img_channels, img_size, _ = leaked_sample.size()
        expanded_img = leaked_sample[:, 0].unsqueeze(1).expand(-1, n, -1, -1, -1)

        # src
        src = self.src_encode_sample(leaked_sample).mean(1)
        env = self.env_encode_sample(leaked_sample).mean(1)

        # env noise
        z = torch.randn((batch_size, n, self.style_dim), device=leaked_sample.device)
        w = self.env_noise_mapper(z)

        # remove_mean
        if remove_noise_mean:
            w = w - w.mean(1, keepdim=True)
        noisy_env = env.unsqueeze(1) + w

        # env img
        env_img = self.env_decoder(noisy_env.view(batch_size * n, *noisy_env.size()[2:]))
        env_img = env_img.view(batch_size, n, *env_img.size()[1:])
        env_img = torch.cat((env_img, expanded_img), dim=2)

        # generate sample
        x = self.generate_img(env_img=env_img, src=src)

        # img attention
        if self.use_img_att:
            x = self.img_att(
                x1=expanded_img.reshape(batch_size*n, *expanded_img.size()[2:]),
                x2=x.view(batch_size*n, *x.size()[2:])
            )
            x = x.view(batch_size, n, *x.size()[1:])
        return x

    def src_encode_sample(self, sample):
        batch_size = sample.size(0)
        sample_size = sample.size(1)
        x = self.src_encoder(sample.view(batch_size*sample_size,*sample.size()[2:]))
        x = x.view(batch_size, sample_size, *x.size()[1:])
        return x

    def env_encode_sample(self, sample):
        batch_size = sample.size(0)
        sample_size = sample.size(1)
        x = self.env_encoder(sample.view(batch_size*sample_size,*sample.size()[2:]))
        x = x.view(batch_size, sample_size, *x.size()[1:])
        return x

    def generate_img(self, env_img, src):
        batch_size = env_img.size(0)
        n = env_img.size(1)

        gen_img = self.img2img(
            x=env_img.contiguous().view(batch_size*n, *env_img.size()[2:]),
            style=src.unsqueeze(1).expand(-1, n, -1).contiguous().view(batch_size*n, self.style_dim)
        )

        gen_img = gen_img.view(batch_size, n, *gen_img.size()[1:])
        return gen_img


########################################################################################################################
# Model Getters
########################################################################################################################
def get_im(img_size, img_channels, style_dim, use_img_att=False, num_env_noise_layers=4):
    """"""
    src_encoder = Encoder(img_size=img_size, img_channels=img_channels, style_dim=style_dim)
    env_encoder = Encoder(img_size=img_size, img_channels=img_channels, style_dim=style_dim)
    decoder = EnvDecoder(img_size=img_size, img_channels=img_channels, style_dim=style_dim)
    img2img = AdaInImage2Image(
        img_size=img_size,
        in_channels=2 * img_channels,
        out_channels=img_channels,
        style_dim=style_dim
    )
    env_noise_mapper = mb.MLP([style_dim for _ in range(num_env_noise_layers + 1)])
    im = GIMFaceImpersonator(
        src_encoder=src_encoder,
        env_encoder=env_encoder,
        env_decoder=decoder,
        img2img=img2img,
        env_noise_mapper=env_noise_mapper,
        use_img_att=use_img_att
    )
    return im


def get_au(img_size, img_channels, style_dim):
    """"""
    stat = GIMMeanStdFcStat(style_dim=style_dim, fc_n_stats=2, fc_hidden_layers=(style_dim * 2, style_dim * 3, style_dim * 2))
    dis = GIMFaceDis(src_dim=style_dim, env_dim=style_dim, stat=stat)
    src_encoder = Encoder(img_size=img_size, img_channels=img_channels, style_dim=style_dim)
    env_encoder = Encoder(img_size=img_size, img_channels=img_channels, style_dim=style_dim)
    au = GIMFaceAuthenticator(
        src_encoder=src_encoder,
        env_encoder=env_encoder,
        dis=dis
    )
    return au


########################################################################################################################
# UNIT TEST
########################################################################################################################
if __name__ == '__main__':
    def test_im():
        batch_size = 4
        img_size = 64
        img_channels = 3
        style_dim = 512
        use_img_att=False
        m = 1
        n = 10
        im = get_im(img_size=img_size, img_channels=img_channels, style_dim=style_dim, use_img_att=use_img_att)
        leaked_sample = torch.randn((batch_size, m, img_channels, img_size, img_size))
        fake_sample = im(leaked_sample, n, True)
        print(fake_sample.size())


    def test_au():
        batch_size = 4
        img_size = 64
        img_channels = 3
        style_dim = 512
        n = 5
        k = 3
        au = get_au(img_size=img_size, img_channels=img_channels, style_dim=style_dim)
        test_sample = torch.randn((batch_size, n, img_channels, img_size, img_size))
        si_sample = torch.randn((batch_size, k, img_channels, img_size, img_size))
        out = au(test_sample, si_sample)
        print(out.size())


    test_im()
    test_au()