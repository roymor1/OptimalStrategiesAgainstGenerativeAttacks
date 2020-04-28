# imports
import os
import sys
import torch
import torch.nn as nn


# local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(project_root)
import models.model_blocks as mb
from models.gim_basic_models import GIMMeanStdFcStat, GIMMeanStdStat

########################################################################################################################
# GIM Discriminators
########################################################################################################################
class GIMGaussianDis(nn.Module):
    """"""
    def __init__(self, src_dim, stat):
        """"""
        super().__init__()
        self.src_dim = src_dim
        self.stat = stat
        self.n_stats = stat.n_stats
        self.mlp = mb.MLP((self.n_stats * src_dim * 2, src_dim, 2 * src_dim, 1))

        # initialization
        self.mlp.apply(mb.weights_init('kaiming'))

    def forward(self, test_sample, si_sample):
        """

        :param test_sample: [batch_size, n, src_dim]
        :param si_sample: [batch_size, k, src_dim]
        :return: [batch_size, 1]
        """
        test_stat = self.stat(test_sample) # [batch_size, src_dim * n_stats]
        si_stat = self.stat(si_sample) # [batch_size, src_dim * n_stats]
        x = torch.cat((test_stat, si_stat), dim=-1)  # [batch_size, 2 * src_dim * n_stats]
        x = self.mlp(x) # [batch_size, 1]
        return x


########################################################################################################################
# Authenticator
########################################################################################################################
class GIMGaussianAuthenticator(nn.Module):
    """"""
    def __init__(self, dis):
        """"""
        super().__init__()
        self.dis = dis

    def forward(self, test_sample, si_sample):
        """"""
        x = self.dis(
            test_sample=test_sample,
            si_sample=si_sample,
        )
        return x


########################################################################################################################
# Impersonator
########################################################################################################################
class GIMGaussianImpersonator(nn.Module):
    """"""
    def __init__(self, src_dim, env_noise_mapper):
        """"""
        super().__init__()
        self.src_dim = src_dim
        self.env_noise_mapper = env_noise_mapper
        self.out_mlp = mb.MLP((2 * src_dim, 2 * src_dim, src_dim))

    def forward(self, leaked_sample, n, remove_noise_mean=True):
        """"""
        batch_size, m, src_dim = leaked_sample.size()
        src = leaked_sample.mean(1) # [batch_size, src_dim]

        # env noise
        z = torch.randn((batch_size, n, self.src_dim), device=leaked_sample.device)
        w = self.env_noise_mapper(z) # [batch_size, n, src_dim]

        # remove_mean
        if remove_noise_mean:
            x = w - w.mean(1, keepdim=True) + src.unsqueeze(1).expand(-1, n, -1)
        else:
            x = w + src.unsqueeze(1).expand(-1, n, -1)
        return x


########################################################################################################################
# Model Getters
########################################################################################################################
def get_im(src_dim):
    """"""
    env_noise_mapper = mb.MLP([src_dim, src_dim])
    im = GIMGaussianImpersonator(src_dim=src_dim, env_noise_mapper=env_noise_mapper)
    return im


def get_au(src_dim):
    """"""
    stat = GIMMeanStdStat()
    dis = GIMGaussianDis(src_dim=src_dim, stat=stat)
    au = GIMGaussianAuthenticator(dis=dis)
    return au


########################################################################################################################
# UNIT TEST
########################################################################################################################
if __name__ == '__main__':
    # Test Functions Defs
    def test_au():
        batch_size = 4
        src_dim = 512
        n = 5
        k = 3
        x = torch.normal(
            mean=torch.zeros((batch_size, n, src_dim)),
            std=torch.ones((batch_size, n, src_dim))
        )
        a = torch.normal(
            mean=torch.zeros((batch_size, k, src_dim)),
            std=torch.ones((batch_size, k, src_dim))
        )
        au = get_au(src_dim)
        out = au(x, a)
        print(out.size())


    def test_im():
        batch_size = 4
        src_dim = 512
        m = 2
        n = 5
        leaked_sample = torch.normal(
            mean=torch.zeros((batch_size, m, src_dim)),
            std=torch.ones((batch_size, m, src_dim))
        )
        im = get_im(src_dim)
        out = im(leaked_sample, n, remove_noise_mean=True)
        print(out.size())


    # Run Tests
    test_au()
    test_im()