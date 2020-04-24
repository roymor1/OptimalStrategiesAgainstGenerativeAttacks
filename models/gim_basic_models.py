# imports
import os
import sys
import copy
import random
import numpy as np
import torch
import torch.nn as nn

# local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(project_root)
import models.model_blocks as mb
# from training.utils import clones


########################################################################################################################
# GIM Sample Statistic
########################################################################################################################
class GIMMeanStat(nn.Module):
    """"""

    def __init__(self):
        """"""
        super().__init__()
        self.n_stats = 1

    def forward(self, x):
        """

        :param x: [batch, sample_size, latent]
        :return: [batch, latent]
        """
        return x.mean(1)


class GIMStdStat(nn.Module):
    """"""

    def __init__(self):
        """"""
        super().__init__()
        self.n_stats = 1

    def forward(self, x):
        """

        :param x: [batch, sample_size, latent]
        :return: [batch, latent]
        """
        return mb.custom_std(x)


class GIMLogVarStat(nn.Module):
    """"""

    def __init__(self):
        """"""
        super().__init__()
        self.n_stats = 1

    def forward(self, x):
        """

        :param x: [batch, sample_size, latent]
        :return: [batch, latent]
        """
        return torch.log(x.var(1) + 1e-8)


class GIMMeanStdStat(nn.Module):
    """"""

    def __init__(self):
        """"""
        super().__init__()
        self.n_stats = 2
        self.sample_mean = GIMMeanStat()
        self.sample_std = GIMStdStat()

    def forward(self, x):
        """

        :param x: [batch, sample_size, latent]
        :return: [batch, latent]
        """
        x1 = self.sample_mean(x)
        x2 = self.sample_std(x)
        return torch.cat((x1, x2), dim=-1)


class GIMMeanLogVarStat(nn.Module):
    """"""

    def __init__(self):
        """"""
        super().__init__()
        self.n_stats = 2
        self.sample_mean = GIMMeanStat()
        self.sample_log_var = GIMLogVarStat()

    def forward(self, x):
        """

        :param x: [batch, sample_size, latent]
        :return: [batch, latent]
        """
        x1 = self.sample_mean(x)
        x2 = self.sample_log_var(x)
        return torch.cat((x1, x2), dim=-1)


class GIMFCStat(nn.Module):
    """"""

    def __init__(self, style_dim, n_stats=1, hidden_layers=()):
        """"""
        super().__init__()
        self.style_dim = style_dim
        self.n_stats = n_stats
        self.fc_layer_dims = [style_dim] + [*hidden_layers] + [n_stats * style_dim]
        self.stat = mb.MLP(self.fc_layer_dims)
        self.sample_mean = GIMMeanStat()

    def forward(self, x):
        """"""
        return self.sample_mean(self.stat(x))


class GIMDoubleFCStat(nn.Module):
    """"""

    def __init__(self, style_dim, n_stats=1, hidden_layers1=(), hidden_layers2=()):
        """"""
        super().__init__()
        self.style_dim = style_dim
        self.n_stats = n_stats
        self.fc1_layer_dims = [style_dim] + [*hidden_layers1] + [n_stats * style_dim]
        self.fc2_layer_dims = [n_stats * style_dim] + [*hidden_layers2] + [n_stats * style_dim]
        self.stat1 = mb.MLP(self.fc1_layer_dims)
        self.stat2 = mb.MLP(self.fc2_layer_dims)
        self.sample_mean = GIMMeanStat()

    def forward(self, x):
        """"""
        x = self.stat1(x)
        x = self.sample_mean(x)
        x = self.stat2(x)
        return x


class GIMMeanStdFcStat(nn.Module):
    """"""

    def __init__(self, style_dim, fc_n_stats, fc_hidden_layers):
        """"""
        super().__init__()
        self.n_stats = 2 + fc_n_stats
        self.sample_mean = GIMMeanStat()
        self.sample_std = GIMStdStat()
        self.fc = GIMFCStat(style_dim=style_dim, n_stats=fc_n_stats, hidden_layers=fc_hidden_layers)

    def forward(self, x):
        """

        :param x: [batch, sample_size, latent]
        :return: [batch, latent]
        """
        x1 = self.sample_mean(x)
        x2 = self.sample_std(x)
        x3 = self.fc(x)
        return torch.cat((x1, x2, x3), dim=-1)


########################################################################################################################
# GIM Discriminators
########################################################################################################################


########################################################################################################################
# GIM Latent Mappers
########################################################################################################################


########################################################################################################################
# Authenticator
########################################################################################################################


########################################################################################################################
# Impersonator
########################################################################################################################


########################################################################################################################
# Unit Test
########################################################################################################################

if __name__ == '__main__':
    pass