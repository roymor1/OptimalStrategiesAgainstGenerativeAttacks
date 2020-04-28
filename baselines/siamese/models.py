# imports
import os
import sys
import torch
import torch.nn as nn


# local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
sys.path.append(project_root)


# Embedding nets
class ProtonetEmbeddingNet(nn.Module):
    """"""
    def __init__(self, inp_n_channels, inp_img_size, hidden_dim=64, z_dim=64):
        """

        :param n_classes:
        :param inp_n_channels:
        :param inp_img_size: A power of 2
        """
        super(ProtonetEmbeddingNet, self).__init__()
        self.inp_n_channels = inp_n_channels
        self.inp_img_size = inp_img_size
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            self._conv_block(inp_n_channels, hidden_dim),
            self._conv_block(hidden_dim, hidden_dim),
            self._conv_block(hidden_dim, hidden_dim),
            self._conv_block(hidden_dim, z_dim),
        )

    def forward(self, x):
        """
        :param x: [batch, channels, img_size, img_size]
        :return: [batch, n_classes]
        """
        batch_size = x.size(0)
        out = self.encoder(x)
        return out.view(batch_size, -1)

    @property
    def embedding_dim(self):
        """"""
        out_img_size = int(self.inp_img_size / (2 ** 4))
        return int(self.z_dim * out_img_size * out_img_size)

    @staticmethod
    def _conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )


class SimpleEmbeddingNet(nn.Module):
    def __init__(self):
        super(SimpleEmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class SimpleEmbeddingNetL2(SimpleEmbeddingNet):
    def __init__(self):
        super(SimpleEmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(SimpleEmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class SiameseNet(nn.Module):
    def __init__(self, embedding_net, embedding_dim):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net
        self.fc = nn.Linear(embedding_dim, 1)

    def encode(self, x):
        return self.embedding_net(x)

    def classify(self, emb1, emb2):
        out = torch.abs(emb1 - emb2)
        out = self.fc(out)
        return out

    def forward(self, x1, x2):
        emb1 = self.encode(x1)
        emb2 = self.encode(x2)
        return self.classify(emb1, emb2)
