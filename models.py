import torch
import torch.nn as nn
from configs import *

class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_size):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=False):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, hidden_size),
            *block(hidden_size,hidden_size),
            nn.Linear(hidden_size,latent_dim)
        )

    def forward(self, z):
        gaussian = self.model(z)
        return gaussian

class Discriminator(nn.Module):
    def __init__(self, latent_dim, hidden_size):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size,1),
            nn.Sigmoid()
        )

    def forward(self, z):
        validity = self.model(z)
        return validity