from models import *
from configs import *
from torch.autograd import Variable
import torch.nn
import torch
import copy

class GAN(object):
    def __init__(self, latent_dim, hidden_size):
        self.G = Generator(latent_dim, hidden_size)
        self.D  = Discriminator(latent_dim, hidden_size)
        self.loss =torch.nn.BCELoss()

        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=LR, betas=(B1, B2))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=LR, betas=(B1, B2))

    def optimize(self, batch_size, real_data, noise):
        fake_data = self.G(noise)

        valid = Variable(torch.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(torch.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        self.optimizer_D.zero_grad()
        real_loss = self.loss(self.D(torch.FloatTensor(real_data)), valid)
        fake_loss = self.loss(self.D(fake_data.detach()),fake)
        d_loss = (real_loss + fake_loss)/2
        d_loss.backward()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        g_loss = self.loss(self.D(fake_data), valid)
        g_loss.backward()
        self.optimizer_G.step()

        return real_data, fake_data.detach().numpy(), d_loss.item(), g_loss.item()

class ASGAN(object):
    def __init__(self, latent_dim, hidden_size, tau):
        self.G = Generator(latent_dim, hidden_size)
        self.D  = Discriminator(latent_dim, hidden_size)
        self.G_old = copy.deepcopy(self.G)
        self.tau = tau

        self.loss =torch.nn.BCELoss()
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=LR, betas=(B1, B2))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=LR, betas=(B1, B2))

    def optimize(self, batch_size, real_data, noise):
        fake_data = self.G(noise)

        valid = Variable(torch.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(torch.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        self.optimizer_D.zero_grad()
        real_loss = self.loss(self.D(torch.FloatTensor(real_data)), valid)
        fake_loss = self.loss(self.D(fake_data.detach()),fake)
        d_loss = (real_loss + fake_loss)/2
        d_loss.backward()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        g_loss = self.loss(self.D(fake_data), valid)
        g_loss.backward()
        self.optimizer_G.step()

        for model_param, old_param in zip(self.G.model.parameters(), self.G_old.parameters()):
            model_param.data.copy_(model_param.data*(1-self.tau) + old_param.data*self.tau)
            old_param.data.copy_(model_param.data)

        return real_data, fake_data.detach().numpy(), d_loss.item(), g_loss.item()