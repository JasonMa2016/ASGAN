import torch
import torch.nn as nn
import torch.nn.functional as F

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

# 4517891 params
class Generator(nn.Module):
    def __init__(self, d=64, ld=100):
        super(Generator, self).__init__()
        self.d = d
        self.linear = nn.Linear(ld, 2*2*d*8)
        self.linear_bn = nn.BatchNorm1d(2*2*d*8)
        self.deconv1 = nn.ConvTranspose2d(d*8, d*4, 5, 2, 1) # changed things
        self.deconv1_bn = nn.BatchNorm2d(d*4)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 5, 2, 2)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 5, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, 3, 5, 2, 1)
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    def forward(self, input):
        x = F.relu(self.linear_bn(self.linear(input)))
        x = x.view(-1, self.d*8, 2, 2)
        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        x = x[:,:,:-1,:-1] # hacky way to get shapes right (like "SAME" in tf)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = x[:,:,:-1,:-1]
        x = torch.tanh(self.deconv4(x))
        x = x[:,:,:-1,:-1]
        return x

# test dimensions
# import time
# G = Generator()
# start = time.time()
# x = torch.zeros(43, 100)
# y = G(x)
# print(time.time()-start)

# 4310401 params
class Discriminator(nn.Module):
    def __init__(self, d=64):
        super(Discriminator, self).__init__()
        self.d = d
        self.conv1 = nn.Conv2d(3, d, 5, 2, 2)
        self.conv2 = nn.Conv2d(d, d*2, 5, 2, 2)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 5, 2, 2)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 5, 2, 2)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.linear = nn.Linear(2*2*d*8, 1)
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = x.view(-1, 2*2*self.d*8)
        x = torch.sigmoid(self.linear(x))
        return x

# test dimensions
# start = time.time()
# D = Discriminator()
# x = torch.zeros(43, 3, 28, 28)
# y = D(x)
# print(time.time()-start)