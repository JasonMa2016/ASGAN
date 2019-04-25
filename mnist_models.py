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

def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)    

def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

# 664933 params
class Generator1(nn.Module):
    def __init__(self, latent_dim=100, image_size=28, conv_dim=32):
        super(Generator1, self).__init__()
        self.fc = deconv(latent_dim, conv_dim*8, 2, stride=1, pad=0, bn=False)
        self.deconv1 = deconv(conv_dim*8, conv_dim*4, 4)
        self.deconv2 = deconv(conv_dim*4, conv_dim*2, 3) # hacky to change kernel size
        self.deconv3 = deconv(conv_dim*2, conv_dim, 4)
        self.deconv4 = deconv(conv_dim, 1, 4, bn=False)
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.fc(z) # (?, 256, 2, 2)
        out = F.leaky_relu(self.deconv1(out), 0.05) # (?, 128, 4, 4)
        out = F.leaky_relu(self.deconv2(out), 0.05) # (?, 64, 7, 7)
        out = F.leaky_relu(self.deconv3(out), 0.05) # (?, 32, 14, 14)
        out = torch.sigmoid(self.deconv4(out)) # (?, 1, 28, 28)
        return out

#2394849 params
class Discriminator2(nn.Module):
    def __init__(self, image_size=28, conv_dim=32):
        super(Discriminator2, self).__init__()
        self.conv_dim = conv_dim
        self.conv1 = conv(1, conv_dim, 4, stride=1, pad=1, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4, stride=1, pad=0)
        self.maxpool= nn.MaxPool2d(2,padding=1)
        self.linear = nn.Linear(conv_dim*2*6*6, 1024)
        self.output = nn.Linear(1024, 1)
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05) # (?, 32, 27, 27)
        out = self.maxpool(out) # (?, 32, 13, 13)
        out = F.leaky_relu(self.conv2(out), 0.05) # (?, 64, 10, 10)
        out = self.maxpool(out) # (?, 64, 6, 6)
        out = out.view(-1, self.conv_dim*2*6*6) # (?, 64*8*8)
        out = F.leaky_relu(self.linear(out), 0.05) # (?, 1024)
        return torch.sigmoid(self.output(out).squeeze())