# adapted (copy pasted) from https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN
import os, time
import matplotlib.pyplot as plt
import itertools
import pickle
# import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.autograd import Variable
from math import ceil, log
from evaluate import evaluate
import numpy as np

# from https://www.zealseeker.com/archives/jensen-shannon-divergence-jsd-python/
class JSD:
    def KLD(self,p,q):
        if 0 in q :
            raise ValueError
        return sum(_p * log(_p/_q) for (_p,_q) in zip(p,q) if _p!=0)
    def JSD_core(self,p,q):
        M = [0.5*(_p+_q) for _p,_q in zip(p,q)]
        return 0.5*self.KLD(p,M)+0.5*self.KLD(q,M)

# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d*4, 4, 1, 0) # changed things
        self.deconv1_bn = nn.BatchNorm2d(d*4)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 3, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, 3, 4, 2, 1) # 1 to 3, 4 to 3
        # self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        # self.deconv4_bn = nn.BatchNorm2d(d)
        # self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1) # 1 to 3
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x))
        # x = F.relu(self.deconv4_bn(self.deconv4(x)))
        # x = torch.tanh(self.deconv5(x)) # deprecated
        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

def stack(x):
    assert(x.shape[0] % 3 == 0)
    return torch.cat([x[::3], x[1::3], x[2::3]], dim=1)

# network
G = generator(32)
G.load_state_dict(torch.load("MNIST_ASGAN_results/generator_param.pkl"))

batch_size = 129 # 64 in pacgan
num_test_sample = 25929

# def log_metrics(self, epoch):
results = []
for i in range(int(ceil(float(num_test_sample) / float(batch_size)))):
    # input_z_samples = self.test_samples[i * batch_size : (i + 1) * batch_size]
    # samples = self.sess.run(self.sampler, feed_dict={self.z[0]: input_z_samples})
    input_z_samples = torch.randn((batch_size, 100)).view(-1, 100, 1, 1)
    samples = G(input_z_samples).data.numpy().transpose(0, 2, 3, 1)
    # todo: reshape into numpy for consumption by keras!??

    dect0 = evaluate(np.reshape(samples[:, :, :, 0], (batch_size, 28, 28, 1)))
    dect1 = evaluate(np.reshape(samples[:, :, :, 1], (batch_size, 28, 28, 1)))
    dect2 = evaluate(np.reshape(samples[:, :, :, 2], (batch_size, 28, 28, 1)))

    new_results = zip(dect0, dect1, dect2)
    if len(results) + min(len(dect0), len(dect1), len(dect2)) > num_test_sample:
        results.extend(list(new_results[0 : (num_test_sample - len(results))]))
    else:
        results.extend(new_results)
        
        # if i % 10 == 0:
        #     save_images(np.reshape(samples[0, :, :, 0], (1, 28, 28, 1)), image_manifold_size(1), os.path.join(self.sample_dir, "eva_epoch{}_i{}_dect{}.png".format(epoch, i, dect0[0])))

map = {}
for result in results:
    if result in map:
        map[result] += 1
    else:
        map[result] = 1

num_mode = len(map.keys())
p = np.zeros(1000)
p[0:num_mode] = list(map.values())
p = p / np.sum(p)
q = [1.0 / 1000.0] * 1000
kl = JSD().KLD(p, q) 

print('num mode', num_mode, 'kl', kl)
print(map)

# with open(self.metric_path, "ab") as csv_file:
#     writer = csv.DictWriter(csv_file, fieldnames=self.field_names)
#     writer.writerow({
#         "epoch": epoch,
#         "mode coverage": num_mode, 
#         "KL": kl,
#         "details": map
#     })