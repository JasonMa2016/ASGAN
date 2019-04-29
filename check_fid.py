# adapted (copy pasted) from https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN
import argparse
import os, time
# import matplotlib.pyplot as plt
# import itertools
# import pickle
# import imageio
import torch
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
from torchvision import datasets, transforms
# from torch.autograd import Variable
from math import ceil, log
from mnist_models import *
from helpers import *

from scipy.imageio import imwrite

parser = argparse.ArgumentParser(description='training runner')
parser.add_argument('--save_dir','-sd',type=str,default='DCGAN_MNIST',help='Save directory')
# 0 dcgan, 1 asgan, 2 ergan, 3 ergan weighted smoothing
parser.add_argument('--arch_type','-a',type=int,default=0,help='Architecture type')
parser.add_argument('--latent_dim','-ld',type=int,default=100,help='Latent dimension')
parser.add_argument('--batch_size','-bs',type=int,default=63,help='Batch size')
parser.add_argument('--gen_file','-gf',type=str,default='generator_param.pkl',help='Save gen filename')
parser.add_argument('--disc_file','-df',type=str,default='discriminator_param.pkl',help='Save disc filename')
args = parser.parse_args()
SAVEDIR = args.save_dir
ARCHTYPE = args.arch_type
GENFILE = args.gen_file
DISCFILE = args.disc_file
# training parameters
latent_dim = args.latent_dim
batch_size = args.batch_size

if torch.cuda.is_available():
    print('using cuda!')
    torch.cuda.set_device(0)
    dtype = torch.cuda.FloatTensor
    is_cuda = True
else:
    dtype = torch.FloatTensor
    is_cuda = False

if not os.path.exists('../data/real'):
    os.mkdir('../data')
    os.mkdir('../data/real') # 2 separate steps
    # data_loader
    img_size = 28
    transform = transforms.Compose([
            transforms.Scale(img_size) if torch.__version__[0] < '1' else transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True, pin_memory = is_cuda, drop_last = True) # TODO: why doesn't this return cuda.FloatTensors?
    i = 0
    for x_, _ in train_loader:
        x_ = stack(x_).permute(0,2,3,1)
        for samp in x_.cpu().numpy():
            imsave('../data/real/'+str(i)+'.png', samp)
            i += 1
        if i >= 3000:
            break

    print('finished generating real images')

if not os.path.exists(SAVEDIR+'/fake'):
    os.mkdir(SAVEDIR+'/fake')

# network
if ARCHTYPE == 0:
    G = Generator()
else:
    G = Generator1()

G = nn.DataParallel(G)
G.load_state_dict(torch.load(SAVEDIR+'/'+GENFILE))
if is_cuda:
    G.cuda()

i=0
for epoch in range(3000):
    input_z_samples = torch.randn((batch_size, latent_dim)).view(-1, latent_dim)
    if is_cuda: input_z_samples = input_z_samples.cuda()
    samples = G(input_z_samples).cpu().data.numpy().transpose(0,2,3,1)
    for samp in samples:
        imsave(SAVEDIR+'/fake/'+str(i)+'.png', samp)
        i += 1

print('finished saving fake images at', SAVEDIR)

# save fake data. check that the thing actually works. check dimensionality of activations.
# how long it take to save the images? not too bad. but passing through inception net takes a while.
# what are the dimensions of the inception thing? does it matter that we pass in ints?
# calculate inception score? fid with mnist-cnn?

from fid_score_pytorch import calculate_fid_given_paths
fid_value = calculate_fid_given_paths(['../data/real',SAVEDIR+'/fake'],
                                          50,
                                          args.gpu != '',
                                          2048)
print('FID', fid_value)


# np.savez('blah.npz',mu=a,sigma=b)