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
from helpers import stack

from fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
# from imageio import imwrite
# from scipy.misc import imsave
from numpy import save

parser = argparse.ArgumentParser(description='training runner')
parser.add_argument('--save_dir','-sd',type=str,default='DCGAN_MNIST',help='Save directory')
# 0 dcgan, 1 asgan, 2 ergan, 3 ergan weighted smoothing
parser.add_argument('--arch_type','-a',type=int,default=0,help='Architecture type')
parser.add_argument('--latent_dim','-ld',type=int,default=100,help='Latent dimension')
parser.add_argument('--batch_size','-bs',type=int,default=300,help='Batch size')
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
    # torch.cuda.set_device(0)
    dtype = torch.cuda.FloatTensor
    is_cuda = True
else:
    dtype = torch.FloatTensor
    is_cuda = False

# code from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py
dims = 2048
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
model = InceptionV3([block_idx], normalize_input = False)
model.eval()
if is_cuda:
    model.cuda()

model = nn.DataParallel(model)

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
    # i = 0
    # for x_, _ in train_loader:
    #     x_ = stack(x_).permute(0,2,3,1)
    #     for samp in x_.cpu().numpy():
    #         imsave('../data/real/'+str(i)+'.png', samp)
    #         i += 1
    #     if i >= 3000:
    #         break
    i = 0
    pred_arr = torch.empty(3000,dims)
    # pred_arr = np.empty((3000, dims))
    for x_, _ in train_loader:
        print(i)
        if is_cuda: x_ = x_.cuda()
        x_ = stack(x_)
        pred = model(x_)[0] # a list of a single element of shape 256,2048,1,1
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred_arr[i:(i+len(pred))] = pred.squeeze(3).squeeze(2)# .cpu().data.numpy()
        i += len(x_)
        if i >= 3000: break

    save('../data/real_activations.npy', pred_arr.cpu().data.numpy())
    print('finished real images')

########################

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

i = 0
samp_arr = torch.empty(3000,3,28,28)
# pred_arr = np.empty((3000, dims))
for epoch in range(30):
    i = 100 * epoch
    print(i)
    input_z_samples = torch.randn((100, latent_dim)).view(-1, latent_dim)
    if is_cuda: input_z_samples = input_z_samples.cuda()
    # samples = G(input_z_samples).cpu().data.numpy().transpose(0,2,3,1)
    # for samp in samples:
    #     imsave(SAVEDIR+'/fake/'+str(i)+'.png', samp)
    #     i += 1
    samp = G(input_z_samples)
    samp_arr[i:(i+len(samp))] = samp

samp_arr = samp_arr.cpu().detach()
torch.save(samp_arr, 'samp_arr.pkl')

for name in dir():
    if not name.startswith('_') and name not in ['torch','dims','model','samp_arr','adaptive_avg_pool2d','save','SAVEDIR']:
        del globals()[name]

pred_arr = torch.empty(3000, dims)
for epoch in range(30):
    i = 100 * epoch
    print(i)
    pred = model(samp_arr[i:(i+100)].cuda())[0] # a list of a single element of shape 256,2048,1,1
    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.shape[2] != 1 or pred.shape[3] != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
    pred_arr[i:(i+len(pred))] = pred.squeeze(3).squeeze(2)# .cpu().data.numpy()
    del pred

save(SAVEDIR+'/fake_activations.npy', pred_arr)
print('finished saving fake images at', SAVEDIR)

# save fake data. check that the thing actually works. check dimensionality of activations.
# how long it take to save the images? not too bad. but passing through inception net takes a while.
# what are the dimensions of the inception thing? does it matter that we pass in ints? (it accepts -1 to 1)
# calculate inception score? fid with mnist-cnn?
'''
what i should do
make a new cnn in pytorch
try a fid where i paste together the activations from each digit separately.
'''

# from fid_score_pytorch import calculate_fid_given_paths
# fid_value = calculate_fid_given_paths(['../data/real',SAVEDIR+'/fake'],
#                                           50,
#                                           args.gpu != '',
#                                           2048)
# print('FID', fid_value)


for name in dir():
    if not name.startswith('_') and name not in ['np','SAVEDIR']:
        del globals()[name]

# np.savez('blah.npz',mu=a,sigma=b)
import tensorflow
from fid.fid_score import compute_fid_from_activations
real_activations, fake_activations = np.load('../data/real_activations.npy'), np.load(SAVEDIR+'/fake_activations.npy')
fid = compute_fid_from_activations(real_activations, fake_activations)
print('fid',fid)