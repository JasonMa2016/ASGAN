# mini-training loop for timing purposes!
# only difference: smaller batch size, no for-loop in each epoch, and no saving!

# sampler = torch.utils.data.SubsetRandomSampler(torch.LongTensor(np.random.choice(np.arange(60000), batch_size * 10)))
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True, transform=transform),
#     batch_size=batch_size, shuffle=False, pin_memory = is_cuda, sampler=sampler) # TODO: why doesn't this return cuda.FloatTensors?



# adapted (copy pasted) from https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN
import numpy as np
import matplotlib.pyplot as plt
import argparse
import copy
import os, time
import pickle
# import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# from torch.autograd import Variable # torch <=0.3
from mnist_models import *
from helpers import *
from collections import deque
from tqdm import tqdm
from math import floor, ceil

parser = argparse.ArgumentParser(description='training runner')
parser.add_argument('--model_type','-m',type=int,default=0,help='Model type')
# 0 dcgan, 1 asgan, 2 ergan, 3 ergan weighted smoothing
parser.add_argument('--arch_type','-a',type=int,default=0,help='Architecture type')
parser.add_argument('--save_dir','-sd',type=str,default='DCGAN_MNIST',help='Save directory')
parser.add_argument('--tau','-t',type=float,default=0.3,help='Alpha smoothing parameter')
parser.add_argument('--latent_dim','-ld',type=int,default=100,help='Latent dimension')
parser.add_argument('--batch_size','-bs',type=int,default=540,help='Batch size') # not 63
parser.add_argument('--num_epochs','-ne',type=int,default=30,help='Number of epochs')
parser.add_argument('--learning_rate','-lr',type=float,default=0.0002,help='Learning rate')
parser.add_argument('--gen_file','-gf',type=str,default='generator_param.pkl',help='Save gen filename')
parser.add_argument('--disc_file','-df',type=str,default='discriminator_param.pkl',help='Save disc filename')
# parser.add_argument('--track_space','-ts',action='store_true',help='Save 2D latent space viz, if ld=2')
args = parser.parse_args()
MODELTYPE = args.model_type
ARCHTYPE = args.arch_type
SAVEDIR = args.save_dir
GENFILE = args.gen_file
DISCFILE = args.disc_file
# training parameters
tau = args.tau
latent_dim = args.latent_dim
batch_size = args.batch_size
train_epoch = args.num_epochs
lr = args.learning_rate

if torch.cuda.is_available():
    print('using cuda!')
    torch.cuda.set_device(0) # sets default gpu
    dtype = torch.cuda.FloatTensor
    is_cuda = True
else:
    dtype = torch.FloatTensor
    is_cuda = False

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

print('batch size:', batch_size, 'len train_loader', len(train_loader))

# 60000 dataset stacked is 20000
# repeat 6 times per epoch to get 120000 (pacgan does 128000)
# alternatively, can use load_mnist() function, but it is much slower
# from load_mnist import *
# img, lab = load_mnist(128000)

if ARCHTYPE == 0:
    G = Generator()
    D = Discriminator()
elif ARCHTYPE == 1:
    G = Generator1()
    D = Discriminator1()

G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G = nn.DataParallel(G)
D = nn.DataParallel(D)
if is_cuda:
    G.cuda()
    D.cuda()

# ASGAN
if MODELTYPE == 1:
    G_old = copy.deepcopy(G)

# ERGAN
z_ = torch.randn((int(batch_size/3), latent_dim)).view(-1, latent_dim)
if is_cuda: z_ = z_.cuda()

old_G_result = G(z_)
memory = deque(maxlen = len(train_loader) * batch_size // 3) # TODO: check that size is right!

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# results save folder
if not os.path.isdir(SAVEDIR):
    os.mkdir(SAVEDIR)
if not os.path.isdir(SAVEDIR+'/Random_results'):
    os.mkdir(SAVEDIR+'/Random_results')
if not os.path.isdir(SAVEDIR+'/Fixed_results'):
    os.mkdir(SAVEDIR+'/Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []
num_iter = 0

print('training start!')
start_time = time.time()

# fewer epochs for sanity
for epoch in tqdm(range(2)):
    D_losses = []
    G_losses = []
    epoch_start_time = time.time()
    for i in range(1):
        for x_, _ in train_loader:
            # train discriminator D
            D.zero_grad()

            x_ = stack(x_) # stacking changes the size
            mini_batch = x_.size()[0]

            y_real_ = torch.ones(mini_batch)
            y_fake_ = torch.zeros(mini_batch)
            if is_cuda: x_, y_real_, y_fake_ = x_.cuda(), y_real_.cuda(), y_fake_.cuda()

            D_result = D(x_).squeeze()
            D_real_loss = BCE_loss(D_result, y_real_)

            if MODELTYPE == 2 and len(memory) > mini_batch//2 and epoch >= 1:
                G_result = patch_with_replay(mini_batch, G, memory)
            else:
                z_ = torch.randn((mini_batch, latent_dim)).view(-1, latent_dim)
                if is_cuda: z_ = z_.cuda()
                G_result = G(z_)

            D_result = D(G_result).squeeze() # TODO: the shape still works out right?

            D_fake_loss = BCE_loss(D_result, y_fake_) # that old way is now deprecated
            # D_fake_score = D_result.data.mean()

            D_train_loss = D_real_loss + D_fake_loss

            D_train_loss.backward()
            D_optimizer.step()

            D_losses.append(D_train_loss.item())

            # train generator G
            # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
            for j in range(2):
                G.zero_grad()

                z_ = torch.randn((mini_batch, latent_dim)).view(-1, latent_dim)
                if is_cuda: z_ = z_.cuda()

                G_result = G(z_)

                # add to memory
                if MODELTYPE == 2:
                    for i in range(G_result.shape[0]):
                        memory.append(G_result[i].detach())

                D_result = D(G_result).squeeze()
                G_train_loss = BCE_loss(D_result, y_real_)

                G_train_loss.backward()
                G_optimizer.step()
                G_losses.append(G_train_loss.item())

            # alpha smoothing
            if MODELTYPE == 1:
                for model_param, old_param in zip(G.parameters(), G_old.parameters()):
                    model_param.data.copy_(model_param.data*(1-tau) + old_param.data*tau)
                    old_param.data.copy_(model_param.data)

            num_iter += 1

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time


    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G_losses))))
    p = SAVEDIR+'/Random_results/' + str(epoch + 1) + '.png'
    fixed_p = SAVEDIR+'/Fixed_results/' + str(epoch + 1) + '.png'
    # show_result((epoch+1), G, save=True, path=p, isFix=False)
    # show_result((epoch+1), G, save=True, path=fixed_p, isFix=True)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    # if epoch % 2 == 0:
    #     torch.save(G.state_dict(), SAVEDIR+'/'+GENFILE)
    #     torch.save(D.state_dict(), SAVEDIR+'/'+DISCFILE) # for safety!
    #     if MODELTYPE == 2:
    #         torch.save(memory, SAVEDIR+'/'+'memory.pkl')

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
# torch.save(G.state_dict(), SAVEDIR+'/'+GENFILE)
# torch.save(D.state_dict(), SAVEDIR+'/'+DISCFILE)
# if MODELTYPE == 2:
#     torch.save(memory, SAVEDIR+'/'+'memory.pkl')

# with open(SAVEDIR+'/train_hist.pkl', 'wb') as f:
#     pickle.dump(train_hist, f)

# show_train_hist(train_hist, save=True, path=SAVEDIR+'/train_hist.png')

# images = []
# for e in range(train_epoch):
#     img_name = SAVEDIR+'/Fixed_results/MNIST_DCGAN_' + str(e + 1) + '.png'
#     images.append(imageio.imread(img_name))
# imageio.mimsave('DCGAN_MNIST/generation_animation.gif', images, fps=5)
