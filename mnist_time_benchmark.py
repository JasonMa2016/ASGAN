# mini-training loop for timing purposes!

# adapted (copy pasted) from https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN
import numpy as np
import matplotlib.pyplot as plt
import argparse
import copy
import os, time
import itertools
import pickle
# import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# from torch.autograd import Variable # torch <=0.3
from mnist_models import Generator, Discriminator


parser = argparse.ArgumentParser(description='training runner')
parser.add_argument('--model_type','-m',type=int,default=0,help='Model type') # 0 dcgan, 1 asgan, 2 ergan
parser.add_argument('--save_dir','-sd',type=str,default='DCGAN_MNIST',help='Save directory')
parser.add_argument('--tau','-t',type=float,default=0.3,help='Alpha smoothing parameter')
parser.add_argument('--latent_dim','-ld',type=int,default=100,help='Latent dimension')
parser.add_argument('--batch_size','-bs',type=int,default=63,help='Batch size')
# parser.add_argument('--num_epochs','-ne',type=int,default=50,help='Number of epochs')
parser.add_argument('--learning_rate','-lr',type=float,default=0.0002,help='Learning rate')
parser.add_argument('--gen_file','-gf',type=str,default='generator_param.pkl',help='Save gen filename')
parser.add_argument('--disc_file','-df',type=str,default='discriminator_param.pkl',help='Save disc filename')
# parser.add_argument('--track_space','-ts',action='store_true',help='Save 2D latent space viz, if ld=2')
args = parser.parse_args()
MODELTYPE = args.model_type
SAVEDIR = args.save_dir
GENFILE = args.gen_file
DISCFILE = args.disc_file
# training parameters
tau = args.tau
latent_dim = args.latent_dim
batch_size = args.batch_size
train_epoch = 1
lr = args.learning_rate

if torch.cuda.is_available():
    print('using cuda!')
    torch.cuda.set_device(0)
    dtype = torch.cuda.FloatTensor
    is_cuda = True
else:
    dtype = torch.FloatTensor
    is_cuda = False

def gen_noise():
    z_ = torch.randn((5 * 5), latent_dim).view(-1, latent_dim).type(dtype)
    return z_.cuda() if is_cuda else z_

fixed_z_ = gen_noise() # fixed noise

def show_result(num_epoch, show = False, save = False, path = 'result.png', isFix=False):
    z_ = gen_noise()
    G.eval()
    if isFix:
        test_images = G(fixed_z_)
    else:
        test_images = G(z_)
    G.train()
    test_images = test_images.cpu().data.numpy().transpose(0,2,3,1) # different manip, all 3 channels!
    test_images = (test_images + 1) / 2
    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)
    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ti = test_images[k]
        # transform?? vmin # (ti - np.min(ti)) / (np.max(ti) - np.min(ti))
        ax[i, j].imshow(ti) # , cmap='gray')
    label = 'Epoch {0} (min {1} max {2})'.format(num_epoch, np.min(test_images), np.max(test_images))
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)
    if show:
        plt.show()
    else:
        plt.close()
        

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))
    y1 = hist['D_losses']
    y2 = hist['G_losses']
    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(path)
    if show:
        plt.show()
    else:
        plt.close()

# data_loader
img_size = 28
transform = transforms.Compose([
        transforms.Scale(img_size) if torch.__version__[0] < '1' else transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
sampler = torch.utils.data.SubsetRandomSampler(torch.LongTensor(np.random.choice(np.arange(60000), batch_size * 10)))
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=False, pin_memory = is_cuda, sampler=sampler) # TODO: why doesn't this return cuda.FloatTensors?

print('batch size:', batch_size, 'and whole thing is that times 10')
# usually there are 953 batches per epoch, each batch is 63

# 60000 dataset stacked is 20000
# repeat 6 times per epoch to get 120000 (pacgan does 128000)
# alternatively, can use load_mnist() function, but it is much slower
# from load_mnist import *
# img, lab = load_mnist(128000)

G = Generator()
D = Discriminator()
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
if is_cuda:
    G.cuda()
    D.cuda()

if MODELTYPE == 1:
    G_old = copy.deepcopy(G)

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

def stack(x):
    # assert(x.shape[0] % 3 == 0)
    return torch.cat([x[::3], x[1::3], x[2::3]], dim=1)


# results save folder
# if not os.path.isdir(SAVEDIR):
#     os.mkdir(SAVEDIR)
# if not os.path.isdir(SAVEDIR+'/Random_results'):
#     os.mkdir(SAVEDIR+'/Random_results')
# if not os.path.isdir(SAVEDIR+'/Fixed_results'):
#     os.mkdir(SAVEDIR+'/Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []
num_iter = 0

z_ = torch.randn((int(batch_size/3), latent_dim)).view(-1, latent_dim)
if is_cuda: z_ = z_.cuda()
old_G_result = G(z_) # ERGAN

print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    D_losses = []
    G_losses = []
    epoch_start_time = time.time()
    for i in range(1):
        for x_, _ in train_loader:
            # train discriminator D
            D.zero_grad()

            x_ = stack(x_) # new!
            mini_batch = x_.size()[0]

            y_real_ = torch.ones(mini_batch)
            y_fake_ = torch.zeros(mini_batch)

            if is_cuda: x_, y_real_, y_fake_ = x_.cuda(), y_real_.cuda(), y_fake_.cuda()
            D_result = D(x_).squeeze()
            D_real_loss = BCE_loss(D_result, y_real_)

            z_ = torch.randn((mini_batch, latent_dim)).view(-1, latent_dim)
            if is_cuda: z_ = z_.cuda()
            G_result = G(z_)

            D_result = D(G_result).squeeze()
            D_fake_loss = BCE_loss(D_result, y_fake_)
            D_fake_score = D_result.data.mean()

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

                # G_result = G(z_)
                # D_result = D(G_result).squeeze()
                # G_train_loss = BCE_loss(D_result, y_real_)
                # G_train_loss.backward()
                # G_optimizer.step()
                # G_losses.append(G_train_loss.item())

                G_result = G(z_)
                D_result = D(G_result).squeeze()
                old_D_result = D(old_G_result).squeeze()
                old_G_result = G_result # TODO: will this work?
                G_train_loss = tau * BCE_loss(D_result, y_real_) + (1-tau) * BCE_loss(old_D_result, y_real_)
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
    # p = SAVEDIR+'/Random_results/' + str(epoch + 1) + '.png'
    # fixed_p = SAVEDIR+'/Fixed_results/' + str(epoch + 1) + '.png'
    # show_result((epoch+1), save=True, path=p, isFix=False)
    # show_result((epoch+1), save=True, path=fixed_p, isFix=True)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    # if epoch % 2 == 0:
    #     torch.save(G.state_dict(), SAVEDIR+'/'+GENFILE)
    #     torch.save(D.state_dict(), SAVEDIR+'/'+DISCFILE) # for safety!

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... DON'T save training results")
# torch.save(G.state_dict(), SAVEDIR+'/'+GENFILE)
# torch.save(D.state_dict(), SAVEDIR+'/'+DISCFILE)
# with open(SAVEDIR+'/train_hist.pkl', 'wb') as f:
#     pickle.dump(train_hist, f)

# show_train_hist(train_hist, save=True, path=SAVEDIR+'/train_hist.png')

# images = []
# for e in range(train_epoch):
#     img_name = SAVEDIR+'/Fixed_results/MNIST_DCGAN_' + str(e + 1) + '.png'
#     images.append(imageio.imread(img_name))
# imageio.mimsave(SAVEDIR+'/generation_animation.gif', images, fps=5)
