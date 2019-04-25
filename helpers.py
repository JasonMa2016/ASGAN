# adapted (copy pasted) from https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN
import numpy as np
import matplotlib.pyplot as plt
import itertools
import torch
import random
from math import floor, ceil

if torch.cuda.is_available():
    print('using cuda!')
    torch.cuda.set_device(0) # sets default gpu
    dtype = torch.cuda.FloatTensor
    is_cuda = True
else:
    dtype = torch.FloatTensor
    is_cuda = False

def gen_noise(latent_dim = 100):
    z_ = torch.randn((5 * 5), latent_dim).view(-1, latent_dim).type(dtype)
    return z_.cuda() if is_cuda else z_

fixed_z_ = gen_noise() # fixed noise

def show_result(num_epoch, G, show = False, save = False, path = 'result.png', isFix=False):
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

def stack(x):
    # assert(x.shape[0] % 3 == 0)
    return torch.cat([x[::3], x[1::3], x[2::3]], dim=1)

def patch_with_replay(mini_batch, G, memory, latent_dim = 100, sample_type=0, prop=0.5):
    z_ = torch.randn((floor((1-prop)*mini_batch), latent_dim)).view(-1, latent_dim)
    if is_cuda: z_ = z_.cuda()
    G_result = G(z_)
    # sample from experience
    if sample_type == 0:
        samples = random.sample(memory, ceil(prop*mini_batch))
    elif sample_type == 1:
        lm = len(memory)
        weight_dist = np.linspace(.8/lm, 1.2/lm, lm)
        samples = np.random.choice(memory, ceil(prop*mini_batch), p=weight_dist)
    samples = torch.stack(samples)
    G_result = torch.cat((G_result, samples))
    return G_result

