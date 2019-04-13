from utils import *
from gan import *
import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.distributions as ds
import matplotlib.pyplot as plt

latent_dim = 2 # 2d Gaussian
HIDDEN_SIZE = 128
EPOCH_SIZE = 20000
BATCH_SIZE = 1000

tau = 0.3

gan = ERGAN(latent_dim, HIDDEN_SIZE)
grid = False

if grid:
    DIR_NAME = 'ERGAN_2dgrid'
    create = create_grid
    mode_count = count_mode_grid
else:
    DIR_NAME = 'ERGAN_2dring'
    create = create_ring
    mode_count = count_mode_ring
    
os.makedirs(DIR_NAME, exist_ok=True)

for i in range(20):
    if i == 10:
        DIR_NAME = 'ERGAN_2dgrid'
        create = create_grid
        mode_count = count_mode_grid
        os.makedirs(DIR_NAME, exist_ok=True)
        print("Now, 2d-grid")
    for epoch in range(EPOCH_SIZE):
        real_data = create(BATCH_SIZE)
        # z = Variable(torch.FloatTensor(np.random.normal(0,1, (BATCH_SIZE,latent_dim))))
        
        
        real_data, fake_data, d_loss, g_loss = gan.optimize(BATCH_SIZE, real_data, epoch)
        
        if epoch % 100 == 0:
            # print("[Epoch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, EPOCH_SIZE, d_loss, g_loss))
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.scatter(x=real_data[:,0], y=real_data[:,1], c='g', s=100)
            ax1.scatter(x=fake_data[:,0], y=fake_data[:,1], c='b',alpha=0.1, s=100)
            # plt.show()
            
            fig.savefig('{}/{}.png'.format(DIR_NAME, epoch))

    batch_size = 2500
    real_normals = create(batch_size)
    z = Variable(torch.FloatTensor(np.random.normal(0,1, (batch_size,latent_dim))))
    fake_normals = gan.G(z)
    fake_normals = fake_normals.detach().numpy()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x=real_normals[:,0], y=real_normals[:,1], c='g', s=100)
    ax1.scatter(x=fake_normals[:,0], y=fake_normals[:,1], c='b',alpha=0.02, s=100)
    plt.show()

    fig.savefig('{}/final.png'.format(DIR_NAME))

    mode_count(fake_normals)

    torch.save(gan.D.state_dict(), DIR_NAME+'/asgan_2dring_d.pkl')
    torch.save(gan.G.state_dict(), DIR_NAME+'/asgan_2dring_g.pkl')
# gan.D.load_state_dict(torch.load(DIR_NAME+'/asgan_2dring_d.pkl'))
# gan.G.load_state_dict(torch.load(DIR_NAME+'/asgan_2dring_g.pkl'))