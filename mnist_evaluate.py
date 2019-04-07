# copied from PacGan repo evaluate.py
# import theano
# from scipy import misc
import numpy as np
import keras
model = keras.models.load_model("mnist_cnn.hdf5")

def evaluate(x):
    output = model.predict(x)
    return list(np.argmax(output, axis=1))

# if __name__ == "__main__":
#     p2 = numpy.reshape(misc.imread("2.png"), (1, 28, 28, 1))
#     p9 = numpy.reshape(misc.imread("9.png"), (1, 28, 28, 1))
#     print(evaluate(np.concatenate((p2, p9), axis=0)))


# adapted (copy pasted) from https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN
import argparse
# import os, time
# import matplotlib.pyplot as plt
# import itertools
# import pickle
# import imageio
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.autograd import Variable
from math import ceil, log
from mnist_models import Generator

parser = argparse.ArgumentParser(description='training runner')
parser.add_argument('--save_dir','-sd',type=str,default='DCGAN_MNIST',help='Save directory')
parser.add_argument('--latent_dim','-ld',type=int,default=100,help='Latent dimension')
parser.add_argument('--batch_size','-bs',type=int,default=63,help='Batch size')
parser.add_argument('--gen_file','-gf',type=str,default='generator_param.pkl',help='Save gen filename')
parser.add_argument('--disc_file','-df',type=str,default='discriminator_param.pkl',help='Save disc filename')
args = parser.parse_args()
SAVEDIR = args.save_dir
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

# from https://www.zealseeker.com/archives/jensen-shannon-divergence-jsd-python/
class JSD:
    def KLD(self,p,q):
        if 0 in q :
            raise ValueError
        return sum(_p * log(_p/_q) for (_p,_q) in zip(p,q) if _p!=0)
    def JSD_core(self,p,q):
        M = [0.5*(_p+_q) for _p,_q in zip(p,q)]
        return 0.5*self.KLD(p,M)+0.5*self.KLD(q,M)

def stack(x):
    assert(x.shape[0] % 3 == 0)
    return torch.cat([x[::3], x[1::3], x[2::3]], dim=1)

# network
G = Generator()
G.load_state_dict(torch.load(SAVEDIR+'/'+GENFILE))
if is_cuda:
    G.cuda()

batch_size = 63 # 129 previously
num_test_sample = ceil(26000/batch_size) * batch_size # yes, this is lazy

# def log_metrics(self, epoch):
results = []
for i in range(int(ceil(float(num_test_sample) / float(batch_size)))):
    # input_z_samples = self.test_samples[i * batch_size : (i + 1) * batch_size]
    # samples = self.sess.run(self.sampler, feed_dict={self.z[0]: input_z_samples})
    input_z_samples = torch.randn((batch_size, latent_dim)).view(-1, latent_dim)
    if is_cuda: input_z_samples = input_z_samples.cuda()
    samples = G(input_z_samples).cpu().data.numpy().transpose(0, 2, 3, 1)

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
# print(map)

# with open(self.metric_path, "ab") as csv_file:
#     writer = csv.DictWriter(csv_file, fieldnames=self.field_names)
#     writer.writerow({
#         "epoch": epoch,
#         "mode coverage": num_mode, 
#         "KL": kl,
#         "details": map
#     })
