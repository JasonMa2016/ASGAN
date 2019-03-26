import numpy as np
import torch
import torch.distributions as ds
import itertools
import collections

def create_grid(batch_size, num_components=25):
    cat = ds.Categorical(torch.ones(num_components)) # class distro
    mus = [torch.FloatTensor([i,j]) for i,j in itertools.product(range(-10,7,4),range(-10,7,4))]
    s = 0.05
    sigmas = [torch.eye(2)*s**2 for i in range(num_components)]
    components = list((ds.MultivariateNormal(mu,sigma) for (mu, sigma) in zip(mus, sigmas)))

    sampled_category = cat.sample(torch.Size([batch_size]))
    data = []
    for i in sampled_category:
        sample = components[i].sample()
        data.append(sample)
    data = np.stack(data)
    # plt.scatter(x=data[:,0], y=data[:,1], s=100)
    # plt.show()
    return data

def create_ring(batch_size, num_components=8):
    cat = ds.Categorical(torch.ones(num_components)) # class distro
    mus = [torch.FloatTensor([np.cos(th),np.sin(th)]) for th in np.linspace(0, 2*np.pi, num_components, endpoint=False)]
    s = 0.05
    sigmas = [torch.eye(2)*s**2 for i in range(num_components)]
    components = list((ds.MultivariateNormal(mu,sigma) for (mu, sigma) in zip(mus, sigmas)))

    sampled_category = cat.sample(torch.Size([batch_size]))
    data = []
    for i in sampled_category:
        sample = components[i].sample()
        data.append(sample)
    data = np.stack(data)
    # plt.scatter(x=data[:,0], y=data[:,1], s=100)
    # plt.show()
    return data

def count_mode(MEANS,fake_data):
	l2_store=[]
	for x_ in fake_data:
		l2_store.append([np.sum((x_-i)**2)  for i in MEANS])

	mode=np.argmin(l2_store,1).flatten().tolist()
	dis_ = [l2_store[j][i] for j,i in enumerate(mode)]
	mode_counter = [mode[i] for i in range(len(mode)) if np.sqrt(dis_[i])<=0.15]

	mode_count = len(collections.Counter(mode_counter))
	points = np.sum(np.array(collections.Counter(mode_counter).values()))
	good_points = 0
	for i in points:
		good_points += i

	print('Number of Modes Captured: {}'.format(mode_count))
	print('Number of Points Falling Within 3 std. of the Nearest Mode {}'.format(good_points))

	return mode_count, good_points

def count_mode_grid(fake_data):
	MEANS = np.array([np.array([i, j]) for i, j in itertools.product(range(-10, 7, 4),
                                                           range(-10, 7, 4))])

	return count_mode(MEANS, fake_data)

def count_mode_ring(fake_data):
	MEANS = np.array([np.array([np.cos(th),np.sin(th)]) for th in np.linspace(0, 2*np.pi, 8, endpoint=False)])

	return count_mode(MEANS, fake_data)
