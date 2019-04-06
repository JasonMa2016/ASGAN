# mnist is 70k. pacgan stacked mnist is 128k

# load digits. do they come with classes? or pre-trained 3rd party classifier (pacgan)
# TODO: number of modes captured, KL div between generated distro over modes and expected (uniform) one

import gzip
import numpy as np
import matplotlib.pyplot as plt

def load_images():
    """
    Loads the training images from train-images-idx3-ubyte.gz.
    Make sure it's located in the data folder one level back!
    """
    hw = 28 ** 2  # Number of pixels per image
    n = 60000     # Number of images
    with gzip.open('../data/train-images-idx3-ubyte.gz', 'r') as f:
        f.read(16)
        buffer = f.read(hw * n)
        images = np.frombuffer(buffer, dtype=np.uint8)
        images = images.reshape(n, hw)
    return images


def load_labels():
    """
    Loads the training labels from train-labels-idx1-ubyte.gz.
    Make sure it's located in the data folder one level back!
    """
    n = 60000     # Number of images
    with gzip.open('../data/train-labels-idx1-ubyte.gz', 'r') as f:
        f.read(8)
        buffer = f.read(n)
        labels = np.frombuffer(buffer, dtype=np.uint8)
        labels = labels.reshape(n)
    return labels

def load_mnist(num_training_sample):
    img = load_images().reshape(-1, 28, 28) # 60k, 28, 28
    lab = load_labels() # 60k
    ids = np.random.randint(0, img.shape[0], size=(num_training_sample, 3)) # 25600, 3
    img_1 = np.zeros(shape=(ids.shape[0], 28, 28, ids.shape[1])).astype(int) # 25600, 28, 28, 3
    lab_1 = np.zeros(ids.shape) # 25600, 3
    for i in range(ids.shape[0]):
        for j in range(ids.shape[1]):
            img_1[i, :, :, j] = img[ids[i, j], :, :]
            lab_1[i, j] = lab[ids[i, j]]
    return img_1, lab_1

# sample image
if __name__ == "__main__":
    img_1, lab_1 = load_mnist(128000)
    plt.imshow(img_1[42])
    plt.show()
    print('label of image', lab_1[42])