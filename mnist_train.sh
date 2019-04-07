python mnist_gan.py -mt 0 -sd DCGAN_MNIST &&
python mnist_evaluate.py -sd DCGAN_MNIST &&
python mnist_gan.py -mt 1 -sd ASGAN_MNIST &&
python mnist_evaluate.py -sd ASGAN_MNIST &&
python mnist_gan.py -mt 2 -sd ERGAN_MNIST &&
python mnist_evaluate.py -sd ERGAN_MNIST # -halt