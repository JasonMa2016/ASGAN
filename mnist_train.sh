#! /bin/bash
echo "remember to aws configure and conda install keras inside pytorch_p36 first"

source activate pytorch_p36
sudo mkdir -p ../data

# python mnist_gan.py -m 0 -sd DCGAN_MNIST &&
# python mnist_evaluate.py -sd DCGAN_MNIST > DCGAN_MNIST/results.txt &&
# python mnist_ergan.py -m 2 -sd ERGAN_MNIST &&
# aws s3 cp ./ERGAN_MNIST s3://am221 &&
# conda install keras &&
# python mnist_evaluate.py -sd ERGAN_MNIST > ERGAN_MNIST/results.txt
# python mnist_gan.py -m 2 -sd ERGAN_MNIST &&
# python mnist_evaluate.py -sd ERGAN_MNIST > ERGAN_MNIST/results.txt # -halt
# oops! on 4/11 i accidentally ran mnist_ergan with alpha smoothing too!

# TODO: figure out the way to avoid repeating data dir
for i in {1..5}
do
	python mnist_gan.py -m 2 -sd ../data/ERGAN_MNIST_1 &&
	python mnist_evaluate.py -sd ../data/ERGAN_MNIST_1 >> ../data/ERGAN_MNIST_1/results.txt
	aws s3 cp ../data/ERGAN_MNIST_1 s3://am221
done

# might need :set ff=unix (if \r causing problems)
# j=stringystring; echo message > $j.txt