#! /bin/bash

# echo "remember to aws configure and conda install keras inside pytorch_p36 first"
source activate pytorch_p36
mkdir -p ../data

for i in {1..5}
do
	python mnist_gan.py -m 2 -sd ../data/ERGAN_MNIST
	python mnist_evaluate.py -sd ../data/ERGAN_MNIST >> ../data/ERGAN_MNIST/results.txt
	aws s3 cp ../data/ERGAN_MNIST/Fixed_results/25.png "s3://am221/ERGAN_MNIST/${i}.png"
done

# test weighted sampling
python mnist_gan.py -m 3 -sd ../data/ERGAN_MNIST_8
python mnist_evaluate.py -sd ../data/ERGAN_MNIST_8 >> ../data/ERGAN_MNIST_8/results.txt
aws s3 cp ../data/ERGAN_MNIST_8/Fixed_results/25.png s3://am221/ERGAN_MNIST_8/weightedsampling.png