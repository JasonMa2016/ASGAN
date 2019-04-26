#! /bin/bash

# echo "remember to aws configure and conda install keras inside pytorch_p36 first"
source activate pytorch_p36
mkdir -p ../data

datadir="../data/ERGAN_MNIST"
# mkdir -p $datadir
# python mnist_gan.py -m 2 -sd $datadir -bs 63
# python mnist_evaluate.py -sd $datadir >> ${datadir}/results.txt
# aws s3 cp ${datadir}/Fixed_results/25.png s3://am221/ERGAN_MNIST/ergan0.png
# python mnist_gan.py -m 2 -sd $datadir -bufp 0.1
# python mnist_evaluate.py -sd $datadir >> ${datadir}/results.txt
# aws s3 cp ${datadir}/Fixed_results/25.png s3://am221/ERGAN_MNIST/ergan1.png
python mnist_gan.py -m 2 -sd $datadir -bufs 1000
python mnist_evaluate.py -sd $datadir >> ${datadir}/results.txt
aws s3 cp ${datadir}/Fixed_results/25.png s3://am221/ERGAN_MNIST/ergan2.png
python mnist_gan.py -m 2 -sd $datadir -bs 63 -bufs 1000 -bufp 0.1
python mnist_evaluate.py -sd $datadir >> ${datadir}/results.txt
aws s3 cp ${datadir}/Fixed_results/25.png s3://am221/ERGAN_MNIST/ergan3.png

python mnist_gan.py -m 2 -sd $datadir
python mnist_evaluate.py -sd $datadir >> ${datadir}/results.txt
aws s3 cp ${datadir}/Fixed_results/25.png s3://am221/ERGAN_MNIST/erganvanilla.png

# for i in {1..5}
# do
# 	python mnist_gan.py -m 2 -sd ../data/ERGAN_MNIST
# 	python mnist_evaluate.py -sd ../data/ERGAN_MNIST >> ../data/ERGAN_MNIST/results.txt
# 	aws s3 cp ../data/ERGAN_MNIST/Fixed_results/25.png "s3://am221/ERGAN_MNIST/${i}.png"
# done

# # test weighted sampling
# datadir="../data/ERGAN_MNIST_8"
# mkdir -p $datadir
# python mnist_gan.py -m 3 -sd $datadir
# python mnist_evaluate.py -sd $datadir >> $datadir/results.txt
# aws s3 cp $datadir/Fixed_results/25.png s3://am221/ERGAN_MNIST_8/weightedsampling.png