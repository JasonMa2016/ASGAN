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
python mnist_gan.py -m 3 -sd $datadir
python mnist_evaluate.py -sd $datadir >> $datadir/results.txt
aws s3 cp $datadir/Fixed_results/25.png s3://am221/ERGAN_MNIST_8/weightedsampling.png



# might need :set ff=unix (if \r causing problems)
# j=stringystring; echo message > $j.txt
# python mnist_gan.py -m 0 -sd DCGAN_MNIST &&
# python mnist_evaluate.py -sd DCGAN_MNIST > DCGAN_MNIST/results.txt &&
# python mnist_ergan.py -m 2 -sd ERGAN_MNIST &&
# aws s3 cp ./ERGAN_MNIST s3://am221 &&
# python mnist_evaluate.py -sd ERGAN_MNIST > ERGAN_MNIST/results.txt
# python mnist_gan.py -m 2 -sd ERGAN_MNIST &&
# python mnist_evaluate.py -sd ERGAN_MNIST > ERGAN_MNIST/results.txt # -halt
# oops! on 4/11 i accidentally ran mnist_ergan with alpha smoothing too!
# for i in {1..5}
# do
# 	python mnist_gan.py -m 2 -sd ../data/ERGAN_MNIST_1 &&
# 	python mnist_evaluate.py -sd ../data/ERGAN_MNIST_1 >> ../data/ERGAN_MNIST_1/results.txt
# 	aws s3 cp ../data/ERGAN_MNIST_1 s3://am221
# done

# mkdir -p ../data/DCGAN_MNIST
# mkdir -p ../data/ERGAN_MNIST
# mkdir -p ../data/ASGAN_MNIST_1
# mkdir -p ../data/ASGAN_MNIST_2
# mkdir -p ../data/ERGAN_MNIST_8

# run dcgan and ergan 5 times each
# for i in {1..5}
# do
# 	python mnist_gan.py -sd ../data/DCGAN_MNIST
# 	python mnist_evaluate.py -sd ../data/DCGAN_MNIST >> ../data/DCGAN_MNIST/results.txt
# 	aws s3 cp ../data/DCGAN_MNIST/Fixed_results/25.png "s3://am221/DCGAN_MNIST/${i}.png"
# done

# for i in {1..5}
# do
# 	python mnist_gan.py -m 2 -sd ../data/ERGAN_MNIST
# 	python mnist_evaluate.py -sd ../data/ERGAN_MNIST >> ../data/ERGAN_MNIST/results.txt
# 	aws s3 cp ../data/ERGAN_MNIST/Fixed_results/25.png "s3://am221/ERGAN_MNIST/${i}.png"
# done

# test alphagan for different alpha (.1,.2,.4,.6,.8)
# python mnist_gan.py -m 1 -sd ../data/ASGAN_MNIST_1 -t 0.1
# echo "alpha=0.1" >> ../data/ASGAN_MNIST_1/results.txt
# python mnist_evaluate.py -sd ../data/ASGAN_MNIST_1 >> ../data/ASGAN_MNIST_1/results.txt
# aws s3 cp ../data/ASGAN_MNIST_1/Fixed_results/25.png s3://am221/ASGAN_MNIST_1/point1.png
# python mnist_gan.py -m 1 -sd ../data/ASGAN_MNIST_1 -t 0.2
# echo "alpha=0.2" >> ../data/ASGAN_MNIST_1/results.txt
# python mnist_evaluate.py -sd ../data/ASGAN_MNIST_1 >> ../data/ASGAN_MNIST_1/results.txt
# aws s3 cp ../data/ASGAN_MNIST_1/Fixed_results/25.png s3://am221/ASGAN_MNIST_1/point2.png
# python mnist_gan.py -m 1 -sd ../data/ASGAN_MNIST_1 -t 0.4
# echo "alpha=0.4" >> ../data/ASGAN_MNIST_1/results.txt
# python mnist_evaluate.py -sd ../data/ASGAN_MNIST_1 >> ../data/ASGAN_MNIST_1/results.txt
# aws s3 cp ../data/ASGAN_MNIST_1/Fixed_results/25.png s3://am221/ASGAN_MNIST_1/point4.png
# python mnist_gan.py -m 1 -sd ../data/ASGAN_MNIST_1 -t 0.6
# echo "alpha=0.6" >> ../data/ASGAN_MNIST_1/results.txt
# python mnist_evaluate.py -sd ../data/ASGAN_MNIST_1 >> ../data/ASGAN_MNIST_1/results.txt
# aws s3 cp ../data/ASGAN_MNIST_1/Fixed_results/25.png s3://am221/ASGAN_MNIST_1/point6.png
# python mnist_gan.py -m 1 -sd ../data/ASGAN_MNIST_1 -t 0.8
# echo "alpha=0.8" >> ../data/ASGAN_MNIST_1/results.txt
# python mnist_evaluate.py -sd ../data/ASGAN_MNIST_1 >> ../data/ASGAN_MNIST_1/results.txt
# aws s3 cp ../data/ASGAN_MNIST_1/Fixed_results/25.png s3://am221/ASGAN_MNIST_1/point8.png

# test asgan_temp
# python mnist_asgan_temp.py -m 1 -sd ../data/ASGAN_MNIST_2
# python mnist_evaluate.py -sd ../data/ASGAN_MNIST_2 >> ../data/ASGAN_MNIST_2/results.txt
# aws s3 cp ../data/ASGAN_MNIST_2/Fixed_results/25.png s3://am221/ASGAN_MNIST_2/3epochseven.png

# test weighted sampling
# python mnist_gan.py -m 3 -sd ../data/ERGAN_MNIST_8
# python mnist_evaluate.py -sd ../data/ERGAN_MNIST_8 >> ../data/ERGAN_MNIST_8/results.txt
# aws s3 cp ../data/ERGAN_MNIST_8/Fixed_results/25.png s3://am221/ERGAN_MNIST_8/weightedsampling.png

# aws s3 cp ../data/*/results.txt s3://am221
