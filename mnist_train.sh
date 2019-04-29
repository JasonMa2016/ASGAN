#! /bin/bash

# echo "remember to aws configure and conda install keras/tensorflow inside pytorch_p36 first"
source activate pytorch_p36
pip install tensorflow_gan
mkdir -p ../data
bucketname="am221"
# copy Fixed_results and train_hist for training examination, and results.txt

# orig dcgan 2 times
datadir="DCGAN_MNIST"
for i in {1..2}
do
	python mnist_gan.py -sd ../data/${datadir} &&
	python mnist_evaluate.py -sd ../data/${datadir} >> ../data/${datadir}/results.txt
	python check_fid.py -sd ../data/${datadir} >> ../data/${datadir}/results.txt
	aws s3 sync ../data/${datadir}/Fixed_results s3://${bucketname}/${datadir}
done

# asgan 0.05 0.1 0.2 0.4 0.6 0.8
datadir="ASGAN_MNIST_pt05"
python mnist_gan.py -m 1 -sd ../data/${datadir} -t 0.05 &&
python mnist_evaluate.py -sd ../data/${datadir} >> ../data/${datadir}/results.txt
python check_fid.py -sd ../data/${datadir} >> ../data/${datadir}/results.txt
aws s3 sync ../data/${datadir}/Fixed_results s3://${bucketname}/${datadir}
datadir="ASGAN_MNIST_pt1"
python mnist_gan.py -m 1 -sd ../data/${datadir} -t 0.1 &&
python mnist_evaluate.py -sd ../data/${datadir} >> ../data/${datadir}/results.txt
python check_fid.py -sd ../data/${datadir} >> ../data/${datadir}/results.txt
aws s3 sync ../data/${datadir}/Fixed_results s3://${bucketname}/${datadir}
datadir="ASGAN_MNIST_pt2"
python mnist_gan.py -m 1 -sd ../data/${datadir} -t 0.2 &&
python mnist_evaluate.py -sd ../data/${datadir} >> ../data/${datadir}/results.txt
python check_fid.py -sd ../data/${datadir} >> ../data/${datadir}/results.txt
aws s3 sync ../data/${datadir}/Fixed_results s3://${bucketname}/${datadir}
datadir="ASGAN_MNIST_pt4"
python mnist_gan.py -m 1 -sd ../data/${datadir} -t 0.4 &&
python mnist_evaluate.py -sd ../data/${datadir} >> ../data/${datadir}/results.txt
python check_fid.py -sd ../data/${datadir} >> ../data/${datadir}/results.txt
aws s3 sync ../data/${datadir}/Fixed_results s3://${bucketname}/${datadir}
datadir="ASGAN_MNIST_pt6"
python mnist_gan.py -m 1 -sd ../data/${datadir} -t 0.6 &&
python mnist_evaluate.py -sd ../data/${datadir} >> ../data/${datadir}/results.txt
python check_fid.py -sd ../data/${datadir} >> ../data/${datadir}/results.txt
aws s3 sync ../data/${datadir}/Fixed_results s3://${bucketname}/${datadir}
datadir="ASGAN_MNIST_pt8"
python mnist_gan.py -m 1 -sd ../data/${datadir} -t 0.8 &&
python mnist_evaluate.py -sd ../data/${datadir} >> ../data/${datadir}/results.txt
python check_fid.py -sd ../data/${datadir} >> ../data/${datadir}/results.txt
aws s3 sync ../data/${datadir}/Fixed_results s3://${bucketname}/${datadir}

# asgan 3 epochs .5 .3 .2
datadir="ASGAN_MNIST_3epochs"
python mnist_asgan_temp.py -m 1 -sd ../data/${datadir} &&
python mnist_evaluate.py -sd ../data/${datadir} >> ../data/${datadir}/results.txt
python check_fid.py -sd ../data/${datadir} >> ../data/${datadir}/results.txt
aws s3 sync ../data/${datadir}/Fixed_results s3://${bucketname}/${datadir}

# ergan 0.2 buf prop, 0.7 buf prop
datadir="ERGAN_MNIST_pt2bufp"
python mnist_gan.py -m 2 -sd ../data/$datadir -bufp 0.2 &&
python mnist_evaluate.py -sd ../data/$datadir >> ../data/${datadir}/results.txt
python check_fid.py -sd ../data/${datadir} >> ../data/${datadir}/results.txt
aws s3 sync ../data/${datadir}/Fixed_results s3://${bucketname}/${datadir}
datadir="ERGAN_MNIST_pt7bufp"
python mnist_gan.py -m 2 -sd ../data/$datadir -bufp 0.7 &&
python mnist_evaluate.py -sd ../data/$datadir >> ../data/${datadir}/results.txt
python check_fid.py -sd ../data/${datadir} >> ../data/${datadir}/results.txt
aws s3 sync ../data/${datadir}/Fixed_results s3://${bucketname}/${datadir}

# ergan 1000 buf size
datadir="ERGAN_MNIST_1000bufs"
python mnist_gan.py -m 2 -sd ../data/$datadir -bufs 0.2 &&
python mnist_evaluate.py -sd ../data/$datadir >> ../data/${datadir}/results.txt
python check_fid.py -sd ../data/${datadir} >> ../data/${datadir}/results.txt
aws s3 sync ../data/${datadir}/Fixed_results s3://${bucketname}/${datadir}

# ergan weighted
datadir="ERGAN_MNIST_weighted"
python mnist_gan.py -m 3 -sd ../data/$datadir -bufp 0.2 &&
python mnist_evaluate.py -sd ../data/$datadir >> ../data/${datadir}/results.txt
python check_fid.py -sd ../data/${datadir} >> ../data/${datadir}/results.txt
aws s3 sync ../data/${datadir}/Fixed_results s3://${bucketname}/${datadir}

# ergan after 5 epochs
datadir="ERGAN_MNIST_5rse"
python mnist_gan.py -m 2 -sd ../data/$datadir -rse 5 &&
python mnist_evaluate.py -sd ../data/$datadir >> ../data/${datadir}/results.txt
python check_fid.py -sd ../data/${datadir} >> ../data/${datadir}/results.txt
aws s3 sync ../data/${datadir}/Fixed_results s3://${bucketname}/${datadir}

# ergan selective sampling
datadir="ERGAN_MNIST_se"
python mnist_gan.py -m 2 -sd ../data/$datadir -se 0.3 &&
python mnist_evaluate.py -sd ../data/$datadir >> ../data/${datadir}/results.txt
python check_fid.py -sd ../data/${datadir} >> ../data/${datadir}/results.txt
aws s3 sync ../data/${datadir}/Fixed_results s3://${bucketname}/${datadir}

# todo: asergan

# todo: er within pacgan

# todo: i want to show that alpha smoothing actually smooths the training. contrast the histograms or images per epoch?
# and i want to show the speed/robustness to mode collapse tradeoff