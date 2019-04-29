#! /bin/bash

# echo "remember to aws configure and conda install keras inside pytorch_p36 first"
source activate pytorch_p36
mkdir -p ../data

# copy Fixed_results and train_hist for training examination, and results.txt

# orig dcgan 2 times

# asgan 0.05 0.1 0.2 0.4 0.6 0.8

# asgan 3 epochs .5 .3 .2

# ergan 0.2 buf prop, 0.7 buf prop

# ergan 1000 buf size

# ergan weighted

# ergan after 5 epochs

# ergan after 5 epochs and selective sampling

# asergan

# todo: er within pacgan

# todo: i want to show that alpha smoothing actually smooths the training. contrast the histograms or images per epoch?
# and i want to show the speed/robustness to mode collapse tradeoff