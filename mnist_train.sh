# python mnist_gan.py -m 0 -sd DCGAN_MNIST &&
# python mnist_evaluate.py -sd DCGAN_MNIST > DCGAN_MNIST/results.txt &&
python mnist_gan.py -m 1 -sd ASGAN_MNIST &&
python mnist_evaluate.py -sd ASGAN_MNIST > ASGAN_MNIST/results.txt &&
aws s3 cp ./ASGAN_MNIST s3://am221 
# python mnist_gan.py -m 2 -sd ERGAN_MNIST &&
# python mnist_evaluate.py -sd ERGAN_MNIST > ERGAN_MNIST/results.txt # -halt

git status
git add --all
git commit -am "asgan mnist train"
git push
