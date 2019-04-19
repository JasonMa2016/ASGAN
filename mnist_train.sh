source activate pytorch_p36
conda install keras
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

for i in {1..5}
do
	python mnist_ergan.py -m 2 -sd ERGAN_MNIST_0 &&
	python mnist_evaluate.py -sd ERGAN_MNIST_0 >> ERGAN_MNIST_0/results.txt
done