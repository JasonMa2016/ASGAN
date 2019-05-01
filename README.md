# Memory-based Techniques for Stabilizing GAN Code Repository

## Prerequisites

- Python 3.6+
- Pytorch 
- Keras
- Numpy
- Scipy

- AWS machine (GPU preferred, i.e. p3.2xlarge instance)

## Usage
Set up AWS S3 storage credential:
	$ aws configure

Activate Pytorch environment and run scripts:

	$ source activate pytorch_p36
	$ conda intall keras
	$ sh mnist_train.sh

## Note
You may have to modify the bash script to make the referenced directory match the directory name on your S3 storage. Default: AM221
