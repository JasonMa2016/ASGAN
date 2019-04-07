#! /bin/bash
# https://kevinzakka.github.io/2017/08/13/aws-pytorch/

# drivers
wget http://us.download.nvidia.com/tesla/375.66/nvidia-diag-driver-local-repo-ubuntu1604_375.66-1_amd64.deb
dpkg -i nvidia-diag-driver-local-repo-ubuntu1604_375.66-1_amd64.deb
sudo apt-get update
sudo apt-get -y install cuda-drivers
sudo apt-get update && sudo apt-get -y upgrade

# python
sudo apt-get install unzip
sudo apt-get --assume-yes install python3-tk
sudo apt-get --assume-yes install python3-pip
sudo pip3 install --upgrade pip
sudo pip3 install virtualenv numpy scipy matplotlib

# virtualenv
mkdir envs
cd envs
virtualenv --system-site-packages deepL

# pytorch
source ~/envs/deepL/bin/activate
pip install http://download.pytorch.org/whl/cu91/torch-0.4.0-cp35-cp35m-linux_x86_64.whl
pip install torchvision tqdm

sudo reboot


#### https://github.com/anihamde/cs287-s18/blob/master/Azure%20Setup%20Instruction.ipynb
bash
# wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.1.105-1_amd64.deb
# sudo dpkg -i cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_10.1.105-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda

sudo apt-get install python3-pip
sudo pip3 install http://download.pytorch.org/whl/cu90/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl
# Check now that running python3 gives you version 3.5, and that you can import torch and run torch.LongTensor([1,2,3]).cuda()
sudo pip3 install torchtext