# Deep Learning Image Classifier for Food

Currently there are over 170,000,000 food photos on social media sites but searching and discovering data is still a manual process via tags. The aim of the project is to investigate image classification using convolutional neural networks as a means to automatically tag food within images.

## Setup 

### Hardware Requirements

The experiments ran on AWS with the following specifications:

* Deep Learning AMI with Conda (Ubuntu) (ami-1812bb61)
* p2.xlarge (https://aws.amazon.com/ec2/instance-types/p2/) 
  * 1 GPU
  * i7 4 Cores
  * 61 GB RAM
  * 100GB EBS Volume

### Instructions

Create the EC2 instance on AWS and save the private key that they provide.
SSH onto the instance and run the following commands
```
# Install Anaconda 2
wget https://repo.continuum.io/archive/Anaconda2-5.0.1-Linux-x86_64.sh
./Anaconda2-5.0.1-Linux-x86_64.sh
#  Change the conda path to Anaconda2
vim ~/.bashrc
#  Reload profile
source ~/.bashrc

#  Installgit
sudo apt install git
mkdir workspace
cd workspace
git clone https://github.com/winghymliu/deeplearning-foodclassifier.git
cd ..

#  Download dataset
sudo apt-get install zip
wget http://foodcam.mobi/dataset256.zip
unzip dataset256.zip -d ./workspace/deeplearning-foodclassifier
cd ./workspace/deeplearning-foodclassifier

#  Setup environment
conda env create -f requirements/food-classifier-linux.yml
source activate food-classifier
KERAS_BACKEND=tensorflow python -c "from keras import backend"
pip install -r ./requirements/requirements.txt

#  Setup jupyter notebook to run remotely
ipython
from IPython.lib import passwd
passwd('CHOOSE A PASSWORD')
#  Copy sha hash and exit
exit()

jupyter notebook --generate-config
mkdir certs
cd certs
sudo openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout mycert.pem -out mycert.pem

vim ~/.jupyter/jupyter_notebook_config.py
#  Paste the following

c = get_config()
c.IPKernelApp.pylab = 'inline' 
c.NotebookApp.certfile = u'/home/ubuntu/certs/mycert.pem' 
c.NotebookApp.ip = '*' 
c.NotebookApp.open_browser = False

#  Save and exit

#  Almost there, edit the ssh config to avoid timeouts 
vim ~/.ssh/config
#  paste the following
Host *
ServerAliveInterval 60
#  save and exit

#  Finally run the thing!
jupyter notebook MachineLearningNanodegreeCaptstone-FoodClassifier.ipynb
#  Open the URL Firefox and add exception, then enter your password
https://[YOUR EC2 DNS]:8888
```
