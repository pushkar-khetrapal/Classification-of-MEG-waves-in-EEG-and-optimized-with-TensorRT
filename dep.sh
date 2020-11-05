pip install pillow matplotlib
pip install tensorflow-gpu==2.0.0
wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
dpkg -i nvidia-machine-learning-repo-*.deb
apt-get update
sudo apt-get install libnvinfer5
dpkg -l | grep nvinfer