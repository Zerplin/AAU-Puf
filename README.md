# Impersonation of IoT devices through AI hacking

https://towardsdatascience.com/neural-network-based-attack-methods-to-break-the-large-xor-pufs-faebabba1c5a

## Set-Up NVIDIA drivers on claaudia

https://www.server-world.info/en/note?os=Ubuntu_22.04&p=nvidia

```shell
ssh ubuntu@claaudia.jupiops.net
```

```shell
sudo adduser jupiops
sudo usermod -aG sudo jupiops
sudo cp -r .ssh ../jupiops/
su jupiops
cd
sudo chown -R jupiops:jupiops .
chmod 700 ~/.ssh
chmod 644 ~/.ssh/authorized_keys
exit
exit
```

```shell
ssh jupiops@claaudia.jupiops.net
```

```shell
sudo deluser --remove-home ubuntu
sudo apt update
# apt search nvidia-driver
sudo apt install nvidia-driver-515-server
sudo shutdown -r now
```

```shell
nvidia-smi
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
# sudo dpkg -i cuda-keyring_1.0-1_all.deb
# rm cuda-keyring_1.0-1_all.deb
# sudo apt update
sudo apt install nvidia-cuda-toolkit 
# apt search cuda-toolkit
# sudo apt install cuda-toolkit-11-7
# export PATH=/usr/local/cuda-12.0/bin${PATH:+:${PATH}}
# export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib\
#                          ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
nvcc --version
sudo shutdown -r now
```

```shell
# apt search libcudnn8
# sudo apt install libcudnn8=${cudnn_version}-1+cuda11.8
# sudo apt install libcudnn8-dev=${cudnn_version}-1+cuda11.8
```

```shell
# sudo apt install python3-pip python3-dev python3-venv gcc g++ make
sudo apt install nvidia-cudnn python3-pip python3-dev python3-venv gcc g++ make
sudo shutdown -r now
```

```shell
python3 -m venv --system-site-packages ~/tensorflow
source ~/tensorflow/bin/activate
pip3 install --upgrade tensorflow
python3 -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```