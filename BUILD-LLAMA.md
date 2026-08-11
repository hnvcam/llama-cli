# 1. Install cuda-toolkit:
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-0
```
# 2. Build with Ninja:
```
cmake -B build -G Ninja -DGGML_CUDA=ON
cmake --build build --config Release -j10
```
