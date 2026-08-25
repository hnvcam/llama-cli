# Build llama.cpp (2x CUDA: 5070 Ti + 4060)

## 1. CUDA toolkit

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-0
```

## 2. NCCL

```
sudo apt-get -y install libnccl2 libnccl-dev
```

> Required for `-sm tensor` speed. `GGML_CUDA_NCCL` is already ON by default -- the library just has to be installed BEFORE you configure.

## 3. Build

```
cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j10
```

> Don't use `--config Release` -- Ninja ignores it; `-DCMAKE_BUILD_TYPE` is the one that works.

## 4. Verify NCCL got linked

```
ldd build/bin/libggml-cuda.so | grep nccl
```

> Want `libnccl.so.2 => ...`. Empty means no NCCL: `rm -rf build` and redo step 3.

## 5. Benchmark

```
cd build/bin
./llama-bench -m <model.gguf> -ngl 99 -dev CUDA0/CUDA1 -fa 1 -p 0 -n 128 -r 3 \
  -sm tensor -ts "0.75/0.25"
```

> Pre-NCCL baseline on gemma-4-31B Q4_K_M: 41.0 t/s (`-sm tensor`) vs 23.9 t/s (`-sm layer`).
