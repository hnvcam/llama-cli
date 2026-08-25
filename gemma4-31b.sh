#!/bin/bash
set -m   # turn on job control, even in a non-interactive script

cd ~/Workplace/llama.cpp/build/bin/

# gemma4-31B is DENSE (60 blocks, 17.05 GiB of Q4_K_M weights), so every weight
# is read on every token and token generation is pure memory bandwidth. The two
# cards are wildly mismatched: CUDA0 = 5070 Ti (~896 GB/s, 16 GB, PCIe 4.0 x16),
# CUDA1 = 4060 (~272 GB/s, 8 GB, PCIe 3.0 x8).
#
# -sm tensor is what makes this fast. The default -sm layer is a PIPELINE: the
# cards take turns, so a token costs W0/BW0 + W1/BW1 and the 4060 ends up eating
# ~60% of the time for ~33% of the weights. -sm tensor splits each tensor across
# both cards so they run CONCURRENTLY -- the cost becomes max() instead of sum().
# Measured on this box (llama-bench tg128, b10566):
#     -sm layer  -ts 0.68,0.32   23.9 t/s   <- what this script used to do
#     -sm layer  -ts 0.86,0.14   30.1 t/s
#     -sm tensor -ts 0.72,0.28   35.5 t/s
#     -sm tensor -ts 0.78,0.22   41.0 t/s   <- speed optimum, does NOT fit
# In-server (32k, q8 KV, real request): 23.0 t/s before -> 36.2 t/s now.
#
# -sm row is NOT an option: the CUDA backend no longer implements split buffers
# at all (only SYCL exports ggml_backend_split_buffer_type), so it dies with
# "device CUDA0 does not support split buffers" -- see llama-model.cpp:1005.
#
# On -ts: under -sm tensor the KV cache splits by the SAME ratio as the weights,
# so -ts is capped by VRAM, not by speed. 0.78 is the speed optimum but OOMs on
# CUDA0; 0.765 also OOMs; 0.75 fits with ~800 MiB spare. If you lower -c or drop
# to -ctv q4_0 you free CUDA0 and can walk -ts back up toward 0.78.
#
# -np 1: llama-server defaults to 4 slots, and the compute buffer carries an
# n_kv * n_ubatch * 4 * n_seq KQ mask, so 4 slots cost ~100 MiB per card for
# nothing if you only ever run one conversation. Dropping to 1 slot is what
# bought the headroom to go from -ts 0.68 to 0.75.
#
# Regenerate:  ./tensor-override.py <model.gguf> -q8 -c 32768 -ub 512 --parallel 1
# (omit -c to be told the largest context that still fits). It now plans -sm
# tensor directly and picks -ts from gpu-bandwidth.json; it lands on 0.731 here,
# a little under the 0.75 below because its per-card reserve is deliberately
# generous. Both load.

./llama-server -m /media/hnvcam/AI/LLAMA_Models/gemma-4-31B-it-Q4_K_M.gguf \
    -ngl 99 \
    -dev CUDA0,CUDA1 \
    -sm tensor \
    -ts 0.75,0.25 \
    -c 32768 \
    -np 1 \
    -ub 512 \
    -ctk q8_0 \
    -ctv q8_0 \
    -fa on \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --port 1234

cd ~/Workplace/llama-cli
