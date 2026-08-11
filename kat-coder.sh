#!/bin/bash
set -m   # turn on job control, even in a non-interactive script

cd ~/Workplace/llama.cpp-ling-flash/build/bin/

./llama-server -m /media/hnvcam/AI/LMStudio_Models/mradermacher/KAT-Coder-V2.5-Dev-i1-GGUF/KAT-Coder-V2.5-Dev.i1-Q4_K_M.gguf \
  -ngl 99 \
  -dev CUDA0,CUDA1 \
  -c 65535 \
  -ub 256 \
  -fa on \
  --temp 0.7 --top-p 0.95 --top-k 20 --no-ui \
  --port 1234

cd ~/Workplace/llama-cli