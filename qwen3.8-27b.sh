#!/bin/bash
set -m   # turn on job control, even in a non-interactive script

cd ~/Workplace/llama.cpp/build/bin/

# -ts 0.7197,0.2803 puts blocks 0-47 on CUDA0 and 48-64 + output.weight on
# CUDA1. Without it llama.cpp splits by free VRAM and overfills the 4060: at
# -c 100000 that leaves the 4060 with 197 MiB free and the 5070 Ti with 2659.
#
# This is the LONG-CONTEXT config, MTP off. qwen35 does support MTP -- the
# 'unused tensor blk.64.nextn.*' warnings only appear because --spec-type
# draft-mtp is absent (qwen35.cpp:41 skips those tensors unless load_mtp).
# Turning it on costs ~58k tokens of context, mostly NOT the nextn block
# (335 MiB) but the recurrent state: --draft-max defaults to 3, so llama.cpp
# keeps 1+3 rollback copies of the SSM state and 598 MiB becomes 2394 MiB.
#
# MTP variant, verified to load (837 MiB free on CUDA0, 395 on CUDA1):
#     -ts 0.75,0.25 -c 73728 --spec-type draft-mtp
#
# Regenerate: ./tensor-override.py <model.gguf> -q8 --no-mtp   (drop --no-mtp
# for the MTP variant; MTP is on by default whenever the GGUF ships nextn).

./llama-server -m /media/hnvcam/AI/LLAMA_Models/Qwen3.8-27B-UD-Q4_K_M.gguf \
    -ngl 99 \
    -dev CUDA0,CUDA1 \
    -ts 0.75,0.25 \
    -c 86016 \
    -ub 256 \
    -ctk q8_0 \
    -ctv q8_0 \
    --spec-type draft-mtp \
    -fa on \
  --temp 1.0 --top-p 0.95 --top-k 20 \
  --port 1234

cd ~/Workplace/llama-cli