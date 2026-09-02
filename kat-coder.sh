#!/bin/bash
set -m   # turn on job control, even in a non-interactive script

# No -sm tensor here, and no -ot either.
#
# -sm tensor looks right on paper (both cards work on every block instead of
# taking turns) but it cannot split THIS model. llama.cpp rounds every tensor
# cut down to a granularity, and KAT-Coder's expert tensors are only 512
# elements wide on the split axis with a granularity of 256 -- so the only
# reachable cuts are 0, 50% and 100%. Worse, the rotation in
# llama_meta_device_get_split_state alternates which card takes the rounding
# remainder, so half the blocks land 50/50 and the other half land 100/0 on
# CUDA0. Whatever -ts says, CUDA0 ends up with ~75% of 18 GiB of experts and
# OOMs. Measured at -ts 0.6828,0.3172:
#     load_tensors: Meta() model buffer size = 14901.13 MiB   (CUDA0's share)
# against a 14,815 MiB budget, before a single KV cell is allocated.
#
# With -ctk/-ctv q8_0 and -ub 256 the model + 58k of KV fits across the two
# cards on a plain layer split, so llama.cpp's own placement is enough:
#   CUDA0 14,905 MiB of 16,303   CUDA1 7,209 MiB of 8,188   (predicted)
#   CUDA0 14,656 MiB             CUDA1 7,036 MiB            (measured, loaded)
# Re-check with:  ./tensor-override.py <gguf> -q8 -ub 256 -c 59392 --measure

cd ~/Workplace/llama.cpp/build/bin/

./llama-server -m /media/hnvcam/AI/LMStudio_Models/mradermacher/KAT-Coder-V2.5-Dev-i1-GGUF/KAT-Coder-V2.5-Dev.i1-Q4_K_M.gguf \
  -ngl 99 \
  -dev CUDA0,CUDA1 \
  -c 131072 \
  -ub 256 \
  -np 2 \
  -kvu \
  -ctk q8_0 \
  -ctv q8_0 \
  -fa on \
  --temp 0.7 --top-p 0.95 --top-k 20 \
  --host 0.0.0.0 --port 1234

cd ~/Workplace/llama-cli
