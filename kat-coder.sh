#!/bin/bash
set -m   # turn on job control, even in a non-interactive script

cd ~/Workplace/llama.cpp-ling-flash/build/bin/

./llama-server -m /media/hnvcam/AI/LMStudio_Models/mradermacher/KAT-Coder-V2.5-Dev-i1-GGUF/KAT-Coder-V2.5-Dev.i1-Q4_K_M.gguf \
  -ngl 99 \
  -dev CUDA0,CUDA1 \
  -ts 1,0 \
  -c 131072 \
  -ub 512 \
  -ot 'blk\.(23|24|25|26|27|28|29|30|31|32|33|34|35|36|37)\.(ffn_down_exps\.weight|ffn_gate_exps\.weight|ffn_up_exps\.weight)$=CUDA1,^token_embd\.weight$=CUDA1,blk\.(38|39)\.(ffn_down_exps\.weight|ffn_gate_exps\.weight|ffn_up_exps\.weight)$=CPU' \
  -fa on \
  --temp 0.7 --top-p 0.95 --top-k 20 --no-ui \
  --host 0.0.0.0 --port 1234

cd ~/Workplace/llama-cli