#!/bin/bash
set -m   # turn on job control, even in a non-interactive script

cd ~/Workplace/llama.cpp-ling-flash/build/bin/

./llama-server -m /media/hnvcam/AI/LLAMA_Models/Ling-3.0-flash-MXFP4_MOE.gguf \
  -ngl 99 \
  -dev CUDA0,CUDA1 \
  -ts 1,0 \
  -c 65535 \
  -ub 512 \
  -ot 'blk\.(8|9|10|11)\.(ffn_down_exps\.weight|ffn_gate_exps\.weight|ffn_up_exps\.weight)$=CUDA1,blk\.(12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41)\.(ffn_down_exps\.weight|ffn_gate_exps\.weight|ffn_up_exps\.weight)$=CPU' \
  -fa on \
  --temp 0.6 --top-p 0.95 --top-k 20 --no-ui \
  --port 1234

cd ~/Workplace/llama-cli