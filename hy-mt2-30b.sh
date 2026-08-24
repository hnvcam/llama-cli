#!/bin/bash
set -m   # turn on job control, even in a non-interactive script

cd ~/Workplace/llama.cpp-ling-flash/build/bin/

./llama-server -m /media/hnvcam/AI/LLAMA_Models/Hy-MT2-30B-A3B.i1-Q5_K_M.gguf \
  -ngl 99 \
  -dev CUDA0,CUDA1 \
  -ts 1,0 \
  -c 65535 \
  -ub 512 \
  -ot 'blk\.(19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36)\.(ffn_down_exps\.weight|ffn_gate_exps\.weight|ffn_up_exps\.weight)$=CUDA1,blk\.(37|38|39|40|41|42|43|44|45|46|47)\.(ffn_down_exps\.weight|ffn_gate_exps\.weight|ffn_up_exps\.weight)$=CPU,^token_embd\.weight$=CPU' \
  -fa on \
  --temp 0.7 --top-p 1 --top-k -1 --repeat-penalty 1.0 \
  --host 0.0.0.0 --port 1234

cd ~/Workplace/llama-cli