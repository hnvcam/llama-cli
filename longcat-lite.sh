#!/bin/bash
set -m   # turn on job control, even in a non-interactive script

cd ~/Workplace/llama.cpp-longcat-flash/build/bin/

./llama-server -m /media/hnvcam/AI/LLAMA_Models/LongCat-Flash-Lite-uncensored-heretic-Native-MTP-Preserved-Q4_K_M.gguf \
  -ngl 99 \
  -dev CUDA0,CUDA1 \
  -ts 1,0 \
  -c 65535 \
  -ub 512 \
  -ot 'blk\.(14|16|18|20|22)\.(ffn_down_exps\.weight|ffn_gate_exps\.weight|ffn_up_exps\.weight)$=CUDA1,blk\.(24|26)\.(ffn_down_exps\.weight|ffn_gate_exps\.weight|ffn_up_exps\.weight)$=CPU,^ngram_embd\.=CPU' \
  -fa on \
  --temp 0.7 --top-p 0.95 --top-k 40 --repeat-penalty 1.06 \
  --jinja --chat-template-file ~/Workplace/llama-cli/longcat-lite.jinja \
  -rea off \
  --reasoning-budget 8192 \
  --reasoning-budget-message $'\n\nI have thought long enough, let me answer now.' \
  --port 1234

cd ~/Workplace/llama-cli