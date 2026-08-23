#!/bin/bash
set -m   # turn on job control, even in a non-interactive script

cd ~/Workplace/llama.cpp-longcat-flash/build/bin/

./llama-server -m /media/hnvcam/AI/LLAMA_Models/LongCat-Flash-Lite-uncensored-heretic-Native-MTP-Preserved-Q4_K_M.gguf \
  -ngl 99 \
  -dev CUDA0,CUDA1 \
  -ts 1,0 \
  -c 131072 \
  -ub 512 \
  -ot 'blk\.(10|12|14|16|18)\.(ffn_down_exps\.weight|ffn_gate_exps\.weight|ffn_up_exps\.weight)$=CUDA1,blk\.(20|22|24|26)\.(ffn_down_exps\.weight|ffn_gate_exps\.weight|ffn_up_exps\.weight)$=CPU,^ngram_embd\.=CPU' \
  -fa on \
  --temp 0.7 --top-p 0.95 --top-k 40 --repeat-penalty 1.06 \
  --jinja --chat-template-file ~/Workplace/llama-cli/longcat-lite.jinja \
  -rea off \
  --host 0.0.0.0 --port 1234

cd ~/Workplace/llama-cli