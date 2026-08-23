#!/bin/bash
set -m   # turn on job control, even in a non-interactive script

cd ~/Workplace/llama.cpp/build/bin/

# -ts 1,0 pins every layer -- and therefore the whole KV cache -- to CUDA0;
# -ot then pushes expert weights out to CUDA1 and CPU. blk 0-5 keep their
# experts on CUDA0, 6-9 go to CUDA1, 10-35 to CPU. token_embd is read only
# through ggml_get_rows, so evicting it costs microseconds per token and buys
# 584 MiB; llama.cpp keeps it on the CPU buffer for this arch anyway.
#
# gpt-oss is a SLIDING-WINDOW arch: openai-moe.cpp:8 sets swa_type STANDARD
# with period 2, so llama_kv_cache_iswa gives the 18 even layers a 1024-cell
# window cache and only the 18 odd ones a full 65536-cell row. At -c 65535 f16
# that is 2304 + 36 = 2340 MiB, not the 4608 MiB a flat per-layer row implies.
# tensor-override.py used to size it flat and left two expert layers' worth of
# CUDA0 idle; measured occupancy was 11352 MiB of 16303.
#
# Measured with this layout (llama-server -lv 5):
#     CUDA0  model 9697 + KV 2340 + compute 667  CUDA1  model 6472 + compute 79
#
# Regenerate: ./tensor-override.py <model.gguf> --no-mtp --measure
# (add -q8 for a q8_0 KV cache: 2340 MiB -> 1243 MiB, room for one more layer.)

./llama-server -m /media/hnvcam/AI/LMStudio_Models/unsloth/gpt-oss-120b-GGUF/gpt-oss-120b-Q8_0-00001-of-00002.gguf \
  -ngl 99 \
  -dev CUDA0,CUDA1 \
  -ts 1,0 \
  -c 65535 \
  -ub 2048 \
  -ot 'blk\.(6|7|8|9)\.(ffn_down_exps\.bias|ffn_down_exps\.weight|ffn_gate_exps\.bias|ffn_gate_exps\.weight|ffn_up_exps\.bias|ffn_up_exps\.weight)$=CUDA1,blk\.(10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35)\.(ffn_down_exps\.bias|ffn_down_exps\.weight|ffn_gate_exps\.bias|ffn_gate_exps\.weight|ffn_up_exps\.bias|ffn_up_exps\.weight)$=CPU,^token_embd\.weight$=CPU' \
  -fa on \
  --temp 1.0 --top-p 0.95 --top-k 20 \
  --port 1234

cd ~/Workplace/llama-cli
