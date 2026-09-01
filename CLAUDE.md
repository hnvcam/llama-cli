# CLAUDE.md

Cached facts for this repo so a session does not have to re-read llama.cpp
source, re-scan the GGUF collection, or re-probe the hardware. Everything here
was verified on 2026-09-01 against llama.cpp `bb4caa754` (v0.2.0). If a claim
below is load-bearing for a decision, the source file:line is given — re-read it
only if llama.cpp has moved on.

Companion doc: **[LLM.md](LLM.md)** — the same material written for a human who
is new to this. Keep the two in sync when facts change.

---

## The box

| | |
|---|---|
| CUDA0 | RTX 5070 Ti — 16,303 MiB total, ~15,839 MiB free idle, **896 GB/s** |
| CUDA1 | RTX 4060 — 8,188 MiB total, ~7,804 MiB free idle, **272 GB/s** |
| CPU | i5-13500, 62 GiB RAM, ~50 GB/s (`_cpu_gb_s` in `gpu-bandwidth.json`) |
| llama.cpp | `~/Workplace/llama.cpp`, binaries in `build/bin/` |

Bandwidths live in `gpu-bandwidth.json`; only the *ratio* matters, since the
speed-optimal split is `ts_i = BW_i / sum(BW)` = **0.7671 / 0.2329** here.

`memory.free` is lower than `memory.total` even on an idle card (driver
reserve). `tensor-override.py` plans against **free** and subtracts a further
`--reserve0 1024` / `--reserve1 512` MiB, because the compute buffer is one
large contiguous `cudaMalloc` that fails well before VRAM is nominally
exhausted (512 MiB reserve was measured to OOM with 306 MiB still "free").
Expect predicted occupancy to run ~250 MiB above what `nvidia-smi` shows after
a successful load — that is the reserve doing its job.

**Always check `nvidia-smi` before running `tensor-override.py`.** It reads
`memory.free`, so a forgotten llama-server makes every budget wrong.

---

## The finding of this session: `-ts` is a request, not a ratio

`-sm tensor` (tensor parallelism) slices each tensor across both cards so they
work concurrently — a token costs `max(bytes_i/BW_i)` instead of
`sum(bytes_i/BW_i)`. But **llama.cpp does not honour `-ts` proportionally.**

`llama_meta_device_get_split_state` (`src/llama-model.cpp:361-727`) computes
every device's slice as:

```c
high  = ne_segment * ts_scan[j] / ts_scan.back();
high -= high % granularity;          // llama-model.cpp:706-708
```

so each cut is **rounded DOWN to a granularity** and the last device in a
rotation takes the remainder. Two consequences:

1. **Coarse tensors barely move.** If a tensor's split axis is only N
   granularities wide, there are only N+1 reachable cuts. KAT-Coder's expert
   FFN is 512 wide with granularity 256 → cuts of 0, 256 or 512 only, i.e.
   0% / 50% / 100%. Every `-ts` from 0.5 to 0.9999 produces an identical layout.
2. **The error is one-sided.** `rotation = get_il_eff(il) % n_devices`
   (`llama-model.cpp:413-421`) alternates which card is served first, and the
   card served *last* collects the rounding remainder. On 2 cards, half the
   blocks round CUDA0 down to 50% and the other half round CUDA1 down to 0%
   (leaving CUDA0 100%). Model-wide average: **75% on CUDA0 for any `-ts` in
   (0.5, 1.0)**.

### Where granularity comes from

`get_split_granularity` (`llama-model.cpp:607-685`) — the smallest chunk that is
simultaneously legal for all of:

| constraint | why | value here |
|---|---|---|
| whole quant blocks | a Q4_K/Q6_K block of 256 values shares one scale/min header; half a block is meaningless | 256 |
| whole attention heads | softmax/normalisation runs per head | `n_embd_head_k` = 256 (KAT), SSM `head_dim` = 128 |
| kernel width | `std::lcm(blck_size, 128)` so wide kernels stay usable | ≥128 |

The block size is read off a **reference tensor, not the tensor being cut**
(`get_tensor_config_impl(axis, "ffn_down.weight", "ffn_down_exps.weight")`,
`llama-model.cpp:504`). Reason: cutting `ffn_up`'s columns is free (whole rows),
but the *same index* cuts `ffn_down` inside its rows, where block alignment
binds. Both sides of the matmul must agree on the cut point.

### Mirrored vs sliced

A tensor no pattern matches is `GGML_BACKEND_SPLIT_AXIS_MIRRORED` — a **full
copy on every card**, charged to both budgets. Norms are trivial, but a tied
`token_embd` used as the output projection is 756 MiB *per card* on gemma-4.
`ffn_down_exps.bias` is `SPLIT_AXIS_PARTIAL`, which costs the same as mirrored.
`token_embd` itself is always CPU-resident (`dev_input` is unconditionally the
CPU, `llama-model.cpp:1377`).

### The Meta() log lines report **device 0's share**, not the total

```
load_tensors:       Meta() model buffer size = 14901.13 MiB
llama_kv_cache:     Meta() KV buffer size    =    31.88 MiB   (total 42.50)
llama_memory_recurrent: Meta() RS buffer size=   172.73 MiB   (total 251.25)
```

Needs `-lv 5`. Add `-cmoe`/`--measure` for real compute-buffer numbers.

---

## `tensor-override.py`

`meta_split_config` / `meta_split_segments` / `meta_split_granularity` /
`meta_rotation` / `split_fracs` are a **verbatim port** of the above. Validated
to the MiB against real loads:

| | port | llama.cpp |
|---|---|---|
| KAT-Coder model buffer @ `-ts 0.6828` | 14901.12 | 14901.13 |
| KAT-Coder KV buffer | 31.88 | 31.88 |
| KAT-Coder RS buffer | 172.73 | 172.73 |
| gemma-4-31B model buffer @ `-ts 0.75` | 13285.10 | 13285.19 |

If these ever disagree after a llama.cpp bump, the port is stale — re-read
`llama-model.cpp:361-727` and fix it there, not by fudging budgets.

Because occupancy is a **step function** of `-ts`, `tensor_fit` walks the flat
intervals from `split_breakpoints()` and prices each exactly; the old closed
form survives only *inside* an interval, for tensors cut finer than
`SPLIT_MAX_STEPS` (512). Do not reintroduce `ts * bytes` anywhere.

### When the script recommends `-sm tensor`

- **MoE** (`moe_tensor_section`): a **fit** question, not a speed one. Off if no
  `-ts` fits, and off if the plan needs `n_cpu_blocks > 0`. Rationale: `-ot`
  chooses *which* weights leave VRAM (expert weights, read `n_expert_used /
  n_expert` of the time — 8/256 on KAT); `-sm tensor` slices everything
  uniformly, so anything it spills is read at full rate every token.
- **Dense** (`dense_report`): a **throughput** comparison — `max()` vs `sum()`
  of bytes/bandwidth at the measured 67% (`MBU_TENSOR`) / 86% (`MBU_LAYER`) of
  nameplate. Ties go to tensor. When `-ts` is clamped far from the optimum the
  cards stop finishing together and the layer split wins.

### Usage notes

```bash
./tensor-override.py MODEL.gguf -q8                     # largest ctx that fits
./tensor-override.py MODEL.gguf -q8 -c 59392 --measure  # exact plan for a ctx
```

`--measure` loads the model once and reads the real compute buffer. **Use it for
any tight fit** — the estimate is a coarse upper bound and was 1,024 MiB vs a
measured 462 MiB on KAT-Coder, which is the difference between "needs `-ot`" and
"fits on a plain layer split".

---

## Model inventory

Paths, and the split resolution that decides whether `-sm tensor` is usable.
"steps" = split-axis width / granularity on the FFN; more steps = finer control.

| model | arch | kind | size | FFN cut | steps | `-sm tensor` on this box |
|---|---|---|---|---|---|---|
| `~/…/LLAMA_Models/gemma-4-31B-it-Q4_K_M.gguf` | gemma4 | dense | 17.05 GiB | 21504/256 | **84** | ✅ used in `gemma4-31b.sh` (`-ts 0.75 -c 32768 -np 1`) |
| `~/…/LLAMA_Models/Qwen3.8-27B-UD-Q4_K_M.gguf` | qwen35 | dense | 15.32 GiB | 17408/256 | **68** | ✅ fits, `-c 64512 -ts 0.7169` |
| `~/…/LMStudio_Models/…/KAT-Coder-V2.5-Dev.i1-Q4_K_M.gguf` | qwen35moe | MoE 256e | 19.70 GiB | 512/256 | **2** | ❌ impossible — see below |
| `~/…/LLAMA_Models/Hy-MT2-30B-A3B.i1-Q5_K_M.gguf` | hy_v3 | MoE 128e | 19.90 GiB | 768/256 | **3** | ❌ jumps 50% → 83%, neither fits |
| `~/…/LLAMA_Models/Ling-3.0-flash-MXFP4_MOE.gguf` | bailingmoe3 | MoE 512e | 65.04 GiB | 768/128 | 6 | ❌ far too large anyway |
| `~/…/LMStudio_Models/…/gpt-oss-120b-Q8_0-00001-of-00002.gguf` | gpt-oss | MoE 128e | 59.02 GiB | 2880/128 | 22 | ❌ far too large anyway |
| `~/…/LLAMA_Models/LongCat-Flash-Lite-…-Q4_K_M.gguf` | — | — | — | — | — | file not present as of this session |

**Rule of thumb: a many-expert MoE is a bad `-sm tensor` candidate.** Capacity
comes from expert *count*, so each expert's FFN is narrow, and tensor
parallelism cuts *inside* an expert. Dense models have one wide FFN and split
cleanly.

### KAT-Coder-V2.5 — worked example

Only three model-wide splits are reachable (budgets 14,815 / 7,292 MiB, `-c
59392`, q8 KV, `-ub 256`):

| CUDA0 share | CUDA0 | CUDA1 |
|---|---|---|
| 25% | 5,829 ok | 16,138 **over by 8.8 GiB** |
| 50% | 10,984 ok | 10,984 **over by 3.6 GiB** |
| 75% | 16,138 **over by 1.3 GiB** | 5,829 ok |

No fourth option exists. With the measured compute buffer the model fits on a
**plain layer split** with no `-ot` and no `-ts` — verified loaded and
generating at CUDA0 14,656 MiB / CUDA1 7,036 MiB. That is what `kat-coder.sh`
now runs.

Also worth knowing: this arch (`qwen35moe`, Qwen3-Next family) is 40 blocks — 30
recurrent (SSM/KDA) + 10 full-attention, `full_attention_interval = 4` so
`il % 4 == 3` are the attention layers. Only those 10 get a KV cache; the other
30 get the recurrent state.

---

## Gotchas hit this session

- `pkill -f 'llama-server -m'` **matches the wrapper shell's own command line**
  and kills the whole compound command. Kill by port or `pgrep` first, or run
  the kill as its own call.
- Piping llama-server through `grep` buffers the output; a killed pipeline loses
  everything. Redirect to a file instead.
- Buffer-size lines need `-lv 5`.
- llama-server prints `failed to fit params to free device memory:
  llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort` under
  `-sm tensor` — that is informational, not the OOM.
