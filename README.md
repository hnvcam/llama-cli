# llama-cli — personal llama.cpp notes and a tensor-placement planner

> **Read this first.** This repo is my own working notes for running large MoE GGUF
> models on my desktop. It is published for reference, not as a tool. It is not
> packaged, not tested anywhere but here, and I am not trying to make it portable.
> If it helps you, take the ideas — but expect to edit the code.

## What is here

| File | What it is |
| --- | --- |
| `tensor-override.py` | Computes llama.cpp `-ot` tensor overrides for large MoE models that do not fit in VRAM, and prints a ready-to-run `llama-server` command line. |
| `BUILD-LLAMA.md` | How I build llama.cpp with CUDA. |
| `ling-flash.sh`, `kat-coder.sh` | Launch scripts for two specific models, generated from the planner's output and then frozen. |
| `CLI-HELP.md`, `SERVER-HELP.md` | Dumps of `llama-cli --help` / `llama-server --help` for the build I use, so I can grep flags without launching anything. |

## The machine this was written for

Everything here is tuned to one box, and several constants in `tensor-override.py`
were measured on it, not derived:

```
CUDA0   RTX 5070 Ti   16303 MiB   PCIe 4.0 x16   -- the fast card
CUDA1   RTX 4060       8188 MiB   PCIe 3.0 x8    -- expert-weight storage only
driver  580.173.02     CUDA 13.0
llama.cpp b10277 (3a0124fa8), built with GGML_CUDA=ON
```

### Where it will not work for you

`tensor-override.py` will refuse or misbehave outside this shape:

- **Single GPU** — it hard-exits with `error: this script assumes 2 GPUs, found 1`.
  The placement policy is written around a fast card plus a slow card. Making it
  work on one GPU is a real change, not a flag.
- **Non-CUDA backends** (ROCm, Vulkan, Metal, SYCL) — device names are hardcoded as
  `CUDA0` / `CUDA1` throughout, and GPU discovery shells out to `nvidia-smi`.
- **More than 2 GPUs** — extra devices are ignored.
- **No local llama.cpp build** — `--measure` needs a real `llama-server` binary, and
  the emitted command line assumes flags from build b10277.

## Build your own llama.cpp

I strongly recommend compiling llama.cpp yourself against the CUDA toolkit rather
than using a prebuilt release binary or running the model through LM Studio. On my
box that was a significant speedup — not a few percent. See
[`BUILD-LLAMA.md`](BUILD-LLAMA.md); the short version is:

```bash
cmake -B build -G Ninja -DGGML_CUDA=ON
cmake --build build --config Release -j10
```

Two honest caveats: this is my result on my hardware and my models, not a
benchmark, and part of the win is simply that a self-built binary lets you use
current `master` — new architectures and MoE-offload improvements land in
llama.cpp faster than they reach any packaged distribution. A self-built binary
also lets you pass `-ot` at all, which is the entire point of this repo.

---

# `tensor-override.py`

## The problem it solves

A large MoE model has two very different kinds of weight:

- **Latency-critical, touched every token**: attention, KV cache, recurrent state,
  norms, routers, dense FFN, shared experts, output projection. These must be on
  the fast GPU or throughput collapses.
- **Expert weights**: huge, but only `n_expert_used` of `n_expert` are read per
  token. These tolerate living on a slower GPU, or even in system RAM.

llama.cpp's `-ot` flag can place tensors by regex, but writing those regexes by
hand means knowing every tensor's exact size, how big the KV cache will be at your
context length, and how much scratch space the CUDA compute buffer will want.
This script reads the GGUF header, computes all of that exactly, packs the
devices, and prints the command.

Crucially it also emits `-ts 1,0`. `-ot` only relocates *weights* — the KV cache
for a layer follows `-ngl`/`-ts`, so without `-ts` llama.cpp would split the KV
cache across both cards by free VRAM. `-ts 1,0` pins every layer (hence all KV) to
CUDA0, and `-ot` then pushes expert weights outward.

## Quick start

```bash
# plan it (fast, reads only the GGUF header)
./tensor-override.py /path/to/model.gguf -c 65535 -q8

# plan it accurately (loads the model once, ~1 min)
./tensor-override.py /path/to/model.gguf -c 65535 -q8 --measure
```

It prints a memory accounting table and a per-device breakdown on **stderr**, and
the final `llama-server` command on **stdout** — so `./tensor-override.py model.gguf > run.sh`
gives you a runnable script and still shows you the reasoning.

## Arguments

### Model and context

| Flag | Default | What it does |
| --- | --- | --- |
| `model` | *(required)* | Path to the `.gguf`. For a split model, pass any shard — it finds `-00001-of-000NN` siblings automatically and errors if one is missing. |
| `-c`, `--ctx` | *(searched)* | Context size. Drives the KV cache calculation, which is usually the single biggest non-weight consumer of CUDA0. **Omit it** and the script reports the largest `-c` that keeps every weight in VRAM — separately for the `-ot` plan and for `-sm tensor`, since the two ceilings differ, printing both commands (the recommended one live, the other commented out). It falls back to `65535` with a CPU spill only when no context at all is GPU-resident. |
| `-q8`, `--q8` | off | Quantise the KV cache to `q8_0` instead of `f16`, and add `-ctk q8_0 -ctv q8_0` to the emitted command. Roughly halves KV size; on a 16 GB card at long context this is often what makes a plan feasible at all. |
| `-ub`, `--ubatch` | `512` | Physical batch size. Only affects the *estimated* compute buffer (larger ubatch → more scratch). Passed through to the emitted command. |
| `-np`, `--parallel` | `4` | Number of server slots, spelled like llama-server's own `-np`. Sizes the recurrent (SSM/KDA) state, which is per-sequence and independent of context length. Matches llama-server's auto default of 4 slots with unified KV. **Every emitted command carries `-np` explicitly**, together with `-kvu` (or `-no-kvu` under `--no-kv-unified`): naming `-np` at all — even `-np 4` — turns off the auto `kv_unified` that llama-server only applies to `-np -1` (`server.cpp:152-158`), so the KV flag has to be stated alongside it or the cache is sized differently from the plan. |

### MTP / speculative decoding

| Flag | Default | What it does |
| --- | --- | --- |
| `--no-mtp` | MTP is **on** | Do not use the model's MTP/nextn blocks. By default, if the GGUF actually ships `nextn` tensors, the plan budgets VRAM for them and the emitted command carries `--spec-type draft-mtp`. |

See [Should you use MTP?](#should-you-use-mtp) below — for any model that has to
spill experts to CPU, **I recommend `--no-mtp`**.

The script distinguishes three MTP states and tells you which one you are in:
declared-and-present (usable), declared-but-tensors-absent (the quant dropped
them; MTP is impossible), and present-but-disabled. It also knows that some
architectures — currently `bailingmoe3` — refuse to load nextn blocks at all and
log `model has unused tensor blk.N.* -- ignoring`; with `--no-mtp` those blocks
are excluded from the budget and that VRAM is reclaimed for experts.

#### `--spec-type draft-mtp` builds a whole second context

This is the expensive part, and it is not the weights. `common_speculative_init_result`
calls `llama_init_from_model()` a **second time** on the already-loaded target model
with `ctx_type = LLAMA_CONTEXT_TYPE_MTP`. No weights are re-read — but a context
owns a KV cache and a compute buffer, and for most architectures llama.cpp installs
no layer filter on that second KV cache, so it is a *full duplicate*. llama-server
prices it itself:

```
srv load_model: [spec] adding 2203.01 MiB to fit_params_target for device CUDA0
srv load_model: [spec] estimated memory usage of MTP context is 2203.01 MiB
```

Measured on LongCat-Flash-Lite Q4_K_M at `-c 65535 -ub 512`:

| | KV cache | compute (CUDA0) | compute (CUDA1) |
| --- | --- | --- | --- |
| target context | 2088.00 MiB | 983.96 MiB | 150.07 MiB |
| MTP context | 2088.00 MiB | 115.01 MiB | 0.00 MiB |

So MTP costs **2.2 GiB on CUDA0 here — about 1.6 expert layers** — before you count
the nextn weights. The planner budgets all of it. The archs where llama.cpp *does*
filter the MTP KV down to just the nextn blocks (`qwen35`, `qwen35moe`, `step35`,
`hy_v3`) are listed in `MTP_KV_FILTERED_ARCHS`; everything else gets the duplicate.

Getting this wrong is not a rounding error — it loads the target model fine and then
dies at the very last allocation:

```
common_speculative_init_result: creating MTP draft context against the target model
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 2088.00 MiB on device 0: cudaMalloc failed: out of memory
llama_init_from_model: failed to initialize the context: failed to allocate buffer for kv cache
srv load_model: failed to create MTP context
```

Note that `-ngl 99` disables llama.cpp's own auto-fit (`failed to fit params to free
device memory: n_gpu_layers already set by user to 99, abort`), so nothing else is
going to catch this for you. Under `--measure` the second context's compute buffer
is read from the log too, alongside the target's.

### Compute buffer sizing

This is the one number that cannot be derived from the GGUF, and the three flags
below are three ways to pin it down. Priority is `--compute-buf` > `--measure` >
estimate.

| Flag | Default | What it does |
| --- | --- | --- |
| `--measure` | off | Load the model once through `llama-server -lv 5` and read the **real** compute buffer sizes out of llama.cpp's own log. |
| `--compute-buf MiB` | — | Hardcode the CUDA0 compute buffer. Overrides both the estimate and `--measure`. |
| `--llama-server PATH` | search | Which `llama-server` `--measure` should run. Falls back to `PATH`, then `~/Workplace/llama.cpp*/build/bin/llama-server`. |
| `--measure-port` | `18099` | Port the probe server binds. Change it if something already owns it. |
| `--measure-timeout` | `600` | Seconds to wait for the probe. |
| `--compute-buf1-frac` | `0.35` | CUDA1's compute buffer as a fraction of CUDA0's, used only when not measuring. Measured 328 vs 1022 MiB on Ling at `-ub 2048`. |

### Why `--measure` is not the default

Because it is the only expensive thing the script does. Everything else is a
GGUF header parse that finishes in milliseconds and touches no GPU. `--measure`
spawns a real `llama-server`, loads the entire model (minutes, for a 100 GB MoE
off spinning storage), binds a TCP port, and can fail on its own with an OOM or a
timeout. Turning that on by default would break `--json` / `--flags-only`
scripting use and make the common "just show me the plan" case unusable.

### Why you still want it

Only the compute buffer is affected — weights, KV cache and recurrent state are
computed exactly from the header either way. But the compute buffer depends on
the *graph* llama.cpp builds, which is architecture-specific and does not follow
any cheap formula. Measured on this box:

```
bailingmoe3          ub=2048   1022.02 MiB
qwen35moe            ub=2048   ~504    MiB
longcat-flash-ngram  ub= 512   1031.96 MiB
```

LongCat needs more scratch at a 4× *smaller* ubatch than Ling does. No function of
`(n_expert, n_ff_exp, n_embd, ubatch)` fits all three — an earlier version of the
formula predicted 128 MiB against an actual 1032 MiB and produced a plan that
OOM'd. So the built-in estimate is now a deliberately generous envelope with a
hard 1024 MiB floor. It is **safe but fat**: it errs high, which wastes VRAM that
could have held another expert layer. That is why the output nags you with
`^ a coarse upper bound; re-run with --measure for the real value` on every
un-measured run.

The probe itself keeps exactly one expert layer on each GPU and spills the rest,
which reproduces the graph shape that drives buffer size without loading much.
Measured on LongCat at `-ub 512`: `-cmoe` (no experts on any GPU) reported 762 MiB
(26% too low), the probe reported 983.96 MiB, the full plan used 1031.96 MiB —
hence the 1.06× safety multiplier applied to measured values.

**Practical workflow:** run `--measure` once per (model, context, ubatch, MTP)
combination, note the `measured: CUDA0 …` line, then pass `--compute-buf <MiB>`
on every later run. It takes priority over everything and costs nothing.

### VRAM budget

| Flag | Default | What it does |
| --- | --- | --- |
| `--reserve0 MiB` | `1024` | Held back on CUDA0 for the CUDA context plus allocator headroom. The context itself measures ~248 MiB; the rest is headroom, because the compute buffer is a single ~1 GiB *contiguous* `cudaMalloc` that fails well before VRAM is nominally exhausted. `--reserve0 512` was measured to OOM with 306 MiB still nominally free. |
| `--reserve1 MiB` | `512` | Same for CUDA1 (context measures ~104 MiB). |
| `--vram` | detected | Override detected VRAM, MiB, comma-separated: `--vram 16303,8188`. Useful for planning against a machine you are not sitting at, or for leaving room for a desktop session. |

### Output

| Flag | What it does |
| --- | --- |
| *(none)* | Full report on stderr, `llama-server` command on stdout. When both a `-sm tensor` and a no-`-sm` plan exist, stdout carries the recommended one live and the other commented out — a `#` comment eats the trailing `\`, so piping into a shell still runs exactly one. |
| `--prefer {auto,no-sm,tensor}` | Which plan comes out **uncommented**, default `auto` (whatever the report recommends). This is how you copy-paste or pipe the other one without stripping a `#` off every line. Both plans are still computed and reported, unlike `--no-tensor-split`. |
| `--no-tensor-split` | Do not compute the `-sm tensor` plan at all; report and emit only the layer split / `-ot` placement. |
| `--flags-only` | Print just the `-ot` argument (or a comment saying none is needed). |
| `--json` | Machine-readable plan: byte accounting, per-device expert layer lists, lookup table placement, and every `-ot` pattern. |
| `--server-bin` | How the emitted command should invoke llama-server. Default `./llama-server`. |

## How it places things

In strict priority order:

1. **CUDA0 first, non-negotiable**: KV cache, recurrent/SSM/KDA state, compute
   buffer, then all core weights (attention, dense FFN, shared experts, routers,
   norms, `output.weight`). If these alone overflow CUDA0 the script refuses to
   emit anything and tells you what to shrink — because no tensor override can
   rescue a configuration whose every-token tensors don't fit.
2. **CUDA0's leftover** is filled with whole per-layer expert sets.
3. **CUDA1** is filled with whole per-layer expert sets.
4. **Lookup tables** (`token_embd`, LongCat's `ngram_embd.*`, nextn embeddings)
   get whatever is still free. These are read only through `ggml_get_rows`, so
   putting one on CPU costs microseconds per token — they are the cheapest thing
   to evict, and they deliberately rank *below* experts, where a spilled GiB
   costs real tokens/sec. The script checks for a separate `output.weight` first:
   with tied embeddings, evicting `token_embd` would put a full vocab matmul on
   the CPU every token, so it is reclassified as core.
5. **Everything left** goes to CPU. If the CPU share fits comfortably in available
   RAM the script suggests `--no-mmap`; if it doesn't, it explicitly tells you to
   keep mmap on so you don't swap.

## Should you use MTP?

**My recommendation: use `--no-mtp` whenever the plan has to put expert layers on
the CPU.** In my use it did not pay for itself there. Two things go wrong, and it
is worth separating them because only one is obvious.

**1. It costs VRAM you needed for experts.** Two separate charges:

*The nextn block weights must be resident.* On an architecture like `bailingmoe3` —
which does not even load those blocks unless you ask for MTP — that is 1.55 GiB on
Ling 3.0 Flash, against an expert layer size of 1,530 MiB.

*And the second context costs more than the weights do.* As above: a duplicate KV
cache plus a second compute buffer, 2,203 MiB on LongCat at 64K context. That one
scales with `-c`, so it gets worse exactly where you wanted the context.

Together, on LongCat, MTP moves **two more expert layers** from GPU to CPU,
permanently, on every token — bought to fund a speculative gamble.

**2. Batched verification stops amortising on CPU-resident experts.** This is the
part worth being precise about, and it is a refinement of "MoE experts don't seem
to get used properly under MTP" — which is the right instinct with a slightly
different mechanism.

Speculative decoding works because verifying *k* drafted tokens in one batch is
supposed to cost about the same as generating one token: you read each weight from
memory once and reuse it across all *k* tokens. That premise holds for dense
layers and for anything already resident in VRAM.

It does not hold for MoE experts. Each token routes to its own top-*k* experts, so
a batch of *k* tokens activates the **union** of their expert sets — up to
`k × n_expert_used` distinct experts per layer instead of `n_expert_used`. For
experts sitting in VRAM this widening is nearly free: the weights are already
there, so you only spend extra compute, and the GPU had spare compute anyway (it
actually improves GEMM utilisation). For experts sitting in system RAM, every
additional activated expert is another block of weights dragged across the memory
bus. The cost of the verify pass grows roughly with the number of drafted tokens
instead of staying flat.

So it *is* a bandwidth effect — just not the one you'd guess. The problem isn't
that the link is too slow in absolute terms; it's that batching, the thing that
normally makes speculative decoding free, specifically fails to amortise on the
offloaded expert layers. And when the draft is rejected you paid that widened cost
for tokens you threw away.

This is why the same flag flips sign: **MTP is a real win when the whole model
fits on GPU, and a net loss once experts spill to CPU.** Run it both ways on your
own box before trusting me — this is reasoning from the mechanism plus
my own observation, not a controlled benchmark.

## Example

```
$ ./tensor-override.py /media/hnvcam/AI/LLAMA_Models/Ling-3.0-flash-MXFP4_MOE.gguf -c 65535 --no-mtp

Model         : Ling-3.0-flash-MXFP4_MOE.gguf
Architecture  : bailingmoe3  (Source)
Blocks        : 42 live -> 35 recurrent (SSM/KDA), 7 attention [MLA]
MoE           : 512 experts, 8 used/token, 40 layers carry expert tensors
MTP           : available but disabled (--no-mtp); 1 nextn block(s)
                (1.55 GiB) are not loaded by bailingmoe3
Context       : 65535   KV type: f16   ubatch: 512

Memory accounting
  model weights, total              63.50 GiB
    non-expert (core)                3.33 GiB
    lookup tables (get_rows)         0.40 GiB   (1 tensors, evictable)
    expert tensors                  59.77 GiB   (40 layers, 1,530 MiB/layer)
  KV cache @ 65535 (f16)              576 MiB
  recurrent state (4 seq)             300 MiB
  compute buffer  (estimated)       1,026 MiB
      ^ a coarse upper bound; re-run with --measure for the real value

  CUDA0 (NVIDIA GeForce RTX 5070 Ti)
    total VRAM                     16,303 MiB
    - reserved (CUDA ctx)           1,024 MiB
    = budget                       15,279 MiB
      KV + recurrent + compute      1,902 MiB
      core weights                  3,411 MiB
      6 expert layers               9,180 MiB
      1 lookup tables                 408 MiB
    used of budget                 14,901 MiB   (97.5%)
    predicted actual occupancy     15,149 MiB   of 16,303 MiB
  CUDA1 (NVIDIA GeForce RTX 4060)
    ...
      4 expert layers               6,120 MiB
    used of budget                  6,479 MiB   (84.4%)
  CPU
    30 expert layers
    total                           44.82 GiB

=> 30 expert layer(s) must live on CPU.
   -ts 1,0 pins all layers (hence the whole KV cache) to CUDA0;
   -ot then moves expert weights out to CUDA1 / CPU.
   44.82 GiB lands on CPU and you have 53.98 GiB available;
   add --no-mmap for better performance.
```

The command on stdout from exactly this run is what `ling-flash.sh` contains.
Both launch scripts in this repo are frozen snapshots — if you change context
length, KV type, or llama.cpp version, re-run the planner rather than editing the
regex by hand.

## Known sharp edges

- Constants named "measured on this box" (`CUDA_CONTEXT_MEASURED`,
  `RESIDUAL_PER_*`, `MEASURE_SAFETY`, the reserve defaults) are exactly that.
  They are correct for a 5070 Ti + 4060 on driver 580 / CUDA 13, and approximate
  everywhere else.
- `NEXTN_IGNORED_ARCHS` is a hand-maintained list of architectures observed to
  skip nextn blocks. Anything unlisted is assumed to load them, which is the safe
  direction: over-counting only wastes packing efficiency, while under-counting
  silently overcommits VRAM and OOMs. If you see `unused tensor blk.N.*` warnings
  for another architecture, add it there and reclaim the VRAM.
- `MTP_KV_FILTERED_ARCHS` is likewise hand-maintained, and errs the same way:
  anything unlisted is assumed to duplicate the *whole* KV cache in the MTP draft
  context. If your arch gained an MTP layer filter in `llama_model::create_memory`,
  add it there. The MTP compute-buffer fallback (`MTP_COMPUTE_ESTIMATE`, 256 MiB)
  is a generous stand-in for the 115 MiB measured on LongCat; `--measure` reads
  the real one.
- The ggml type table is dumped from `libggml-base.so` at b10277. A future ggml
  release that renumbers or adds types will need it refreshed.
- All overrides are emitted as a **single** `-ot` argument, because b10277 warns
  `argument '-ot' specified multiple times ... only last value will be used` and
  silently drops all but the final pattern if you repeat the flag.
