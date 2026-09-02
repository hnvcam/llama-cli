# Running big models on two mismatched GPUs — the plain-English version

This is the story of one bug we chased, written so it makes sense without a
background in LLM internals. The machine-facing version of the same material is
[CLAUDE.md](CLAUDE.md).

**The box:** an RTX 5070 Ti (16 GB, fast) and an RTX 4060 (8 GB, slower). 24 GB
of VRAM between them, and models that want ~20 GB. Everything below is about
getting a model to fit across two unequal cards without running out of memory.

---

## 1. What a model is made of

A model is a stack of **blocks** (also called layers). KAT-Coder has 40. Each
block holds a few big grids of numbers — **tensors**. Running the model means
multiplying your text through every block, in order.

The numbers are **quantised** to save space. Instead of 32 bits each, Q4_K
stores them at roughly 4 bits — but not independently. Values are packed in
groups of **256**, and each group of 256 shares one small header with a scale
factor. This detail matters later, so hold onto it:

> **256 numbers are glued together. You cannot take half a group — the 4-bit
> values are meaningless without the scale that lives in the group's header.**

---

## 2. Dense vs MoE — two ways to build a model

**Dense** (gemma-4-31B, Qwen3.8): every block has one big feed-forward network,
and *all* of it runs on *every* word. A 31-billion-parameter dense model reads
all 31 billion parameters per word.

**MoE** — Mixture of Experts (KAT-Coder, gpt-oss, Ling): every block has many
small feed-forward networks called **experts**, plus a tiny **router** that
picks a few per word. KAT-Coder has **256 experts per block and uses 8** of
them. So the model *stores* 20 GB but only *reads* about 3% of the expert
weights per word.

```
DENSE block                      MoE block (KAT-Coder)
                                 
  ┌───────────────┐                ┌──┐┌──┐┌──┐  …  ┌──┐   256 experts
  │               │                │e1││e2││e3│     │e256│  (only 8 run
  │  one wide FFN │                └──┘└──┘└──┘     └──┘    per word)
  │  21504 wide   │                 └── each only 512 wide ──┘
  └───────────────┘                        ▲
   all of it runs                      router picks 8
```

**This shape difference is the whole story of this session.** A dense model has
one *wide* FFN. An MoE gets its power from having *many narrow* ones. Remember
21504 vs 512.

---

## 3. Two ways to use two GPUs

### `-sm layer` — the default: split by block

Blocks 0–25 on CUDA0, blocks 26–39 on CUDA1. Simple, always works. But the
cards **take turns**: CUDA0 does its blocks while CUDA1 waits, then hands over.
Time per word = `time0 + time1`. The slow card is paid for in full.

```
CUDA0  ███████░░░░░░░
CUDA1  ░░░░░░░████████        ← one card is always idle
       └── total time ──┘
```

### `-sm tensor` — split *inside* every block

Cut every tensor down the middle-ish and give each card a piece. Now both cards
work on **the same block at the same time**, and time per word =
`max(time0, time1)`.

```
CUDA0  ██████████
CUDA1  ██████████            ← both busy; whoever finishes last sets the pace
       └─ total ─┘
```

Measured on gemma-4-31B on this box: **23.9 → 41.0 tokens/sec**. It's a big win
when it works.

### `-ts` — who gets how much

`-ts 0.75,0.25` means "give CUDA0 75%, CUDA1 25%". Under `-sm tensor` this is
really a **speed** knob, not just a capacity knob: since the cards run
concurrently, they should finish *together*, which means splitting in proportion
to their bandwidth:

```
ts = 896 / (896 + 272) = 0.767   → -ts 0.767,0.233 is the speed optimum here
```

### `-ot` — the MoE-only trick

Since an MoE reads only 8 of 256 experts per word, expert weights are the
*cheapest* thing to banish to the slow card or even to system RAM. `-ot` lets
you say exactly which tensors go where. **`-ot` chooses; `-sm tensor` can't.**
That distinction decides which tool wins later.

---

## 4. What "cutting a tensor" actually is

Here's one expert inside a KAT-Coder block:

```
your text (2048 numbers)
      │
      ▼   ffn_up:  2048 → 512
   hidden layer: 512 numbers
      │
      ▼   ffn_down: 512 → 2048
   result (2048 numbers)
```

To split that across two cards, you pick a **cut point** in that 512-wide hidden
layer:

```
      cut at 256
          ▼
 hidden: [0 ─────── 255][256 ─────── 511]
          CUDA0 does these  CUDA1 does these
          (and owns the matching slices of ffn_up AND ffn_down)
```

Each card computes half the hidden layer and half the result, then they add
their halves together. **The cut point is the only number in play.**

---

## 5. Why the cut can't land anywhere it likes

Four rules constrain where that cut may fall:

1. **Whole groups of 256.** From §1 — quantised numbers come glued in 256s. A
   cut inside a group is impossible.
2. **Both sides must agree.** Look at the diagram again: whichever slice of
   `ffn_up` CUDA0 owns, it needs the *matching* slice of `ffn_down` or it can't
   finish the multiplication. So the cut has to be legal for both tensors, and
   the stricter one wins.
3. **Whole attention heads.** Attention works in independent "heads"; half a
   head computes nothing meaningful. Cuts land on head boundaries (256 here).
4. **Round numbers run faster.** GPU kernels want widths that are multiples of
   128, otherwise they fall back to a slow path.

llama.cpp combines all four into one number it calls the **granularity** — the
smallest chunk you're allowed to hand out. For KAT-Coder's experts that's
**256**.

Then the actual code is just "hand out whole chunks, last card takes the
leftovers":

```c
high  = 512 * ts;        // where you asked for the cut
high -= high % 256;      // slide it back to a whole chunk
```

---

## 6. The bug: `-ts` is a request, not a promise

Now put §2 and §5 together.

```
KAT-Coder (MoE)   expert FFN is  512 wide ÷ 256 granularity =  2 chunks
gemma-4 (dense)   the FFN is   21504 wide ÷ 256 granularity = 84 chunks
```

**KAT-Coder has two chunks to give away.** Not "0.6 gets rounded a bit" —
there is no third chunk in existence. Watch what `-ts` does:

| you ask for | `512 × ts` | slide back to a chunk | CUDA0 actually gets |
|---|---|---|---|
| 0.55 | 281 | 256 | 50% |
| 0.6828 | 349 | 256 | 50% |
| **0.744** | 380 | 256 | **50%** |
| **0.75** | 384 | 256 | **50%** |
| 0.95 | 486 | 256 | 50% |

Every value from 0.5 to 0.9999 is **the same layout**. On gemma-4 the same
formula gives 84 possible positions, so 0.744 and 0.75 genuinely differ and the
knob behaves the way you'd expect. Same rule, opposite experience — purely
because of tensor width.

### The nasty extra twist

llama.cpp alternates which card is served first, block by block, so the
rounding leftovers don't always pile onto the same card. Sounds fair. But when
there are only two chunks:

- blocks served CUDA0-first: CUDA0 asks for 0.68 → gets 1 chunk → **50%**
- blocks served CUDA1-first: CUDA1 asks for 0.32 → gets **0 chunks** → CUDA0
  takes all of it → **100%**

Average: **75% on CUDA0**, whatever you typed. That's ~1.3 GB more on the 16 GB
card than the old planner predicted — which is exactly why it kept saying
*"CUDA0 out of memory"*.

Confirmed against llama.cpp's own log (`-lv 5`), which reports CUDA0's share:

```
load_tensors:  Meta() model buffer size = 14901.13 MiB     ← what really happened
                                          13623    MiB     ← what the old script predicted
```

---

## 7. Why KAT-Coder can't use `-sm tensor` at all

Only three whole-model splits are reachable, and the budgets are 14,815 MiB on
CUDA0 and 7,292 MiB on CUDA1:

| CUDA0 gets | CUDA0 needs | CUDA1 needs | |
|---|---|---|---|
| 25% | 5,829 ✅ | 16,138 | ❌ over by 8.8 GB |
| 50% | 10,984 ✅ | 10,984 | ❌ over by 3.6 GB |
| 75% | 16,138 ❌ over by 1.3 GB | 5,829 ✅ | |

The two splits the small card survives drown the big card, and the one the big
card survives drowns the small card. **There is no fourth option to try.**

So we went back to `-sm layer` — which places *whole blocks* and therefore has
40 positions to choose from, not 2. It fits, it loads, it runs:

```
CUDA0 14,656 MiB of 16,303      CUDA1 7,036 MiB of 8,188
```

---

## 8. The rules to remember

**A many-expert MoE is a poor `-sm tensor` candidate.** Its power comes from
expert *count*, so each expert is narrow, so there's nothing to cut. Dense
models have one wide FFN and split beautifully. Check the width before you
reach for `-sm tensor`.

**For an MoE, prefer `-ot` over `-sm tensor` the moment anything must leave
VRAM.** `-ot` picks *which* weights get exiled and picks the expert weights,
read 8 times in 256. `-sm tensor` slices everything equally, so whatever it
spills is read on every single word. Choosing beats sharing.

**For a dense model it's a straight speed contest** — `max()` beats `sum()`, so
`-sm tensor` usually wins, unless VRAM forces `-ts` so far from the bandwidth
optimum that the cards stop finishing together.

**Never plan against total VRAM.** Plan against *free* VRAM (a card holds back
a few hundred MB for the driver), then keep ~1 GB spare on top: the workspace
buffer is one big contiguous allocation that fails while memory is still
nominally available.

**Shrink the KV cache before you shrink the model.** `-ctk q8_0 -ctv q8_0`
halves the memory your context uses, and lowering `-ub` shrinks the scratch
buffer. Both bought us the headroom that made KAT-Coder fit.

**Measure, don't estimate, when it's tight.** `--measure` loads the model once
and reads the real workspace size. On KAT-Coder the estimate said 1,024 MB and
the truth was 462 MB — the difference between "needs `-ot` juggling" and "just
works".

---

## 9. Practical recipe

```bash
nvidia-smi                                  # 1. nothing else holding VRAM?
./tensor-override.py MODEL.gguf -q8         # 2. what's the biggest context?
./tensor-override.py MODEL.gguf -q8 -c 59392 -ub 256 --measure   # 3. exact plan
```

Step 2 prints **two** commands for an MoE, because "the biggest context" has two
different answers. `-sm tensor` slices every tensor across both cards, so its
ceiling is usually the lower one; dropping `-sm` lets the cards take turns but
fits a longer context. The tool's pick runs, the other is commented out — so
you can pipe it straight into a shell. If you want the other one, do not strip
the `#` by hand — re-run with `--prefer no-sm` (or `--prefer tensor`) and it
comes out live instead. Dense models print the same pair for the same reason.

Then run the command it prints. If it emits a `-ts`, the tool now tells you what
that `-ts` **really** does:

```
NOTE: -ts 0.6875 does NOT mean 68.8% on CUDA0. llama.cpp rounds every cut
down to a granularity, so CUDA0 really gets 76.1% of the sliced weights
```

Trust the second number, not the one you typed.
