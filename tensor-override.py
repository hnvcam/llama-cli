#!/usr/bin/env python3
"""
tensor-override.py -- compute llama.cpp `-ot` tensor overrides for large MoE GGUF models.

Targets a 2-GPU box:
    CUDA0 = RTX 5070 Ti (16 GB, PCIe 4.0 x16)  -- the fast card
    CUDA1 = RTX 4060     (8 GB, PCIe 3.0 x8)   -- expert-weight storage only

Placement policy (in priority order):
  1. Everything latency-critical goes to CUDA0: the whole KV cache, recurrent
     (SSM/KDA) state, attention weights, dense FFN, shared experts, routers
     (ffn_gate_inp), norms, and output.weight.
  2. Whatever VRAM is left on CUDA0 is filled with whole per-layer expert sets.
  3. Then CUDA1 is filled with whole per-layer expert sets.
  4. The remainder stays on CPU.

Note on `-ts`: `-ot` only relocates *weights*. The KV cache for layer `il` is
allocated on `dev_layer[il]`, which is decided purely by `-ngl` / `-ts`
(llama-model.cpp: select_buft -> dev_layer). With no `-ts`, llama.cpp splits
layers by *free VRAM* and the KV cache would be spread over both GPUs. So to
honour rule 1 we emit `-ts 1,0`, pinning every layer (hence all KV) to CUDA0,
and then use `-ot` to push expert weights out to CUDA1/CPU.

Usage:
    ./tensor-override.py MODEL.gguf [-q8] [-c CTX] [options]
"""

import argparse
import glob
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field

MiB = 1024 * 1024
GiB = 1024 * 1024 * 1024

GGUF_MAGIC = 0x46554747

# (blck_size, type_size) dumped from libggml-base.so (b10277) -- exact, not estimated.
GGML_TYPE_TRAITS = {
    0:  (1, 4),      # f32
    1:  (1, 2),      # f16
    2:  (32, 18),    # q4_0
    3:  (32, 20),    # q4_1
    6:  (32, 22),    # q5_0
    7:  (32, 24),    # q5_1
    8:  (32, 34),    # q8_0
    9:  (32, 36),    # q8_1
    10: (256, 84),   # q2_K
    11: (256, 110),  # q3_K
    12: (256, 144),  # q4_K
    13: (256, 176),  # q5_K
    14: (256, 210),  # q6_K
    15: (256, 292),  # q8_K
    16: (256, 66),   # iq2_xxs
    17: (256, 74),   # iq2_xs
    18: (256, 98),   # iq3_xxs
    19: (256, 50),   # iq1_s
    20: (32, 18),    # iq4_nl
    21: (256, 110),  # iq3_s
    22: (256, 82),   # iq2_s
    23: (256, 136),  # iq4_xs
    24: (1, 1),      # i8
    25: (1, 2),      # i16
    26: (1, 4),      # i32
    27: (1, 8),      # i64
    28: (1, 8),      # f64
    29: (256, 56),   # iq1_m
    30: (1, 2),      # bf16
    34: (256, 54),   # tq1_0
    35: (256, 66),   # tq2_0
    39: (32, 17),    # mxfp4
    40: (64, 36),    # nvfp4
    41: (128, 18),   # q1_0
    42: (64, 18),    # q2_0
}

GGML_TYPE_NAMES = {
    0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1", 8: "q8_0",
    9: "q8_1", 10: "q2_K", 11: "q3_K", 12: "q4_K", 13: "q5_K", 14: "q6_K",
    15: "q8_K", 16: "iq2_xxs", 17: "iq2_xs", 18: "iq3_xxs", 19: "iq1_s",
    20: "iq4_nl", 21: "iq3_s", 22: "iq2_s", 23: "iq4_xs", 24: "i8", 25: "i16",
    26: "i32", 27: "i64", 28: "f64", 29: "iq1_m", 30: "bf16", 34: "tq1_0",
    35: "tq2_0", 39: "mxfp4", 40: "nvfp4", 41: "q1_0", 42: "q2_0",
}

# KV cache element types selectable via -ctk/-ctv, as (name, ggml type id).
KV_TYPE_F16 = ("f16", 1)
KV_TYPE_Q8_0 = ("q8_0", 8)


def row_size(type_id, n_elements):
    """ggml_row_size: bytes for n_elements of the given type."""
    if type_id not in GGML_TYPE_TRAITS:
        raise ValueError(f"unsupported/removed ggml type id {type_id}")
    blck, tsize = GGML_TYPE_TRAITS[type_id]
    return tsize * n_elements // blck


def fmt_mib(nbytes):
    return f"{nbytes / MiB:,.0f} MiB"


def fmt_gib(nbytes):
    return f"{nbytes / GiB:.2f} GiB"


# --------------------------------------------------------------------------
# GGUF parsing (header only -- we never read tensor data)
# --------------------------------------------------------------------------

class _Reader:
    def __init__(self, f):
        self.f = f

    def _unpack(self, fmt, n):
        data = self.f.read(n)
        if len(data) != n:
            raise EOFError("unexpected end of GGUF header")
        return struct.unpack(fmt, data)[0]

    def u8(self):   return self._unpack('<B', 1)
    def i8(self):   return self._unpack('<b', 1)
    def u16(self):  return self._unpack('<H', 2)
    def i16(self):  return self._unpack('<h', 2)
    def u32(self):  return self._unpack('<I', 4)
    def i32(self):  return self._unpack('<i', 4)
    def u64(self):  return self._unpack('<Q', 8)
    def i64(self):  return self._unpack('<q', 8)
    def f32(self):  return self._unpack('<f', 4)
    def f64(self):  return self._unpack('<d', 8)
    def boolean(self): return self.u8() != 0

    def string(self):
        n = self.u64()
        return self.f.read(n).decode('utf-8', errors='replace')


def _read_value(r, vtype):
    if vtype == 0:  return r.u8()
    if vtype == 1:  return r.i8()
    if vtype == 2:  return r.u16()
    if vtype == 3:  return r.i16()
    if vtype == 4:  return r.u32()
    if vtype == 5:  return r.i32()
    if vtype == 6:  return r.f32()
    if vtype == 7:  return r.boolean()
    if vtype == 8:  return r.string()
    if vtype == 9:
        elem_type = r.u32()
        count = r.u64()
        return [_read_value(r, elem_type) for _ in range(count)]
    if vtype == 10: return r.u64()
    if vtype == 11: return r.i64()
    if vtype == 12: return r.f64()
    raise ValueError(f"unknown GGUF value type {vtype}")


@dataclass
class Tensor:
    name: str
    shape: tuple
    type_id: int

    @property
    def n_elements(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def nbytes(self):
        return row_size(self.type_id, self.n_elements)

    @property
    def layer(self):
        m = re.match(r'blk\.(\d+)\.', self.name)
        return int(m.group(1)) if m else None

    @property
    def suffix(self):
        """Everything after `blk.<N>.`, or the full name for non-block tensors."""
        m = re.match(r'blk\.\d+\.(.*)$', self.name)
        return m.group(1) if m else self.name


def shard_paths(path):
    """Return every shard of a split GGUF, or just [path] for a single file."""
    m = re.match(r'^(.*)-(\d{5})-of-(\d{5})\.gguf$', path)
    if not m:
        return [path]
    base, _, total = m.group(1), int(m.group(2)), int(m.group(3))
    paths = [f"{base}-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"error: missing GGUF shard(s):\n  " + "\n  ".join(missing))
    return paths


def read_gguf(path):
    """Parse metadata + tensor info from a (possibly sharded) GGUF."""
    metadata = {}
    tensors = []
    for i, p in enumerate(shard_paths(path)):
        with open(p, 'rb') as f:
            r = _Reader(f)
            if r.u32() != GGUF_MAGIC:
                raise SystemExit(f"error: {p} is not a GGUF file")
            r.u32()                      # version
            n_tensors = r.u64()
            n_kv = r.u64()
            shard_md = {}
            for _ in range(n_kv):
                key = r.string()
                shard_md[key] = _read_value(r, r.u32())
            if i == 0:
                metadata = shard_md
            for _ in range(n_tensors):
                name = r.string()
                ndim = r.u32()
                shape = tuple(r.u64() for _ in range(ndim))
                type_id = r.u32()
                r.u64()                  # offset (unused -- sizes come from dtype)
                tensors.append(Tensor(name, shape, type_id))
    return metadata, tensors


# --------------------------------------------------------------------------
# Model shape analysis
# --------------------------------------------------------------------------

@dataclass
class ModelInfo:
    arch: str
    name: str
    n_layer: int                       # block_count (excludes nextn/MTP blocks)
    n_layer_all: int                   # highest block index seen + 1
    n_embd: int
    n_vocab: int
    n_head: int
    n_expert: int
    n_expert_used: int
    n_ff_exp: int
    recr_layers: list = field(default_factory=list)   # recurrent (SSM/KDA) layers
    attn_layers: list = field(default_factory=list)   # live attention layers
    kv_layers: list = field(default_factory=list)     # layers llama.cpp gives a KV row (incl. dead MTP)
    is_mla: bool = False
    n_nextn: int = 0                   # trailing MTP/nextn blocks declared by the model
    mtp_capable: bool = False          # nextn blocks declared AND their tensors present
    dead_layers: set = field(default_factory=set)     # blocks llama.cpp will not load
    n_embd_k_gqa: dict = field(default_factory=dict)  # layer -> K row elements
    n_embd_v_gqa: dict = field(default_factory=dict)  # layer -> V row elements (0 under MLA)
    n_embd_r: int = 0                  # recurrent conv-state row elements
    n_embd_s: int = 0                  # recurrent ssm-state row elements


# Architectures observed to leave their nextn/MTP block weights unloaded on
# llama.cpp b10277 (they log "model has unused tensor blk.<N>.* -- ignoring").
# Anything not listed here is assumed to load them, which is the safe guess.
# If you see those warnings for another arch, add it and reclaim the VRAM.
NEXTN_IGNORED_ARCHS = {"bailingmoe3"}


def _mdget(md, arch, key, default=None):
    return md.get(f"{arch}.{key}", default)


def _per_layer(value, n_layer_all, default=0):
    """Normalise a scalar-or-array hparam into a per-layer list."""
    if value is None:
        value = default
    if isinstance(value, list):
        if len(value) < n_layer_all:
            value = list(value) + [value[-1] if value else default] * (n_layer_all - len(value))
        return value
    return [value] * n_layer_all


def analyse(md, tensors, use_mtp=False):
    arch = md.get("general.architecture", "unknown")

    layer_suffixes = {}
    max_layer = -1
    for t in tensors:
        il = t.layer
        if il is None:
            continue
        max_layer = max(max_layer, il)
        layer_suffixes.setdefault(il, set()).add(t.suffix)

    n_layer_all = max_layer + 1
    n_layer = int(_mdget(md, arch, "block_count", n_layer_all))
    n_embd = int(_mdget(md, arch, "embedding_length", 0))
    n_head = int(_mdget(md, arch, "attention.head_count", 0) or 0)
    if isinstance(_mdget(md, arch, "attention.head_count"), list):
        n_head = max(_mdget(md, arch, "attention.head_count"))

    n_vocab = int(_mdget(md, arch, "vocab_size", 0) or 0)
    if not n_vocab:
        toks = md.get("tokenizer.ggml.tokens")
        n_vocab = len(toks) if toks else 0
    if not n_vocab:
        for t in tensors:
            if t.name == "token_embd.weight":
                n_vocab = int(max(t.shape))

    n_expert = int(_mdget(md, arch, "expert_count", 0) or 0)
    n_expert_used = int(_mdget(md, arch, "expert_used_count", 0) or 0)
    n_ff_exp = int(_mdget(md, arch, "expert_feed_forward_length", 0) or 0)

    # --- MTP / nextn blocks --------------------------------------------------
    # Whether llama.cpp actually LOADS the trailing multi-token-prediction
    # blocks is arch-specific, so this cannot be decided from the GGUF alone:
    #
    #   bailingmoe3          skips them -> "model has unused tensor blk.42.*
    #                        -- ignoring" (21 such warnings on Ling)
    #   longcat-flash-ngram  loads them -> longcat-flash-ngram.cpp:194,
    #                        `for (i = n_layer; i < n_layer_all; ++i)`,
    #                        zero unused-tensor warnings
    #
    # Default to counting them, because under-counting silently overcommits
    # VRAM and OOMs, while over-counting only costs some packing efficiency.
    # Archs below are ones observed to skip them on this llama.cpp build.
    n_nextn = int(_mdget(md, arch, "nextn_predict_layers", 0) or 0)
    # MTP is usable only if the quant actually kept the nextn tensors; some
    # GGUFs declare nextn_predict_layers but ship none of the weights.
    mtp_capable = bool(n_nextn) and any(".nextn." in t.name for t in tensors)
    dead_layers = set()
    if n_nextn and not use_mtp and arch in NEXTN_IGNORED_ARCHS:
        dead_layers = set(range(max(n_layer - n_nextn, 0), n_layer_all))

    # --- classify layers straight from the tensor names (arch-agnostic) ------
    # A layer is recurrent if it carries ssm_* weights; it owns a KV cache if it
    # carries real attention projections. This survives archs whose head_count_kv
    # stays non-zero on recurrent layers (e.g. qwen35moe).
    # kv_layers deliberately INCLUDES dead MTP blocks: llama.cpp refuses to load
    # their weights ("unused tensor blk.42.* -- ignoring") but still allocates a
    # KV cache row for them. Measured on Ling: KV = 306.00 MiB = 8 * 576 * 65536
    # * 34/32, i.e. 8 attention layers, of which blk.42 is one -- 7 would give
    # 267.75 MiB. Excluding it here under-counts by a full layer of KV.
    recr_layers, attn_layers, kv_layers = [], [], []
    is_mla = False
    for il in range(n_layer_all):
        sfx = layer_suffixes.get(il, set())
        has_ssm = any(s.startswith("ssm_") for s in sfx)
        has_mla = any(s.startswith("attn_kv_a_mqa") for s in sfx)
        has_attn = has_mla or any(
            s.startswith(p) for s in sfx
            for p in ("attn_k.", "attn_v.", "attn_qkv.", "attn_kv_b.")
        )
        if has_ssm:
            if il not in dead_layers:
                recr_layers.append(il)
        elif has_attn:
            kv_layers.append(il)
            if il not in dead_layers:
                attn_layers.append(il)
            is_mla = is_mla or has_mla

    # --- KV row widths, mirroring llama_hparams::n_embd_{k,v}_gqa ------------
    head_count_kv = _per_layer(_mdget(md, arch, "attention.head_count_kv"), n_layer_all,
                               default=n_head)
    default_head_dim = (n_embd // n_head) if n_head else 0
    key_len = int(_mdget(md, arch, "attention.key_length", default_head_dim) or default_head_dim)
    val_len = int(_mdget(md, arch, "attention.value_length", default_head_dim) or default_head_dim)

    n_embd_k_gqa, n_embd_v_gqa = {}, {}
    for il in kv_layers:
        n_kv_head = int(head_count_kv[il]) if il < len(head_count_kv) else 0
        if n_kv_head == 0:
            n_kv_head = 1
        n_embd_k_gqa[il] = key_len * n_kv_head
        # Under MLA llama.cpp allocates K only (has_v = !is_mla).
        n_embd_v_gqa[il] = 0 if is_mla else val_len * n_kv_head

    # --- recurrent state widths, mirroring n_embd_r() / n_embd_s() -----------
    n_embd_r = n_embd_s = 0
    if recr_layers:
        wkv_head_size = int(_mdget(md, arch, "wkv.head_size", 0) or 0)
        shortconv_l = int(_mdget(md, arch, "shortconv.l_cache", 0) or 0)
        kda_head_dim = int(_mdget(md, arch, "kda.head_dim", 0) or 0)
        d_conv = int(_mdget(md, arch, "ssm.conv_kernel", 0) or 0)
        d_inner = int(_mdget(md, arch, "ssm.inner_size", 0) or 0)
        d_state = int(_mdget(md, arch, "ssm.state_size", 0) or 0)
        n_group = int(_mdget(md, arch, "ssm.group_count", 0) or 0)
        token_shift = int(_mdget(md, arch, "token_shift_count", 2) or 2)

        if wkv_head_size:                                    # RWKV
            n_embd_r = token_shift * n_embd
            n_embd_s = n_embd * wkv_head_size
        elif shortconv_l:                                    # LFM2
            n_embd_r = n_embd * (shortconv_l - 1)
            n_embd_s = 0
        elif kda_head_dim:                                   # Kimi KDA (bailingmoe3)
            kda_inner = n_head * kda_head_dim
            n_embd_r = 3 * ((d_conv - 1) if d_conv > 0 else 3) * kda_inner
            n_embd_s = kda_head_dim * kda_head_dim * n_head
        else:                                                # Mamba / Mamba2
            n_embd_r = ((d_conv - 1) if d_conv > 0 else 0) * (d_inner + 2 * n_group * d_state)
            n_embd_s = d_state * d_inner

    return ModelInfo(
        arch=arch,
        name=md.get("general.name", "?"),
        n_layer=n_layer,
        n_layer_all=n_layer_all,
        n_embd=n_embd,
        n_vocab=n_vocab,
        n_head=n_head,
        n_expert=n_expert,
        n_expert_used=n_expert_used,
        n_ff_exp=n_ff_exp,
        recr_layers=recr_layers,
        attn_layers=attn_layers,
        kv_layers=kv_layers,
        is_mla=is_mla,
        n_nextn=n_nextn,
        mtp_capable=mtp_capable,
        dead_layers=dead_layers,
        n_embd_k_gqa=n_embd_k_gqa,
        n_embd_v_gqa=n_embd_v_gqa,
        n_embd_r=n_embd_r,
        n_embd_s=n_embd_s,
    )


def kv_cache_bytes(info, n_ctx, kv_type_id, n_seq):
    """Attention KV cache, sized the way llama_kv_cache does it."""
    # llama.cpp pads kv_size up to a multiple of 256 (or n_ubatch); 256 is close enough.
    kv_size = int(math.ceil(n_ctx / 256.0) * 256)
    total = 0
    for il in info.kv_layers:
        total += row_size(kv_type_id, info.n_embd_k_gqa[il] * kv_size)
        if info.n_embd_v_gqa[il]:
            total += row_size(kv_type_id, info.n_embd_v_gqa[il] * kv_size)
    return total


def recurrent_state_bytes(info, n_seq):
    """SSM/KDA state -- one row per sequence, always f32, independent of n_ctx."""
    per_layer = row_size(0, info.n_embd_r) + row_size(0, info.n_embd_s)
    return per_layer * len(info.recr_layers) * n_seq


# Measured on this box (llama.cpp b10277, driver 580.173, CUDA 13, RTX 5070 Ti).
# Each model was loaded with -cmoe -dev CUDA0 -c 65535 -ctk/-ctv q8_0 -fa on and
# the real nvidia-smi occupancy compared against exact weight + KV + state sizes:
#
#   model  ub     used     accounted   residual (CUDA ctx + compute buffer)
#   ling   512    5409     4386.5      1022.5 MiB
#   ling   2048   5673     4386.5      1286.5 MiB
#   kat    512    3015     2507.0       508.0 MiB
#   kat    2048   3259     2507.0       752.0 MiB
#
# Both models give a per-token slope of 0.159-0.172 MiB, and the ub-independent
# part tracks n_expert * n_moe_layers closely (934.5 vs 426.7 MiB for a 2.0x
# ratio in expert count). The fit below reproduces all four points within ~11%,
# erring high.
RESIDUAL_PER_EXPERT_LAYER = 0.0456 * MiB   # per (expert x MoE layer)
RESIDUAL_PER_UBATCH_TOKEN = 0.18 * MiB     # per ubatch token
CUDA_CONTEXT_BYTES = 264 * MiB             # backed out of the two ubatch points
# Observed directly via nvidia-smi after backend init, before any buffers:
CUDA_CONTEXT_MEASURED = (248 * MiB, 104 * MiB)   # (CUDA0, CUDA1)


def estimate_compute_buffer(info, n_ubatch, n_moe_layers):
    """
    Rough fallback estimate of llama.cpp's "CUDA0 compute buffer size".

    WARNING: this does not generalise across architectures. Measured values:

        bailingmoe3          ub=2048   1022.02 MiB
        qwen35moe            ub=2048   ~504    MiB
        longcat-flash-ngram  ub= 512   1031.96 MiB

    LongCat needs more scratch at ub=512 than Ling does at ub=2048, so no cheap
    function of (n_expert, n_ff_exp, n_embd, ub) fits all three -- an earlier
    version of this formula predicted 128 MiB for LongCat against an actual
    1032 MiB and produced a plan that OOM'd. The envelope below is therefore
    deliberately generous, and `--measure` (which reads the real number out of
    llama.cpp) is the recommended path for a tight fit.
    """
    per_token = RESIDUAL_PER_UBATCH_TOKEN * n_ubatch
    base = RESIDUAL_PER_EXPERT_LAYER * max(info.n_expert, 1) * max(n_moe_layers, 1)
    # Attention scratch scales with the widest KV row actually built per token.
    kv_row = max((info.n_embd_k_gqa.get(il, 0) + info.n_embd_v_gqa.get(il, 0)
                  for il in info.kv_layers), default=0)
    attn = 6 * n_ubatch * kv_row * 4
    return int(max(base + per_token, attn + per_token, 1024 * MiB))


COMPUTE_BUF_RE = re.compile(
    r"sched_reserve:\s+(CUDA\d+) compute buffer size\s*=\s*([0-9.]+) MiB")


# The probe below lands within ~5% of the real compute buffer; this covers the
# rest. Measured on LongCat: probe 983.96 MiB vs 1031.96 MiB for the full plan.
MEASURE_SAFETY = 1.06


def measure_compute_buffers(args, model_path, expert_layers, expert_suffixes, lookup_names,
                            mtp_enabled=False):
    """
    Load the model once under a *representative* placement and read the real
    compute buffer sizes out of llama.cpp's own log.

    The probe keeps exactly one expert layer on CUDA0 and one on CUDA1 and
    pushes everything else to CPU. That is cheap to load but reproduces the
    graph shape that actually drives the buffer size -- the count of expert
    layers barely matters, their presence on each device does. Measured on
    LongCat at -ub 512:

        -cmoe (no experts on any GPU)      762.00 MiB   <- 26% too low
        one expert layer per GPU (probe)   983.96 MiB
        the full 7+5 plan                 1031.96 MiB

    CUDA1's number came out identical (150.07 MiB) for probe and full plan.
    """
    server = args.llama_server or shutil.which("llama-server")
    if not server:
        for cand in sorted(glob.glob(os.path.expanduser("~/Workplace/llama.cpp*/build/bin/llama-server"))):
            server = cand
            break
    if not server or not os.path.exists(server):
        raise SystemExit("error: --measure needs llama-server; pass --llama-server PATH")

    # keep expert_layers[0] on CUDA0, expert_layers[1] on CUDA1, spill the rest
    probe = []
    if len(expert_layers) > 1:
        probe.append((layers_regex(expert_layers[1:2], expert_suffixes), "CUDA1"))
    if len(expert_layers) > 2:
        probe.append((layers_regex(expert_layers[2:], expert_suffixes), "CPU"))
    for pat in lookup_regex(lookup_names):
        probe.append((pat, "CPU"))

    cmd = [server, "-m", model_path, "-ngl", "99", "-dev", "CUDA0,CUDA1", "-ts", "1,0",
           "-c", str(args.ctx), "-ub", str(args.ubatch), "-fa", "on",
           "--parallel", str(args.parallel), "-lv", "5", "--no-ui",
           "--port", str(args.measure_port)]
    if probe:
        cmd += ["-ot", ",".join(f"{p}={d}" for p, d in probe)]
    if args.q8:
        cmd += ["-ctk", "q8_0", "-ctv", "q8_0"]
    if mtp_enabled:
        # the MTP draft head is part of the graph, so it must be in the probe
        cmd += ["--spec-type", "draft-mtp"]

    sys.stderr.write(f"measuring compute buffers via {os.path.basename(server)} "
                     f"(loads the model once, ~1 min)...\n")
    found = {}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    try:
        deadline = time.time() + args.measure_timeout
        for line in proc.stdout:
            m = COMPUTE_BUF_RE.search(line)
            if m:
                found[m.group(1)] = int(float(m.group(2)) * MiB)
            if "CUDA0" in found and "CUDA1" in found:
                break
            if "out of memory" in line or "exiting due to" in line:
                break
            if time.time() > deadline:
                sys.stderr.write("warning: measurement timed out\n")
                break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()

    if "CUDA0" not in found:
        raise SystemExit("error: could not read a CUDA0 compute buffer size from llama-server; "
                         "run it manually with -lv 5 and pass --compute-buf")
    raw = dict(found)
    found = {k: int(v * MEASURE_SAFETY) for k, v in found.items()}
    sys.stderr.write(
        "measured: " + ", ".join(
            f"{k} {fmt_mib(raw[k])} -> {fmt_mib(found[k])} (x{MEASURE_SAFETY})"
            for k in sorted(raw)) + "\n\n")
    return found


# --------------------------------------------------------------------------
# GPU discovery
# --------------------------------------------------------------------------

@dataclass
class Gpu:
    index: int
    name: str
    total_bytes: int


def detect_gpus():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True).stdout
    except Exception as e:
        raise SystemExit(f"error: could not run nvidia-smi ({e}); pass --vram to override")
    gpus = []
    for line in out.strip().splitlines():
        idx, name, total = [p.strip() for p in line.split(",")]
        gpus.append(Gpu(int(idx), name, int(total) * MiB))
    if not gpus:
        raise SystemExit("error: nvidia-smi reported no GPUs")
    return gpus


# --------------------------------------------------------------------------
# Placement
# --------------------------------------------------------------------------

@dataclass
class Plan:
    gpu0_expert_layers: list
    gpu1_expert_layers: list
    cpu_expert_layers: list
    lookup_gpu1: list                  # lookup tables placed on CUDA1
    lookup_cpu: list                   # lookup tables spilled to CPU
    gpu0_used: int
    gpu1_used: int
    cpu_bytes: int
    gpu0_budget: int
    gpu1_budget: int
    core_bytes: int
    kv_bytes: int
    recr_bytes: int
    compute_bytes: int
    feasible: bool = True

    @property
    def n_gpu_expert_layers(self):
        return len(self.gpu0_expert_layers) + len(self.gpu1_expert_layers)


def pack(expert_layers, expert_bytes, lookup_bytes, core_bytes,
         fixed0, fixed1, gpu0_budget, gpu1_budget):
    """
    Place, in strict priority order:
      1. KV + recurrent state + compute buffer, then the core weights, on CUDA0.
         If those alone overflow CUDA0 the configuration is infeasible.
      2. Whole per-layer expert sets: CUDA0's leftover, then CUDA1, then CPU.
      3. Lookup tables (get_rows only) into whatever is still left, else CPU.

    Experts outrank lookup tables because a GB of experts on CPU costs real
    tokens/sec, while a GB of lookup table on CPU costs a few microseconds per
    token. That ordering also reproduces the token_embd rule: it keeps its GPU
    slot only when doing so does not displace an expert layer.
    """
    used0 = fixed0 + core_bytes
    used1 = fixed1
    if used0 > gpu0_budget:
        return None

    gpu0, gpu1, cpu = [], [], []
    for il in expert_layers:
        size = expert_bytes[il]
        if used0 + size <= gpu0_budget:
            gpu0.append(il)
            used0 += size
        elif used1 + size <= gpu1_budget:
            gpu1.append(il)
            used1 += size
        else:
            cpu.append(il)

    lk_gpu1, lk_cpu = [], []
    for name, size in sorted(lookup_bytes.items(), key=lambda kv: -kv[1]):
        if used0 + size <= gpu0_budget:
            used0 += size                      # stays on CUDA0, no -ot needed
        elif used1 + size <= gpu1_budget:
            lk_gpu1.append(name)
            used1 += size
        else:
            lk_cpu.append(name)
    return gpu0, gpu1, cpu, lk_gpu1, lk_cpu, used0, used1


# --------------------------------------------------------------------------
# Regex emission
# --------------------------------------------------------------------------

# Tensors llama.cpp only ever reads through ggml_get_rows -- embedding lookup
# tables. Placing one on CPU costs a gather of a few KB plus a small host->device
# copy per token, i.e. microseconds, so they are the cheapest thing to evict.
#
#   token_embd.weight            the input embedding
#   ngram_embd.<j>.weight        LongCat-Flash n-gram tables; 12 x ~1.4 GiB here.
#                                src/models/longcat-flash-ngram.cpp:359 ->
#                                ggml_get_rows(ctx0, model.ngram_embd[j], ...).
#                                The matmul next to it uses ngram_proj, which is
#                                tiny and stays on GPU as a core tensor.
#   *.nextn.embed_tokens.weight  MTP block input embedding
#
# NOTE: a model with *tied* embeddings reuses token_embd as the output
# projection. Callers must check for a separate output.weight before evicting
# token_embd, or a full vocab matmul lands on the CPU every token.
LOOKUP_TENSOR_RE = re.compile(
    r"^(token_embd\.weight|ngram_embd\.\d+\.weight|.*\.nextn\.embed_tokens\.weight)$")


def is_lookup_tensor(name):
    return bool(LOOKUP_TENSOR_RE.match(name))


def lookup_regex(names):
    """
    Compact `-ot` pattern for a set of lookup tensors. Whole numbered families
    (ngram_embd.0..N) collapse to a prefix; anything else is listed exactly.
    """
    names = set(names)
    parts, remaining = [], set(names)
    for family in ("ngram_embd",):
        members = {n for n in remaining if n.startswith(family + ".")}
        if members and len(members) == len(FAMILY_SIZES.get(family, members)):
            parts.append(rf"^{re.escape(family)}\.")
            remaining -= members
    for n in sorted(remaining):
        parts.append(rf"^{re.escape(n)}$")
    return parts


FAMILY_SIZES = {}


def layers_regex(layers, suffixes):
    """
    Build one `-ot` pattern for a set of layers.

    llama.cpp matches with std::regex_search (substring) and takes the FIRST
    matching -ot arg, so we anchor both ends of the layer index with `\\.` and
    terminate with `$` to make each pattern exact.
    """
    idx = "|".join(str(i) for i in sorted(layers))
    sfx = "|".join(re.escape(s) for s in sorted(suffixes))
    return rf"blk\.({idx})\.({sfx})$"


def build_ot_flags(plan, expert_suffixes):
    flags = []
    if plan.gpu1_expert_layers:
        flags.append((layers_regex(plan.gpu1_expert_layers, expert_suffixes), "CUDA1"))
    for pat in lookup_regex(plan.lookup_gpu1):
        flags.append((pat, "CUDA1"))
    if plan.cpu_expert_layers:
        flags.append((layers_regex(plan.cpu_expert_layers, expert_suffixes), "CPU"))
    for pat in lookup_regex(plan.lookup_cpu):
        flags.append((pat, "CPU"))
    return flags


def render_ot(flags):
    """
    Render every override as ONE `-ot` argument.

    llama.cpp b10277 parses `-ot` as `<pattern>=<buft>,...` and warns
    "argument '-ot' specified multiple times ... only last value will be used"
    if the flag is repeated -- repeating it silently drops all but the final
    pattern. common/arg.cpp splits on ',' then on the first '=', so patterns
    must not contain a comma (ours never do).
    """
    return "-ot '" + ",".join(f"{p}={d}" for p, d in flags) + "'"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Compute llama.cpp -ot tensor overrides for large MoE GGUF models.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="path to the .gguf (first shard is fine)")
    ap.add_argument("-q8", "--q8", action="store_true",
                    help="quantise the KV cache to q8_0 (default: f16)")
    ap.add_argument("-c", "--ctx", type=int, default=65535, help="context size (default 65535)")
    ap.add_argument("-ub", "--ubatch", type=int, default=512,
                    help="physical batch size, drives the compute-buffer estimate (default 512)")
    ap.add_argument("--parallel", type=int, default=4,
                    help="number of server slots; sizes the recurrent state. llama-server's "
                         "default is auto -> 4 slots with kv_unified, so 4 is the default here too.")
    ap.add_argument("--no-mtp", dest="mtp", action="store_false", default=True,
                    help="do NOT use the model's MTP/nextn blocks. By default, if the GGUF ships "
                         "nextn tensors the plan budgets for them and the command carries "
                         "--spec-type draft-mtp.")
    ap.add_argument("--server-bin", default="./llama-server",
                    help="how the emitted command invokes llama-server (default ./llama-server)")
    ap.add_argument("--reserve0", type=int, default=1024,
                    help="MiB reserved on CUDA0 for the CUDA context plus allocator headroom "
                         "(default 1024). The context itself measures ~248 MiB; the rest is "
                         "headroom, because the compute buffer is a single ~1 GiB contiguous "
                         "cudaMalloc that fails well before VRAM is nominally exhausted. "
                         "512 was measured to OOM with 306 MiB nominally still free.")
    ap.add_argument("--reserve1", type=int, default=512,
                    help="MiB reserved on CUDA1 for the CUDA context plus allocator headroom "
                         "(default 512; the context itself measures ~104 MiB)")
    ap.add_argument("--compute-buf1-frac", type=float, default=0.35,
                    help="CUDA1's compute buffer as a fraction of CUDA0's (default 0.35; "
                         "measured 328.06 vs 1022.02 MiB on Ling at -ub 2048)")
    ap.add_argument("--measure", action="store_true",
                    help="load the model once with -cmoe and read the REAL compute buffer sizes "
                         "from llama.cpp instead of estimating. Strongly recommended: the "
                         "estimate does not generalise across architectures.")
    ap.add_argument("--llama-server", default=None,
                    help="path to llama-server for --measure (default: PATH, then ~/Workplace/llama.cpp*/build/bin)")
    ap.add_argument("--measure-port", type=int, default=18099, help="port used by --measure")
    ap.add_argument("--measure-timeout", type=int, default=600,
                    help="seconds to wait for --measure (default 600)")
    ap.add_argument("--compute-buf", type=int, default=None,
                    help="MiB for the CUDA0 compute buffer; overrides estimate and --measure. "
                         "Read the real value from llama-server -lv 5 'compute buffer size'.")
    ap.add_argument("--vram", default=None,
                    help="override detected VRAM, MiB, comma-separated per device, e.g. 16303,8188")
    ap.add_argument("--flags-only", action="store_true", help="print only the -ot flags")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"error: no such file: {args.model}")

    md, tensors = read_gguf(args.model)
    info = analyse(md, tensors, use_mtp=args.mtp)
    # MTP is on whenever the model can do it, unless --no-mtp. It costs the
    # nextn block's weights in VRAM, which the plan below budgets for.
    mtp_enabled = args.mtp and info.mtp_capable

    gpus = detect_gpus()
    if args.vram:
        overrides = [int(x) * MiB for x in args.vram.split(",")]
        for g, v in zip(gpus, overrides):
            g.total_bytes = v
    if len(gpus) < 2:
        raise SystemExit(f"error: this script assumes 2 GPUs, found {len(gpus)}")

    # --- split tensors into expert sets / lookup tables / core ---------------
    expert_bytes = {}
    expert_suffixes = set()
    lookup_bytes = {}
    core_bytes = 0
    dead_bytes = 0
    for t in tensors:
        if t.layer in info.dead_layers:
            dead_bytes += t.nbytes
            continue
        if "_exps." in t.name:
            il = t.layer
            expert_bytes[il] = expert_bytes.get(il, 0) + t.nbytes
            expert_suffixes.add(t.suffix)
        elif is_lookup_tensor(t.name):
            lookup_bytes[t.name] = t.nbytes
        else:
            core_bytes += t.nbytes

    expert_layers = sorted(expert_bytes)
    total_expert_bytes = sum(expert_bytes.values())
    weights_bytes = core_bytes + sum(lookup_bytes.values()) + total_expert_bytes

    # A model with tied embeddings reuses token_embd as the output projection;
    # evicting it would put a full vocab matmul on the CPU every token. Resolve
    # this before measuring, so the probe uses the same placement as the plan.
    if not any(t.name == "output.weight" for t in tensors):
        core_bytes += lookup_bytes.pop("token_embd.weight", 0)
    FAMILY_SIZES["ngram_embd"] = {n for n in lookup_bytes if n.startswith("ngram_embd.")}

    kv_name, kv_type = KV_TYPE_Q8_0 if args.q8 else KV_TYPE_F16
    kv_bytes = kv_cache_bytes(info, args.ctx, kv_type, args.parallel)
    recr_bytes = recurrent_state_bytes(info, args.parallel)
    measured = {}
    if args.compute_buf is not None:
        compute_bytes = args.compute_buf * MiB
        compute_src = "given"
    elif args.measure:
        measured = measure_compute_buffers(args, args.model, expert_layers,
                                           expert_suffixes, list(lookup_bytes),
                                           mtp_enabled=mtp_enabled)
        compute_bytes = measured["CUDA0"]
        compute_src = "measured"
    else:
        compute_bytes = estimate_compute_buffer(info, args.ubatch, len(expert_layers))
        compute_src = "estimated"

    gpu0_budget = gpus[0].total_bytes - args.reserve0 * MiB
    gpu1_budget = gpus[1].total_bytes - args.reserve1 * MiB
    fixed0 = kv_bytes + recr_bytes + compute_bytes
    # CUDA1 holds only expert weights, but it still takes part in the graph and
    # so allocates its own compute buffer. Measured 328.06 MiB (Ling) and
    # 150.07 MiB (LongCat), so a fraction of CUDA0's with a floor.
    fixed1 = max(int(compute_bytes * args.compute_buf1_frac), 256 * MiB)
    if "CUDA1" in measured:
        fixed1 = max(measured["CUDA1"], 256 * MiB)

    packed = pack(expert_layers, expert_bytes, lookup_bytes, core_bytes,
                  fixed0, fixed1, gpu0_budget, gpu1_budget)

    if packed is None:
        # KV + state + compute + core alone overflow CUDA0. There is no valid
        # -ot for this; emitting one would produce a command that cannot load.
        need = fixed0 + core_bytes
        e = sys.stderr.write
        e(f"error: this configuration cannot fit on CUDA0.\n\n")
        e(f"  KV cache @ {args.ctx} ({kv_name})".ljust(38) + f"{fmt_mib(kv_bytes):>12}\n")
        e(f"  recurrent state ({args.parallel} seq)".ljust(38) + f"{fmt_mib(recr_bytes):>12}\n")
        e(f"  compute buffer".ljust(38) + f"{fmt_mib(compute_bytes):>12}\n")
        e(f"  non-expert weights that must be resident".ljust(38)
          + f"{fmt_mib(core_bytes):>12}\n")
        e(f"  {'':36s}{'-' * 12:>12}\n")
        e(f"  required".ljust(38) + f"{fmt_mib(need):>12}\n")
        e(f"  CUDA0 budget".ljust(38) + f"{fmt_mib(gpu0_budget):>12}"
          f"   (short by {fmt_mib(need - gpu0_budget)})\n\n")
        e("These tensors are needed on every token and cannot be moved without\n"
          "wrecking throughput, so no tensor override can rescue this. Options:\n")
        if not args.q8:
            e(f"  -q8                 halves the KV cache "
              f"({fmt_mib(kv_bytes)} -> ~{fmt_mib(kv_bytes * 17 // 32)})\n")
        smaller = max(args.ctx // 2, 4096)
        e(f"  -c {smaller:<16} halves the KV cache\n")
        e(f"  --parallel 1        shrinks the recurrent state\n")
        e(f"  --reserve0 512      reclaims allocator headroom (may OOM at graph reserve)\n")
        raise SystemExit(2)

    g0, g1, cpu_layers, lk_gpu1, lk_cpu, used0, used1 = packed
    cpu_bytes = (sum(expert_bytes[i] for i in cpu_layers)
                 + sum(lookup_bytes[n] for n in lk_cpu))
    plan = Plan(g0, g1, cpu_layers, lk_gpu1, lk_cpu, used0, used1, cpu_bytes,
                gpu0_budget, gpu1_budget, core_bytes, kv_bytes, recr_bytes, compute_bytes)

    fits_entirely = not plan.cpu_expert_layers and not plan.lookup_cpu

    ot_flags = build_ot_flags(plan, expert_suffixes)

    # --- output -------------------------------------------------------------
    if args.json:
        print(json.dumps({
            "model": os.path.basename(args.model),
            "architecture": info.arch,
            "n_layer_all": info.n_layer_all,
            "recurrent_layers": len(info.recr_layers),
            "attention_layers": len(info.attn_layers),
            "is_mla": info.is_mla,
            "fits_entirely_on_gpu": fits_entirely,
            "kv_type": kv_name,
            "bytes": {
                "weights_total": weights_bytes,
                "core": plan.core_bytes,
                "experts_total": total_expert_bytes,
                "kv_cache": kv_bytes,
                "recurrent_state": recr_bytes,
                "compute_buffer": compute_bytes,
                "cpu": plan.cpu_bytes,
            },
            "expert_layers": {
                "CUDA0": plan.gpu0_expert_layers,
                "CUDA1": plan.gpu1_expert_layers,
                "CPU": plan.cpu_expert_layers,
            },
            "lookup_tables": {
                "CUDA0": sorted(set(lookup_bytes) - set(plan.lookup_gpu1) - set(plan.lookup_cpu)),
                "CUDA1": sorted(plan.lookup_gpu1),
                "CPU": sorted(plan.lookup_cpu),
            },
            "ot": [{"pattern": p, "device": d} for p, d in ot_flags],
        }, indent=2))
        return

    if args.flags_only:
        if fits_entirely:
            print("# no tensor override needed -- everything fits on CUDA0 + CUDA1")
        else:
            print(render_ot(ot_flags))
        return

    e = sys.stderr.write
    e(f"Model         : {os.path.basename(args.model)}\n")
    e(f"Architecture  : {info.arch}  ({info.name})\n")
    n_live = info.n_layer_all - len(info.dead_layers)
    n_dense = n_live - len(info.recr_layers) - len(info.attn_layers)
    e(f"Blocks        : {n_live} live -> "
      f"{len(info.recr_layers)} recurrent (SSM/KDA), "
      f"{len(info.attn_layers)} attention{' [MLA]' if info.is_mla else ''}"
      + (f", {n_dense} other" if n_dense else "") + "\n")
    e(f"MoE           : {info.n_expert} experts, {info.n_expert_used} used/token, "
      f"{len(expert_layers)} layers carry expert tensors\n")
    if mtp_enabled:
        e(f"MTP           : ENABLED -- {info.n_nextn} nextn block(s) present, so the command\n"
          f"                carries --spec-type draft-mtp and the plan budgets for their\n"
          f"                weights. Use --no-mtp to turn this off.\n")
    elif info.mtp_capable and info.dead_layers:
        e(f"MTP           : available but disabled (--no-mtp); {len(info.dead_layers)} nextn block(s)\n"
          f"                ({fmt_gib(dead_bytes)}) are not loaded by {info.arch}\n")
    elif info.mtp_capable:
        e(f"MTP           : available but disabled (--no-mtp); {info.arch} still loads the\n"
          f"                nextn weights, so they are counted anyway\n")
    elif info.n_nextn:
        e(f"MTP           : declared ({info.n_nextn} block(s)) but the nextn tensors are not in\n"
          f"                this GGUF, so MTP is unusable\n")
    e(f"Context       : {args.ctx}   KV type: {kv_name}   ubatch: {args.ubatch}\n")
    e("\n")

    total_lookup = sum(lookup_bytes.values())
    e("Memory accounting\n")
    e(f"  model weights, total           {fmt_gib(weights_bytes):>12}\n")
    e(f"    non-expert (core)            {fmt_gib(core_bytes):>12}\n")
    if total_lookup:
        e(f"    lookup tables (get_rows)     {fmt_gib(total_lookup):>12}"
          f"   ({len(lookup_bytes)} tensors, evictable)\n")
    e(f"    expert tensors               {fmt_gib(total_expert_bytes):>12}"
      f"   ({len(expert_layers)} layers, {fmt_mib(total_expert_bytes / max(len(expert_layers),1))}/layer)\n")
    e(f"  KV cache @ {args.ctx} ({kv_name})".ljust(33) + f"{fmt_mib(kv_bytes):>12}\n")
    e(f"  recurrent state ({args.parallel} seq)".ljust(33) + f"{fmt_mib(recr_bytes):>12}\n")
    e(f"  compute buffer  ({compute_src})".ljust(33) + f"{fmt_mib(compute_bytes):>12}\n")
    if compute_src == "estimated":
        e("      ^ a coarse upper bound; re-run with --measure for the real value\n")
    e("\n")

    for i, (gpu, budget, used) in enumerate((
            (gpus[0], gpu0_budget, plan.gpu0_used),
            (gpus[1], gpu1_budget, plan.gpu1_used))):
        reserve = args.reserve0 if i == 0 else args.reserve1
        e(f"  CUDA{i} ({gpu.name})\n")
        e(f"    total VRAM                   {fmt_mib(gpu.total_bytes):>12}\n")
        e(f"    - reserved (CUDA ctx)        {fmt_mib(reserve * MiB):>12}\n")
        e(f"    = budget                     {fmt_mib(budget):>12}\n")
        lk_here = ([n for n in lookup_bytes
                    if n not in plan.lookup_gpu1 and n not in plan.lookup_cpu]
                   if i == 0 else plan.lookup_gpu1)
        if i == 0:
            e(f"      KV + recurrent + compute   {fmt_mib(fixed0):>12}\n")
            e(f"      core weights               {fmt_mib(plan.core_bytes):>12}\n")
            e(f"      {len(plan.gpu0_expert_layers)} expert layers".ljust(33)
              + f"{fmt_mib(sum(expert_bytes[j] for j in plan.gpu0_expert_layers)):>12}\n")
        else:
            e(f"      compute buffer             {fmt_mib(fixed1):>12}\n")
            e(f"      {len(plan.gpu1_expert_layers)} expert layers".ljust(33)
              + f"{fmt_mib(sum(expert_bytes[j] for j in plan.gpu1_expert_layers)):>12}\n")
        if lk_here:
            e(f"      {len(lk_here)} lookup tables".ljust(33)
              + f"{fmt_mib(sum(lookup_bytes[n] for n in lk_here)):>12}\n")
        e(f"    used of budget               {fmt_mib(used):>12}"
          f"   ({100.0 * used / budget:.1f}%)\n")
        ctx_bytes = CUDA_CONTEXT_MEASURED[0] if i == 0 else CUDA_CONTEXT_MEASURED[1]
        e(f"    predicted actual occupancy   {fmt_mib(used + ctx_bytes):>12}"
          f"   of {fmt_mib(gpu.total_bytes)}\n")
    e(f"  CPU\n")
    e(f"    {len(plan.cpu_expert_layers)} expert layers"
      + (f" + {len(plan.lookup_cpu)} lookup tables" if plan.lookup_cpu else "") + "\n")
    e(f"    total                        {fmt_gib(plan.cpu_bytes):>12}\n")
    e("\n")

    llama_args = ["-ngl 99", "-dev CUDA0,CUDA1", f"-c {args.ctx}", f"-ub {args.ubatch}"]
    if args.q8:
        llama_args += ["-ctk q8_0", "-ctv q8_0"]
    if mtp_enabled:
        llama_args += ["--spec-type draft-mtp"]

    head = f"{args.server_bin} -m " + args.model
    if fits_entirely:
        e("=> No tensor override needed: the whole model + KV cache fits on CUDA0 + CUDA1.\n")
        e("   Let llama.cpp do its own layer split.\n\n")
        print(" \\\n  ".join([head] + llama_args + ["-fa on"]))
        return

    # -ts 1,0 pins every layer to CUDA0 so the whole KV cache lands there;
    # -ot then relocates expert weights only.
    llama_args.insert(2, "-ts 1,0")
    e(f"=> {len(plan.cpu_expert_layers)} expert layer(s) must live on CPU.\n")
    e("   -ts 1,0 pins all layers (hence the whole KV cache) to CUDA0;\n")
    e("   -ot then moves expert weights out to CUDA1 / CPU.\n")

    ram_avail = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    ram_avail = int(line.split()[1]) * 1024
    except OSError:
        pass
    if ram_avail and plan.cpu_bytes < ram_avail * 0.9:
        e(f"   {fmt_gib(plan.cpu_bytes)} lands on CPU and you have {fmt_gib(ram_avail)} available;\n")
        e("   add --no-mmap for better performance (llama.cpp recommends it with CPU overrides).\n")
    elif ram_avail:
        e(f"   {fmt_gib(plan.cpu_bytes)} lands on CPU vs {fmt_gib(ram_avail)} available RAM --\n")
        e("   keep mmap enabled (do NOT use --no-mmap) or you will swap.\n")
    e("\n")

    parts = [head] + llama_args + [render_ot(ot_flags), "-fa on"]
    print(" \\\n  ".join(parts))


if __name__ == "__main__":
    main()
