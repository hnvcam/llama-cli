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

Dense models (no `*_exps.*` tensors) have nothing `-ot` can usefully relocate,
since every weight is read on every token. For those the script switches to a
layer-split planner instead: it walks every possible split point, prices each
card SEPARATELY (weights + its share of the KV cache + its share of the
recurrent state + its own full compute buffer), and reports the largest `-c`
that fits together with the `-ts` that balances the two cards. See dense_report().

Pricing the two cards as one pooled budget -- which this script used to do --
is the trap: llama.cpp fills the small card first, so a plan that looks like it
has gigabytes spare can still die on the small card's compute buffer. And that
compute buffer is not a constant: it carries an `n_kv * n_ubatch * 4 * n_seq`
KQ mask, so it grows with `-c` on BOTH cards. See DENSE_COMPUTE_BASE.

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
    kv_layers: list = field(default_factory=list)     # layers carrying attention weights (incl. dead MTP)
    main_kv_layers: list = field(default_factory=list) # layers the MAIN context gives a KV row
    is_mla: bool = False
    n_nextn: int = 0                   # trailing MTP/nextn blocks declared by the model
    mtp_capable: bool = False          # nextn blocks declared AND their tensors present
    dead_layers: set = field(default_factory=set)     # blocks llama.cpp will not load
    n_embd_k_gqa: dict = field(default_factory=dict)  # layer -> K row elements
    n_embd_v_gqa: dict = field(default_factory=dict)  # layer -> V row elements (0 under MLA)
    n_embd_r: int = 0                  # recurrent conv-state row elements
    n_embd_s: int = 0                  # recurrent ssm-state row elements
    n_swa: int = 0                     # sliding-window width, 0 = no SWA
    swa_layers: list = field(default_factory=list)    # layers on the small SWA cache


# Architectures observed to leave their nextn/MTP block weights unloaded on
# llama.cpp b10277 (they log "model has unused tensor blk.<N>.* -- ignoring").
# Anything not listed here is assumed to load them, which is the safe guess.
# If you see those warnings for another arch, add it and reclaim the VRAM.
#
#   bailingmoe3  Ling 3.0: 21 warnings for blk.42.*
#   qwen35       Qwen3.8-27B: 15 warnings for blk.64.* (334.75 MiB)
#
# Note what this list does NOT mean. qwen35.cpp:41 reads
#     int mtp_flags = !ml.load_mtp ? TENSOR_SKIP : 0;
# and common.cpp:1689 sets load_mtp from `--spec-type draft-mtp`, so those
# warnings appear precisely BECAUSE MTP was not requested. The arch supports
# MTP fine; the block is dead weight only while MTP is off.
NEXTN_IGNORED_ARCHS = {"bailingmoe3", "qwen35"}

# --spec-type draft-mtp on a recurrent model also has to checkpoint the SSM
# state so a rejected draft can be rolled back: common.h:391 sets
# cparams.n_rs_seq = speculative.draft.n_max (default 3), and the recurrent
# buffer is then sized for (1 + n_rs_seq) copies. llama-context.cpp:105 clamps
# it back to 0 for archs outside this list (llama-arch.cpp:1028).
#
# Measured on Qwen3.8-27B, 4 slots:  598.50 MiB off -> 2394.00 MiB on. That is
# +1795.50 MiB, by far the largest single cost of turning MTP on, and nothing
# in the old accounting knew about it.
RS_ROLLBACK_ARCHS = {"qwen35", "qwen35moe", "deepseek4",
                     "nemotron_h", "nemotron_h_moe", "lfm2", "lfm2moe"}


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
    # kv_layers deliberately INCLUDES dead MTP blocks: on bailingmoe3 llama.cpp
    # refuses to load their weights ("unused tensor blk.42.* -- ignoring") but
    # still allocates a KV cache row for them. Measured on Ling: KV = 306.00 MiB
    # = 8 * 576 * 65536 * 34/32, i.e. 8 attention layers, of which blk.42 is one
    # -- 7 would give 267.75 MiB. Excluding it there under-counts by a full layer.
    # main_kv_layers, built below, is the subset the MAIN context actually caches;
    # that is what sizing should use, and on some archs it is smaller.
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

    # Whether the MAIN context also caches the nextn blocks is arch-specific.
    # llama-model.cpp installs `filter = [](il) { return il >= n_layer; }` on the
    # MTP context for MTP_KV_FILTERED_ARCHS, which by construction keeps them out
    # of the main one. Measured on Qwen3.8-27B (qwen35, 16 of 65 blocks are full
    # attention, blk.64 is nextn):
    #   llama_kv_cache: size = 1088.00 MiB (32768 cells, 16 layers, 4/1 seqs)
    # i.e. 16, not 17 -- blk.64 gets no row. On bailingmoe3, which has no such
    # filter, the dead nextn block DOES get one (306.00 MiB = 8 layers on Ling,
    # blk.42 among them), so the two cases must be kept apart.
    nextn_blocks = set(range(max(n_layer - n_nextn, 0), n_layer_all)) if n_nextn else set()
    if arch in MTP_KV_FILTERED_ARCHS:
        main_kv_layers = [il for il in kv_layers if il not in nextn_blocks]
    else:
        main_kv_layers = list(kv_layers)

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

    # --- sliding-window attention -------------------------------------------
    # Layers llama.cpp puts on the small SWA cache instead of the n_ctx one.
    # Restricted to main_kv_layers: a layer with no KV row cannot be on either.
    n_swa, swa_all = resolve_swa(md, arch, n_layer, n_layer_all, recr_layers)
    swa_layers = [il for il in main_kv_layers if il in set(swa_all)]
    if not swa_layers:
        n_swa = 0

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
        main_kv_layers=main_kv_layers,
        is_mla=is_mla,
        n_nextn=n_nextn,
        mtp_capable=mtp_capable,
        dead_layers=dead_layers,
        n_embd_k_gqa=n_embd_k_gqa,
        n_embd_v_gqa=n_embd_v_gqa,
        n_embd_r=n_embd_r,
        n_embd_s=n_embd_s,
        n_swa=n_swa,
        swa_layers=swa_layers,
    )


# --------------------------------------------------------------------------
# Sliding-window attention
# --------------------------------------------------------------------------
#
# An arch that sets hparams.swa_type != NONE does NOT get one n_ctx-cell KV row
# per layer. llama_model::create_memory builds a llama_kv_cache_iswa instead
# (llama-model.cpp:2312 hybrid, :2383 pure attention), which is TWO caches with
# a layer filter (llama-kv-cache-iswa.cpp:44-105):
#
#   base   non-SWA layers,  size_base = n_ctx_seq
#   swa    SWA layers,      size_swa  = GGML_PAD(min(size_base,
#                                         n_swa*(unified ? n_seq_max : 1)
#                                         + n_ubatch), 256)
#
# For gpt-oss (n_swa = 128, every other layer SWA) at -c 65535 -ub 512 that is
# 1024 cells against 65536, so half the model's KV costs 1.5% of what a flat
# n_ctx row would. Measured, gpt-oss-120b Q8_0, -c 65535 -ub 512, f16 KV:
#
#   llama_kv_cache_iswa: creating non-SWA KV cache, size = 65536 cells
#   llama_kv_cache: size = 2304.00 MiB (65536 cells, 18 layers, 4/1 seqs)
#   llama_kv_cache_iswa: creating     SWA KV cache, size = 1024 cells
#   llama_kv_cache: size =   36.00 MiB ( 1024 cells, 18 layers, 4/1 seqs)
#
# 2340 MiB, where sizing all 36 layers flat gives 4608 MiB. The 2268 MiB of
# phantom KV cost two whole expert layers of CUDA0 packing, which is what this
# script used to do.
#
# Each cache is allocated per stream: llama-kv-cache.cpp:82 sets
# n_stream = unified ? 1 : n_seq_max, and :231 allocates [row, kv_size,
# n_stream]. llama-context.cpp:292 sets n_ctx_seq = n_ctx / n_seq_max when NOT
# unified, so the base cache totals n_ctx cells either way -- but the SWA cache
# does not, because its size is a floor: it totals size_swa * n_stream.
#
# llama-server's default is -np -1 ("auto"), which is n_parallel = 4 with
# kv_unified = true (tools/server/server.cpp:152-158), matching this script's
# --parallel 4 default. --swa-full (llama.cpp's --swa-full) sizes the SWA cache
# like the base one and is priced by passing swa_full=True.


@dataclass(frozen=True)
class SwaRule:
    """
    How one arch's load_arch_hparams() decides is_swa(il), condensed from
    src/models/<arch>.cpp. An arch missing from SWA_ARCHS is priced with no SWA
    at all, which over-counts rather than OOMs.
    """
    period: int = 2             # swa_period default when the GGUF omits the pattern key
    dense_first: bool = False   # set_swa_pattern(period, dense_first)
    n_swa_default: int = 0      # n_swa the loader hard-codes before reading the key
    needs_key: bool = True      # SWA only when attention.sliding_window is present and > 0
    array_only: bool = False    # loader reads the pattern straight into is_swa_impl
    all_layers: bool = False    # every attention layer is SWA
    non_recurrent: bool = False # SWA on exactly the non-recurrent layers (lfm2)


# Verified against src/models/*.cpp on this llama.cpp checkout. The comment on
# each line is the file that justifies it.
SWA_ARCHS = {
    "gpt-oss":          SwaRule(period=2),                        # openai-moe.cpp:8
    "gemma2":           SwaRule(period=2, n_swa_default=4096, needs_key=False),  # gemma2.cpp:4
    "gemma3":           SwaRule(period=6),                        # gemma3.cpp:7
    "gemma3n":          SwaRule(period=5),                        # gemma3n.cpp:4
    "gemma4":           SwaRule(array_only=True),                 # gemma4.cpp:5
    "gemma4-assistant": SwaRule(array_only=True),                 # gemma4-assistant.cpp:7
    "gemma-embedding":  SwaRule(period=6),                        # gemma-embedding.cpp:5
    "cohere2":          SwaRule(period=4),                        # cohere2.cpp:5
    "cohere2moe":       SwaRule(period=4, dense_first=True),      # cohere2moe.cpp:33
    "exaone-moe":       SwaRule(period=4, n_swa_default=128, needs_key=False),   # exaone-moe.cpp:5
    "llama4":           SwaRule(period=4, n_swa_default=8192),    # llama4.cpp:14,19
    "afmoe":            SwaRule(period=4),                        # afmoe.cpp:17
    "laguna":           SwaRule(period=4, dense_first=True),      # laguna.cpp:41
    "deepseek4":        SwaRule(all_layers=True),                 # deepseek4.cpp:68
    "dflash":           SwaRule(array_only=True),                 # dflash.cpp:74
    "graniteswitch":    SwaRule(array_only=True),                 # granite-swa.cpp:17
    "modern-bert":      SwaRule(period=3, dense_first=True),      # modern-bert.cpp:10
    "lfm2":             SwaRule(non_recurrent=True),              # lfm2.cpp:29
    "lfm2moe":          SwaRule(non_recurrent=True),              # lfm2.cpp:29
    # exaone4 arms SWA only at n_layer() == 64 (exaone4.cpp:7); handled below.
    "exaone4":          SwaRule(period=4, n_swa_default=4096, needs_key=False),
}


def swa_pattern_flags(period, dense_first, n_layer, n_layer_all):
    """llama_hparams::set_swa_pattern (llama-hparams.cpp)."""
    flags = [False] * n_layer_all
    for il in range(min(n_layer, n_layer_all)):
        if dense_first:
            flags[il] = period == 0 or (il % period != 0)
        else:
            flags[il] = period == 0 or (il % period < period - 1)
    return flags


def resolve_swa(md, arch, n_layer, n_layer_all, recr_layers):
    """
    -> (n_swa, [layer indices that are SWA]).

    Returns (0, []) for any arch this script cannot pin down from the GGUF, so
    those keep the old flat-n_ctx accounting.
    """
    rule = SWA_ARCHS.get(arch)
    if rule is None:
        return 0, []
    if arch == "exaone4" and n_layer != 64:
        return 0, []

    n_swa = int(_mdget(md, arch, "attention.sliding_window", 0) or 0)
    if not n_swa:
        if rule.needs_key:
            return 0, []
        n_swa = rule.n_swa_default
    if n_swa <= 0:
        return 0, []

    pattern = _mdget(md, arch, "attention.sliding_window_pattern")
    if isinstance(pattern, list):
        # get_key_or_arr with a per-layer array -> is_swa_impl verbatim
        flags = [bool(v) for v in pattern][:n_layer_all]
        flags += [False] * (n_layer_all - len(flags))
    elif rule.array_only:
        # the loader requires the array and we do not have it -- do not guess
        return 0, []
    elif rule.non_recurrent:
        recr = set(recr_layers)
        flags = [il not in recr for il in range(n_layer_all)]
    elif rule.all_layers:
        flags = [True] * n_layer_all
    else:
        period = int(pattern) if pattern is not None else rule.period
        flags = swa_pattern_flags(period, rule.dense_first, n_layer, n_layer_all)

    return n_swa, [il for il, is_swa in enumerate(flags) if is_swa]


def kv_cell_counts(info, n_ctx, n_seq=1, n_ubatch=512, unified=True, swa_full=False):
    """
    -> (cells per non-SWA layer, cells per SWA layer, n_stream), summed over
    streams. Mirrors llama-context.cpp:288-304 and llama-kv-cache-iswa.cpp:69-81.
    """
    n_ctx = int(math.ceil(n_ctx / 256.0) * 256)          # cparams.n_ctx
    n_seq = max(int(n_seq), 1)
    n_stream = 1 if unified else n_seq
    size_base = n_ctx if unified else int(math.ceil(n_ctx / n_seq / 256.0) * 256)
    if not info.swa_layers or swa_full:
        return size_base * n_stream, size_base * n_stream, n_stream
    window = info.n_swa * (n_seq if unified else 1) + n_ubatch
    size_swa = int(math.ceil(min(size_base, window) / 256.0) * 256)
    return size_base * n_stream, size_swa * n_stream, n_stream


def layer_kv_bytes(info, il, kv_type_id, cells):
    total = row_size(kv_type_id, info.n_embd_k_gqa[il] * cells)
    if info.n_embd_v_gqa[il]:
        total += row_size(kv_type_id, info.n_embd_v_gqa[il] * cells)
    return total


def kv_cache_bytes(info, n_ctx, kv_type_id, layers=None,
                   n_seq=1, n_ubatch=512, unified=True, swa_full=False):
    """
    Attention KV cache, sized the way llama_kv_cache_iswa does it: SWA layers
    get the small window cache, everything else gets the full n_ctx one.
    """
    base_cells, swa_cells, _ = kv_cell_counts(info, n_ctx, n_seq, n_ubatch,
                                              unified, swa_full)
    swa = set(info.swa_layers)
    total = 0
    for il in (info.main_kv_layers if layers is None else layers):
        total += layer_kv_bytes(info, il, kv_type_id,
                                swa_cells if il in swa else base_cells)
    return total


# --------------------------------------------------------------------------
# The MTP draft context
# --------------------------------------------------------------------------
#
# `--spec-type draft-mtp` does NOT just switch a graph on. common/speculative.cpp
# (common_speculative_init_result) calls llama_init_from_model() a SECOND time on
# the already-loaded target model with cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP.
# No model weights are re-read, but a whole second llama_context is built, and a
# context owns a KV cache and a compute buffer. llama-server prices this itself:
#
#   srv load_model: [spec] adding 2203.01 MiB to fit_params_target for device CUDA0
#   srv load_model: [spec] estimated memory usage of MTP context is 2203.01 MiB
#
# (server-context.cpp: `bytes = (has_draft ? dmd[j].model : 0) + dmd[j].context
# + dmd[j].compute` -- model bytes only for a real draft model, context+compute
# always.) Measured on LongCat-Flash-Lite Q4_K_M, -c 65535 -ub 512:
#
#   target context   KV 2088.00 MiB + compute 983.96 MiB (CUDA0), 150.07 (CUDA1)
#   MTP context      KV 2088.00 MiB + compute  115.01 MiB (CUDA0),   0.00 (CUDA1)
#
# That KV cache is a *full duplicate*: llama_model::create_memory installs an
# MTP layer filter only for a few archs, and longcat-flash-ngram is not one of
# them, so the draft context caches all 29 attention layers again. Omitting it
# is what made an otherwise-plausible plan die at load with
# "cudaMalloc failed: out of memory ... failed to allocate buffer for kv cache".
#
# Archs where llama-model.cpp DOES filter the MTP context's KV down to the
# nextn blocks (`filter = [](il) { return il >= hparams.n_layer(); }`):
MTP_KV_FILTERED_ARCHS = {"qwen35", "qwen35moe", "step35", "hy_v3"}

# Fallback for the MTP context's compute buffer. Its graph is one nextn block
# plus the shared head, so it is far smaller than the target's -- but there is
# no cheap formula that generalises (see estimate_compute_buffer), so this errs
# high against the 115.01 MiB measured on LongCat. Use --measure for the real value.
MTP_COMPUTE_ESTIMATE = 256 * MiB


def mtp_kv_cache_bytes(info, n_ctx, kv_type_id=None, n_seq=1, n_ubatch=512,
                       unified=True, swa_full=False):
    """
    KV cache of the second context llama.cpp builds for --spec-type draft-mtp.

    It comes out f16 whatever -ctk/-ctv say. Measured on Qwen3.8-27B run with
    -ctk q8_0 -ctv q8_0:
        llama_kv_cache: size = 256.00 MiB (65536 cells, 1 layers, ...), K (f16)
    which is 1 layer * 65536 * (1024+1024) * 2 B exactly. Sizing it at q8_0
    would under-count it by nearly half.
    """
    if info.arch in MTP_KV_FILTERED_ARCHS and info.n_nextn:
        # only the trailing nextn blocks are cached
        layers = info.kv_layers[-info.n_nextn:]
    else:
        # no filter -> the draft context re-caches every attention layer
        layers = info.kv_layers
    return kv_cache_bytes(info, n_ctx, KV_TYPE_F16[1], layers,
                          n_seq, n_ubatch, unified, swa_full)


def recurrent_state_bytes(info, n_seq, n_rs_seq=0):
    """
    SSM/KDA state -- one row per sequence, always f32, independent of n_ctx.

    n_rs_seq is the speculative rollback depth; the buffer holds (1 + n_rs_seq)
    copies. Measured on Qwen3.8-27B with 4 slots: 598.50 MiB at n_rs_seq=0 and
    2394.00 MiB at n_rs_seq=3, exactly 4x.
    """
    per_layer = row_size(0, info.n_embd_r) + row_size(0, info.n_embd_s)
    return per_layer * len(info.recr_layers) * n_seq * (1 + n_rs_seq)


def rs_rollback_depth(info, args, mtp_enabled):
    """cparams.n_rs_seq: the draft depth, but only on archs that can roll back."""
    if not mtp_enabled or info.arch not in RS_ROLLBACK_ARCHS or not info.recr_layers:
        return 0
    return max(args.draft_max, 0)


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
# Each llama_context finishes its reserve with a "graph nodes = N" line, so that
# is the section terminator for the compute-buffer sizes printed just above it.
GRAPH_NODES_RE = re.compile(r"sched_reserve:\s+graph nodes\s*=")
MTP_CTX_RE = re.compile(r"creating MTP draft context")
# Emitted once everything is allocated; nothing further is reserved after it.
SERVER_READY_RE = re.compile(r"listening on http|all slots are idle")


# The probe below lands within ~5% of the real compute buffer; this covers the
# rest. Measured on LongCat: probe 983.96 MiB vs 1031.96 MiB for the full plan.
MEASURE_SAFETY = 1.06


# llama-server lookup for --measure. The checkout sitting next to this script
# is the one these scripts are written against, so it goes first -- ahead of
# whatever stale llama-server happens to be on $PATH.
LLAMA_SERVER_SEARCH = (
    "../llama.cpp/build/bin/llama-server",
    "../llama.cpp/build/tools/server/llama-server",
    "../llama.cpp*/build/bin/llama-server",
    "~/Workplace/llama.cpp*/build/bin/llama-server",
)


def llama_server_candidates():
    """Search paths in priority order, resolved relative to this script's directory."""
    here = os.path.dirname(os.path.abspath(__file__))
    out = []
    for pat in LLAMA_SERVER_SEARCH:
        pat = os.path.expanduser(pat)
        if not os.path.isabs(pat):
            pat = os.path.normpath(os.path.join(here, pat))
        if pat not in out:      # ~/Workplace/... collapses onto ../... when the
            out.append(pat)     # script already lives under ~/Workplace
    return out


def find_llama_server():
    """First existing candidate, else whatever $PATH offers."""
    for pat in llama_server_candidates():
        for cand in sorted(glob.glob(pat)):
            if os.path.exists(cand):
                return cand
    return shutil.which("llama-server")


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

    Returns (target, mtp): two {device: bytes} dicts. `mtp` is empty unless the
    run created an MTP draft context.

    Parsing note: llama-server reserves graphs several times before the real
    load -- common_get_device_memory_data() builds throwaway no_alloc contexts
    to price the MTP context and to fit params. Those dry runs print the same
    "compute buffer size" lines, so we cannot just take the first ones. Instead
    we split the log into reserve sections (each ends with "graph nodes = N")
    and keep the LAST section before the "creating MTP draft context" marker
    (the real target context) and the last one after it (the real MTP context).
    """
    server = args.llama_server or find_llama_server()
    if not server or not os.path.exists(server):
        raise SystemExit(
            "error: --measure needs llama-server, and none was found. Looked in:\n"
            + "".join(f"  {p}\n" for p in llama_server_candidates())
            + "  $PATH\n"
            "Build it (cmake --build build -j --target llama-server) or pass "
            "--llama-server PATH.")

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
    section = {}          # devices seen since the last "graph nodes" line
    target, mtp = {}, {}  # last completed section on each side of the MTP marker
    seen_mtp = False
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    try:
        deadline = time.time() + args.measure_timeout
        for line in proc.stdout:
            m = COMPUTE_BUF_RE.search(line)
            if m:
                section[m.group(1)] = int(float(m.group(2)) * MiB)
            elif GRAPH_NODES_RE.search(line):
                if section:
                    if seen_mtp:
                        mtp = section
                    else:
                        target = section
                section = {}
            elif MTP_CTX_RE.search(line):
                seen_mtp = True
            if SERVER_READY_RE.search(line):
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

    if "CUDA0" not in target:
        raise SystemExit("error: could not read a CUDA0 compute buffer size from llama-server; "
                         "run it manually with -lv 5 and pass --compute-buf")
    if mtp_enabled and not mtp:
        raise SystemExit("error: --measure could not find the MTP draft context's compute "
                         "buffer; re-run with --no-mtp, or pass --compute-buf")

    def scale(d):
        return {k: int(v * MEASURE_SAFETY) for k, v in d.items()}

    raw_target, raw_mtp = dict(target), dict(mtp)
    target, mtp = scale(target), scale(mtp)
    sys.stderr.write(
        "measured: " + ", ".join(
            f"{k} {fmt_mib(raw_target[k])} -> {fmt_mib(target[k])} (x{MEASURE_SAFETY})"
            for k in sorted(raw_target)) + "\n")
    if raw_mtp:
        sys.stderr.write(
            "  MTP ctx: " + ", ".join(
                f"{k} {fmt_mib(raw_mtp[k])} -> {fmt_mib(mtp[k])} (x{MEASURE_SAFETY})"
                for k in sorted(raw_mtp)) + "\n")
    sys.stderr.write("\n")
    return target, mtp


# --------------------------------------------------------------------------
# GPU discovery
# --------------------------------------------------------------------------

@dataclass
class Gpu:
    index: int
    name: str
    total_bytes: int          # memory.total -- the card's nameplate size
    free_bytes: int           # memory.free  -- what a cudaMalloc can actually get
    budget_bytes: int         # the basis planning uses (free, or total with --use-total-vram)


def detect_gpus():
    """
    Read both memory.total and memory.free.

    Planning against memory.total overstates what is actually allocatable, and
    not only because a compositor or another process may hold VRAM: even on an
    idle card nvidia-smi reports free well below total (464 MiB below on this
    box's 5070 Ti, 384 MiB on the 4060) for driver-reserved memory that no
    cudaMalloc will ever return. memory.free is what llama.cpp can actually get,
    so it is the default basis; --use-total-vram reverts to the nameplate size.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True).stdout
    except Exception as e:
        raise SystemExit(f"error: could not run nvidia-smi ({e}); pass --vram to override")
    gpus = []
    for line in out.strip().splitlines():
        idx, name, total, free = [p.strip() for p in line.split(",")]
        total, free = int(total) * MiB, int(free) * MiB
        gpus.append(Gpu(int(idx), name, total, free, free))
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
    mtp_kv_bytes: int = 0            # second KV cache, --spec-type draft-mtp only
    mtp_compute_bytes: int = 0       # second compute buffer, ditto
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
# Dense models
# --------------------------------------------------------------------------
#
# `-ot` moves *weights* off the hot card. That only pays when a large, rarely
# touched slice of the weights exists -- i.e. per-layer expert sets. A dense
# model has none: every tensor is read on every token, so relocating any of it
# to CUDA1 or CPU costs throughput with nothing gained. The right tool is
# llama.cpp's own layer split (`-ngl 99 -dev CUDA0,CUDA1`, no `-ts`), which
# spreads weights AND the per-layer KV cache across both cards by free VRAM.
#
# So for a dense model this script stops being a tensor-override planner and
# becomes a context-size calculator: given the weights and both cards' VRAM,
# how much room is left for KV, and what -c does that buy at f16 vs q8_0?

# The compute buffer of a dense two-card split
# --------------------------------------------
#
# The MoE path's estimate_compute_buffer() has no n_ctx term at all -- it is a
# function of (n_expert, n_moe_layers, n_ubatch) with a 1 GiB floor. For a dense
# split that is the difference between a plan that loads and one that does not.
# Measured here, llama.cpp b10277, Qwen3.8-27B Q4_K_M, -ub 512 -fa on, KV q8_0,
# 4 server slots, both cards reporting the SAME size:
#
#   -c  32768 ( 32768 cells)   sched_reserve: CUDA0/CUDA1 = 377.13 MiB
#   -c 100000 (100096 cells)   sched_reserve: CUDA0/CUDA1 = 903.13 MiB
#
# Two things fall out of that pair:
#
#  1. Both cards allocate the same buffer. A split graph reserves its scratch on
#     every backend it touches, so CUDA1 does NOT get a fraction of CUDA0's the
#     way it does in the MoE layout (--compute-buf1-frac). Charging CUDA1 35%
#     of CUDA0 under-counts the small card by ~590 MiB at 100k, which is what
#     the "cudaMalloc failed ... 903.13 MiB on device 1" report came down to.
#
#  2. The context-dependent half is the KQ mask, not a fitted slope:
#         n_kv * n_ubatch * 4 bytes * n_seq_max
#         100096 * 512 * 4 * 4 = 782.00 MiB   vs   903.13 - 121.13 = 782.00
#          32768 * 512 * 4 * 4 = 256.00 MiB   vs   377.13 - 121.13 = 256.00
#     Both points land exactly, so only the ub-sized activation scratch below is
#     a constant, and it is the one thing --compute-buf overrides.
DENSE_COMPUTE_BASE = 121 * MiB

# The MTP draft context's own compute buffer, same treatment. Measured on
# Qwen3.8-27B, -ub 512, --spec-type draft-mtp, on the card holding the nextn
# block (CUDA1 here -- llama-server prints "[spec] adding 714.06 MiB to
# fit_params_target for device CUDA1" and 0.00 for CUDA0):
#
#   -c 32768   sched_reserve: CUDA1 = 330.06 MiB
#   -c 65536   sched_reserve: CUDA1 = 458.06 MiB
#
# The delta is again a mask, but a 2-byte one, matching this context's f16 KV:
#     n_kv * n_ubatch * 2 * n_seq
#     65536 * 512 * 2 * 4 = 256.00 MiB   vs   458.06 - 202.06 = 256.00
#     32768 * 512 * 2 * 4 = 128.00 MiB   vs   330.06 - 202.06 = 128.00
#
# The 202 MiB base is one nextn block plus the shared head, so it is far more
# arch-specific than the mask; MTP_COMPUTE_ESTIMATE stays the MoE fallback
# (LongCat measured 115.01 MiB, a different graph entirely).
MTP_COMPUTE_BASE = 202 * MiB


def mtp_compute_buffer(info, n_ctx, n_ubatch, n_seq):
    kv_size = int(math.ceil(n_ctx / 256.0) * 256)
    return MTP_COMPUTE_BASE + kv_size * n_ubatch * 2 * max(n_seq, 1)


def dense_compute_buffer(info, n_ctx, n_ubatch, n_seq, base=None):
    """Per-card compute buffer for a dense layer split. Same size on every card."""
    kv_size = int(math.ceil(n_ctx / 256.0) * 256)
    mask = kv_size * n_ubatch * 4 * max(n_seq, 1) if info.main_kv_layers else 0
    return (DENSE_COMPUTE_BASE if base is None else base) + mask


def dense_weight_map(tensors, info):
    """
    Per-block weight bytes, plus the non-repeating tensors, placed the way
    llama-model.cpp does it:

      dev_input  = cpu_dev, unconditionally ("there is very little benefit to
                   offloading the input layer, so always keep it on the CPU",
                   llama-model.cpp:1377). token_embd.weight therefore costs no
                   VRAM at all -- confirmed by the load log, which reports it as
                   `CPU_Mapped model buffer size = 682.03 MiB` on Qwen3.8-27B.
      dev_output = get_layer_buft_list(n_layer_all), i.e. the device that holds
                   the LAST block. output.weight rides along with it.

    Tied-embedding models have no output.weight; there token_embd is the output
    projection, so it is charged to the output device instead of the CPU.
    """
    layer_bytes = {}
    out_bytes = 0
    cpu_bytes = 0
    tied = not any(t.name == "output.weight" for t in tensors)
    for t in tensors:
        if t.layer is not None:
            if t.layer in info.dead_layers:
                continue
            layer_bytes[t.layer] = layer_bytes.get(t.layer, 0) + t.nbytes
        elif t.name == "token_embd.weight" and not tied:
            cpu_bytes += t.nbytes
        else:
            out_bytes += t.nbytes
    return layer_bytes, out_bytes, cpu_bytes


# --- the layer split itself -------------------------------------------------
#
# llama-model.cpp normalises -ts into cumulative split points and then assigns
#     layer_gpu(il) = upper_bound(splits, (il - i_gpu_start) / act_gpu_layers)
# with act_gpu_layers = min(n_gpu_layers, n_layer_all + 1). With -ngl 99 and 65
# blocks that denominator is 66, NOT 65, and the output layer is evaluated at
# il = n_layer_all. Two boundaries verified against the load log's
# "load_tensors: layer N assigned to device ..." lines on Qwen3.8-27B:
#
#   default (free-VRAM split, 15839:7804)  ->  blocks 0-44 on CUDA0
#   -ts 49,15                              ->  blocks 0-50 on CUDA0
#
# so a requested boundary has to be turned back into a ratio, not guessed.

def split_boundary(f0, n_layer_all, n_gpu_layers=99):
    """How many leading blocks land on CUDA0 for a normalised split point f0."""
    n_split = min(n_gpu_layers, n_layer_all + 1)
    return sum(1 for il in range(n_layer_all + 1) if il / n_split < f0)


def ts_for_boundary(boundary, n_layer_all, n_gpu_layers=99):
    """
    A -ts pair that reproduces `boundary` blocks on CUDA0. Aims at the midpoint
    of the interval that maps to it, then verifies the round trip -- a rounded
    ratio that lands on the wrong side of a block boundary would silently plan
    for a layout llama.cpp will not build.
    """
    n_split = min(n_gpu_layers, n_layer_all + 1)
    if boundary <= 0:
        return (0.0, 1.0)
    f0 = (boundary - 0.5) / n_split
    for digits in (4, 6, 9):
        ts0 = round(f0, digits)
        ts1 = round(1.0 - f0, digits)
        if ts0 > 0 and split_boundary(ts0 / (ts0 + ts1), n_layer_all, n_gpu_layers) == boundary:
            return (ts0, ts1)
    return (f0, 1.0 - f0)


def fmt_ts(ts):
    return "%s,%s" % tuple(("%.6f" % v).rstrip("0").rstrip(".") or "0" for v in ts)


@dataclass
class DenseFit:
    boundary: int
    ts: tuple
    used: tuple                 # (CUDA0, CUDA1) bytes
    budget: tuple
    weights: tuple
    kv: tuple
    recr: tuple
    compute: int                # per card, same on both
    mtp: tuple
    out_dev: int

    @property
    def slack(self):
        return min(b - u for b, u in zip(self.budget, self.used))

    def fits(self):
        return self.slack >= 0


def dense_fit(info, args, layer_bytes, out_bytes, budgets, n_ctx, kv_type,
              boundary, compute_base, mtp_enabled):
    """Price one candidate layer boundary across the two cards."""
    kv_size = int(math.ceil(n_ctx / 256.0) * 256)
    base_cells, swa_cells, _ = kv_cell_counts(info, n_ctx, args.parallel, args.ubatch,
                                              args.kv_unified, args.swa_full)
    swa_set = set(info.swa_layers)
    dev = lambda il: 0 if il < boundary else 1

    weights = [0, 0]
    for il, nb in layer_bytes.items():
        weights[dev(il)] += nb
    out_dev = dev(info.n_layer_all)
    weights[out_dev] += out_bytes

    kv = [0, 0]
    for il in info.main_kv_layers:
        kv[dev(il)] += layer_kv_bytes(info, il, kv_type,
                                      swa_cells if il in swa_set else base_cells)

    n_rs_seq = rs_rollback_depth(info, args, mtp_enabled)
    rs_per_layer = ((row_size(0, info.n_embd_r) + row_size(0, info.n_embd_s))
                    * args.parallel * (1 + n_rs_seq))
    recr = [0, 0]
    for il in info.recr_layers:
        recr[dev(il)] += rs_per_layer

    compute = dense_compute_buffer(info, n_ctx, args.ubatch, args.parallel, compute_base)

    mtp = [0, 0]
    if mtp_enabled:
        # The draft context's KV follows the blocks it caches, at f16, and its
        # compute buffer lands on the card that holds those blocks -- not CUDA0.
        # llama-server prints the split itself:
        #   [spec] adding   0.00 MiB to fit_params_target for device CUDA0
        #   [spec] adding 714.06 MiB to fit_params_target for device CUDA1
        cached = (info.kv_layers[-info.n_nextn:]
                  if info.arch in MTP_KV_FILTERED_ARCHS and info.n_nextn
                  else info.kv_layers)
        for il in cached:
            mtp[dev(il)] += layer_kv_bytes(info, il, KV_TYPE_F16[1],
                                           swa_cells if il in swa_set else base_cells)
        mtp[dev(cached[-1]) if cached else 0] += mtp_compute_buffer(
            info, n_ctx, args.ubatch, args.parallel)

    used = tuple(weights[i] + kv[i] + recr[i] + compute + mtp[i] for i in (0, 1))
    return DenseFit(boundary, ts_for_boundary(boundary, info.n_layer_all),
                    used, tuple(budgets), tuple(weights), tuple(kv), tuple(recr),
                    compute, tuple(mtp), out_dev)


def plan_dense_split(info, args, layer_bytes, out_bytes, budgets, n_ctx, kv_type,
                     compute_base, mtp_enabled):
    """
    Best layer boundary for this context, or None if no boundary fits.

    "Best" is the one that maximises the SMALLER of the two cards' leftovers.
    Pooling both cards into one budget -- which is what this script used to do --
    hides the only failure that matters: llama.cpp fills the small card to the
    brim while the big one still has gigabytes free, and the load dies on the
    small card's compute buffer.
    """
    best = None
    n_split = min(99, info.n_layer_all + 1)
    for boundary in range(0, n_split + 1):
        fit = dense_fit(info, args, layer_bytes, out_bytes, budgets, n_ctx, kv_type,
                        boundary, compute_base, mtp_enabled)
        if fit.fits() and (best is None or fit.slack > best.slack):
            best = fit
    return best


def max_ctx_dense(info, args, layer_bytes, out_bytes, budgets, kv_type, compute_base,
                  mtp_enabled, ctx_cap, granularity=1024):
    """
    Largest n_ctx some layer boundary can hold. Feasibility is monotone in n_ctx
    (KV and the mask both only grow), so bisection is safe; the step function in
    kv_cache padding is why this searches rather than divides.
    """
    def fits(n_ctx):
        return plan_dense_split(info, args, layer_bytes, out_bytes, budgets, n_ctx,
                                kv_type, compute_base, mtp_enabled) is not None

    if not fits(granularity):
        return 0, None
    lo, hi = granularity, max(ctx_cap, granularity)
    if fits(hi):
        return hi, plan_dense_split(info, args, layer_bytes, out_bytes, budgets, hi,
                                    kv_type, compute_base, mtp_enabled)
    while lo < hi:
        mid = (lo + hi + granularity) // 2
        mid -= mid % granularity
        if mid <= lo:
            break
        if fits(mid):
            lo = mid
        else:
            hi = mid - granularity
    return lo, plan_dense_split(info, args, layer_bytes, out_bytes, budgets, lo,
                                kv_type, compute_base, mtp_enabled)


def dense_report(args, md, info, gpus, tensors, weights_bytes, mtp_enabled):
    """Print the VRAM budget and the context sizes a two-card layer split affords."""
    e = sys.stderr.write

    layer_bytes, out_bytes, cpu_weight_bytes = dense_weight_map(tensors, info)
    gpu_weight_bytes = sum(layer_bytes.values()) + out_bytes
    dead_bytes = sum(t.nbytes for t in tensors if t.layer in info.dead_layers)
    compute_base = args.compute_buf * MiB if args.compute_buf is not None else None
    compute_src = "given" if compute_base is not None else "derived"
    budgets = [gpus[0].budget_bytes - args.reserve0 * MiB,
               gpus[1].budget_bytes - args.reserve1 * MiB]

    n_ctx_train = int(_mdget(md, info.arch, "context_length", 0) or 0)
    ctx_cap = n_ctx_train or 1 << 20
    n_live = info.n_layer_all - len(info.dead_layers)

    variants = []      # (flag, kv name, kv type, max n_ctx, DenseFit)
    for flag, (kv_name, kv_type) in (("", KV_TYPE_F16), ("-q8", KV_TYPE_Q8_0)):
        ctx, fit = max_ctx_dense(info, args, layer_bytes, out_bytes, budgets, kv_type,
                                 compute_base, mtp_enabled, ctx_cap)
        variants.append((flag, kv_name, kv_type, ctx, fit))

    if args.json:
        print(json.dumps({
            "model": os.path.basename(args.model),
            "architecture": info.arch,
            "dense": True,
            "n_layer_all": info.n_layer_all,
            "recurrent_layers": len(info.recr_layers),
            "attention_layers": len(info.attn_layers),
            "kv_layers": len(info.main_kv_layers),
            "is_mla": info.is_mla,
            "mtp_enabled": mtp_enabled,
            "mtp_capable": info.mtp_capable,
            "rs_rollback_depth": rs_rollback_depth(info, args, mtp_enabled),
            "n_ctx_train": n_ctx_train,
            "bytes": {
                "weights_total": weights_bytes,
                "weights_on_gpu": gpu_weight_bytes,
                "weights_on_cpu": cpu_weight_bytes,
                "dead_nextn": dead_bytes,
                "budget_cuda0": budgets[0],
                "budget_cuda1": budgets[1],
            },
            "max_ctx": {
                kv_name: None if not ctx else {
                    "n_ctx": ctx,
                    "tensor_split": list(fit.ts),
                    "blocks_on_cuda0": fit.boundary,
                    "cuda0": {"used": fit.used[0], "budget": fit.budget[0],
                              "weights": fit.weights[0], "kv": fit.kv[0],
                              "recurrent": fit.recr[0], "compute": fit.compute},
                    "cuda1": {"used": fit.used[1], "budget": fit.budget[1],
                              "weights": fit.weights[1], "kv": fit.kv[1],
                              "recurrent": fit.recr[1], "compute": fit.compute},
                }
                for _, kv_name, _, ctx, fit in variants},
            "ot": [],
        }, indent=2))
        return

    chosen = variants[1] if args.q8 else variants[0]

    if args.flags_only:
        print("# no tensor override applies -- dense model, use a plain layer split")
        return

    e(f"DENSE MODEL -- no tensor override applies\n\n")
    e(f"Model         : {os.path.basename(args.model)}\n")
    e(f"Architecture  : {info.arch}  ({info.name})\n")
    e(f"Blocks        : {n_live} live -> {len(info.recr_layers)} recurrent (SSM/KDA), "
      f"{len(info.attn_layers)} attention{' [MLA]' if info.is_mla else ''}"
      f"; {len(info.main_kv_layers)} carry a KV cache\n")
    e(f"Dense         : no *_exps.* tensors, so every weight is read on every token.\n"
      f"                -ot would only move hot weights to a slower card. Let\n"
      f"                llama.cpp split by layer across CUDA0+CUDA1 instead; this\n"
      f"                report sizes that split card by card.\n")
    if mtp_enabled:
        dup = "a full duplicate" if info.arch not in MTP_KV_FILTERED_ARCHS \
            else f"{info.n_nextn} nextn layer(s)"
        n_rs_seq = rs_rollback_depth(info, args, mtp_enabled)
        e(f"MTP           : ENABLED -- {info.n_nextn} nextn block(s) present, so the commands\n"
          f"                carry --spec-type draft-mtp. Priced into the -c figures below:\n"
          f"                the nextn block's weights, a second llama_context (KV over\n"
          f"                {dup}, at f16 whatever -ctk says, plus its own\n"
          f"                compute buffer)")
        if n_rs_seq:
            e(f",\n                and {1 + n_rs_seq}x the recurrent state, because n_rs_seq="
              f"{n_rs_seq} (--draft-max)\n"
              f"                keeps a rollback copy per drafted token")
        e(f".\n                Use --no-mtp to reclaim all of it.\n")
    elif info.mtp_capable:
        e(f"MTP           : available ({info.n_nextn} nextn block(s)) but disabled by --no-mtp;\n"
          f"                dropping it is already reflected in the budget below\n")
    elif info.n_nextn:
        e(f"MTP           : declared ({info.n_nextn} block(s)) but the nextn tensors are absent\n")
    if n_ctx_train:
        e(f"Trained ctx   : {n_ctx_train:,}  (the -c figures below are capped here)\n")
    e("\n")

    basis = "memory.total" if args.use_total_vram else "memory.free"
    e("Per-card budget (llama.cpp splits layers, so each card is a separate limit)\n")
    for i, g in enumerate(gpus[:2]):
        short = g.name.replace("NVIDIA GeForce ", "")
        reserve = (args.reserve0 if i == 0 else args.reserve1) * MiB
        e(f"  CUDA{g.index} ({short})\n")
        e(f"    {basis:<28}".ljust(38) + f"{fmt_mib(g.budget_bytes):>12}\n")
        if not args.use_total_vram and g.total_bytes > g.free_bytes:
            e(f"      ({fmt_mib(g.total_bytes)} total, "
              f"{fmt_mib(g.total_bytes - g.free_bytes)} already held)\n")
        e("    - reserved (CUDA ctx + headroom)".ljust(38) + f"{fmt_mib(reserve):>12}\n")
        e("    = budget".ljust(38) + f"{fmt_mib(budgets[i]):>12}\n")
    e("\n")
    e("Weights\n")
    e("  on GPU (blocks + output)".ljust(38) + f"{fmt_gib(gpu_weight_bytes):>12}\n")
    if cpu_weight_bytes:
        e("  token_embd, always CPU-resident".ljust(38) + f"{fmt_mib(cpu_weight_bytes):>12}\n")
    if dead_bytes:
        e(f"  {len(info.dead_layers)} nextn block(s), not loaded".ljust(38)
          + f"{fmt_mib(dead_bytes):>12}\n")
    e("\n")

    if not any(ctx for _, _, _, ctx, _ in variants):
        e(f"No context fits. {fmt_gib(gpu_weight_bytes)} of weights against "
          f"{fmt_mib(sum(budgets))} of budget\n"
          f"leaves no room -- use a smaller quant, or drop -ngl below {n_live}.\n")
        raise SystemExit(2)

    for flag, kv_name, kv_type, ctx, fit in variants:
        if not ctx:
            e(f"KV {kv_name}: does not fit\n\n")
            continue
        capped = "  (capped at trained ctx)" if n_ctx_train and ctx >= n_ctx_train else ""
        mark = "  <- selected" if (flag == "-q8") == bool(args.q8) else ""
        e(f"KV {kv_name}  ->  -c {ctx}   -ts {fmt_ts(fit.ts)}{capped}{mark}\n")
        e(f"  blocks 0-{fit.boundary - 1} on CUDA0, {fit.boundary}-{info.n_layer_all - 1} "
          f"on CUDA1; output.weight on CUDA{fit.out_dev}\n")
        e(f"  {'':22s}{'CUDA0':>12}{'CUDA1':>12}\n")
        rows = [("weights", fit.weights),
                (f"KV cache ({kv_name})", fit.kv),
                (f"recurrent state ({args.parallel} seq)", fit.recr),
                (f"compute buffer ({compute_src})", (fit.compute, fit.compute))]
        if mtp_enabled:
            rows.append(("MTP draft context", fit.mtp))
        for label, pair in rows:
            if not any(pair):
                continue
            e(f"  {label:<22}{fmt_mib(pair[0]):>12}{fmt_mib(pair[1]):>12}\n")
        e(f"  {'-' * 46}\n")
        e(f"  {'used':<22}{fmt_mib(fit.used[0]):>12}{fmt_mib(fit.used[1]):>12}\n")
        e(f"  {'of budget':<22}{fmt_mib(fit.budget[0]):>12}{fmt_mib(fit.budget[1]):>12}\n")
        e(f"  {'left over':<22}{fmt_mib(fit.budget[0] - fit.used[0]):>12}"
          f"{fmt_mib(fit.budget[1] - fit.used[1]):>12}\n")
        e("\n")
        for line in dense_command(args, ctx, flag == "-q8", mtp_enabled, fit).split("\n"):
            e(f"  {line}\n")
        e("\n")

    # What MTP is actually costing, in tokens of context, for the selected KV type.
    if mtp_enabled and chosen[3]:
        info_off = analyse(md, tensors, use_mtp=False)
        lb_off, ob_off, _ = dense_weight_map(tensors, info_off)
        ctx_off, _ = max_ctx_dense(info_off, args, lb_off, ob_off, budgets, chosen[2],
                                   compute_base, False, ctx_cap)
        if ctx_off > chosen[3]:
            e(f"MTP is costing {ctx_off - chosen[3]:,} tokens of context: --no-mtp would take\n"
              f"KV {chosen[1]} from -c {chosen[3]} to -c {ctx_off}. Whether the drafting pays for\n"
              f"that is a throughput question this script cannot answer -- benchmark both.\n\n")

    if compute_src == "derived":
        e("The compute-buffer figure is n_kv*n_ubatch*4*n_seq of KQ mask plus "
          f"{fmt_mib(DENSE_COMPUTE_BASE)} of\n"
          "activation scratch, matched to two measured points on this box. If a load\n"
          "still OOMs, read the real 'compute buffer size' out of llama-server -lv 5\n"
          "and pass its ub-only part via --compute-buf.\n")
    e("Without -ts, llama.cpp splits by free VRAM at load time, which overfills the\n"
      "smaller card: the same weights at -c 100000 leave the 4060 with 197 MiB free\n"
      "and the 5070 Ti with 2659 MiB. The -ts above is what balances them, so keep it.\n\n")

    # stdout carries the single runnable command for the KV type actually
    # selected on the command line, so `... 2>/dev/null | sh` still works.
    if chosen[3]:
        print(dense_command(args, chosen[3], args.q8, mtp_enabled, chosen[4]))
    else:
        fallback = variants[1]
        print(dense_command(args, fallback[3], True, mtp_enabled, fallback[4]))


def dense_command(args, n_ctx, q8, mtp_enabled, fit=None):
    """The llama-server invocation for a dense two-card layer split."""
    parts = [f"{args.server_bin} -m {args.model}", "-ngl 99", "-dev CUDA0,CUDA1"]
    if fit is not None:
        parts.append(f"-ts {fmt_ts(fit.ts)}")
    parts += [f"-c {n_ctx}", f"-ub {args.ubatch}"]
    if q8:
        parts += ["-ctk q8_0", "-ctv q8_0"]
    if mtp_enabled:
        parts += ["--spec-type draft-mtp"]
    parts += ["-fa on"]
    return " \\\n  ".join(parts)


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
    ap.add_argument("--swa-full", action="store_true",
                    help="price the SWA KV cache at full n_ctx, i.e. plan for "
                         "llama.cpp's --swa-full. Without it, SWA layers are sized at "
                         "GGML_PAD(min(n_ctx_seq, n_swa*n_seq_max + n_ubatch), 256) "
                         "cells, which is what llama_kv_cache_iswa actually allocates.")
    ap.add_argument("--no-kv-unified", dest="kv_unified", action="store_false", default=True,
                    help="plan for a per-slot (non-unified) KV cache. llama-server's "
                         "default -np -1 means 4 slots with kv_unified = true "
                         "(server.cpp:152-158), which is the default here; pass this only "
                         "if you also pass an explicit -np N to llama-server.")
    ap.add_argument("--parallel", type=int, default=4,
                    help="number of server slots; sizes the recurrent state. llama-server's "
                         "default is auto -> 4 slots with kv_unified, so 4 is the default here too.")
    ap.add_argument("--no-mtp", dest="mtp", action="store_false", default=True,
                    help="do NOT use the model's MTP/nextn blocks. By default, if the GGUF ships "
                         "nextn tensors the plan budgets for them and the command carries "
                         "--spec-type draft-mtp.")
    ap.add_argument("--draft-max", type=int, default=3,
                    help="llama.cpp's --draft-max, i.e. how many tokens MTP drafts per step "
                         "(default 3, matching common_params_speculative_draft::n_max). On a "
                         "recurrent arch this is also cparams.n_rs_seq, so the recurrent state "
                         "is sized for 1+N copies -- +1795 MiB on Qwen3.8-27B at the default.")
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
                         "measured 328.06 vs 1022.02 MiB on Ling at -ub 2048). MoE layouts "
                         "only -- in a dense layer split both cards allocate the same size, "
                         "so this is ignored there.")
    ap.add_argument("--measure", action="store_true",
                    help="load the model once with -cmoe and read the REAL compute buffer sizes "
                         "from llama.cpp instead of estimating. Strongly recommended: the "
                         "estimate does not generalise across architectures.")
    ap.add_argument("--llama-server", default=None,
                    help="path to llama-server for --measure. Default: the checkout next to "
                         "this script (../llama.cpp/build/bin), then ../llama.cpp*, then "
                         "~/Workplace/llama.cpp*, then $PATH.")
    ap.add_argument("--measure-port", type=int, default=18099, help="port used by --measure")
    ap.add_argument("--measure-timeout", type=int, default=600,
                    help="seconds to wait for --measure (default 600)")
    ap.add_argument("--compute-buf", type=int, default=None,
                    help="MiB for the CUDA0 compute buffer; overrides estimate and --measure. "
                         "Read the real value from llama-server -lv 5 'compute buffer size'. "
                         "For a dense model this sets only the ub-sized activation scratch; "
                         "the KQ-mask term still scales with -c on top of it.")
    ap.add_argument("--vram", default=None,
                    help="override detected VRAM, MiB, comma-separated per device, e.g. 15839,7804")
    ap.add_argument("--use-total-vram", action="store_true",
                    help="plan against nvidia-smi memory.total instead of memory.free "
                         "(default). Use only when the cards will be idle at load time; "
                         "free is lower than total even on an idle card because of "
                         "driver-reserved memory, and planning past it OOMs.")
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
    if args.use_total_vram:
        for g in gpus:
            g.budget_bytes = g.total_bytes
    if args.vram:
        overrides = [int(x) * MiB for x in args.vram.split(",")]
        for g, v in zip(gpus, overrides):
            g.total_bytes = g.free_bytes = g.budget_bytes = v
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

    # A dense model has no `*_exps.*` tensors, so every weight is a core tensor
    # that rule 1 pins to CUDA0. There is nothing this script is allowed to move
    # to CUDA1, and pack() would bail out at `used0 > gpu0_budget` and print a
    # "cannot fit on CUDA0" report that reads as a context/KV problem when the
    # real answer is "wrong tool: use a plain layer split". Report on that split
    # instead -- it is the only sensible way to run a dense model on two cards.
    if not expert_layers:
        dense_report(args, md, info, gpus, tensors, weights_bytes, mtp_enabled)
        return

    # A model with tied embeddings reuses token_embd as the output projection;
    # evicting it would put a full vocab matmul on the CPU every token. Resolve
    # this before measuring, so the probe uses the same placement as the plan.
    if not any(t.name == "output.weight" for t in tensors):
        core_bytes += lookup_bytes.pop("token_embd.weight", 0)
    FAMILY_SIZES["ngram_embd"] = {n for n in lookup_bytes if n.startswith("ngram_embd.")}

    kv_name, kv_type = KV_TYPE_Q8_0 if args.q8 else KV_TYPE_F16
    kv_bytes = kv_cache_bytes(info, args.ctx, kv_type, None, args.parallel,
                              args.ubatch, args.kv_unified, args.swa_full)
    recr_bytes = recurrent_state_bytes(info, args.parallel,
                                       rs_rollback_depth(info, args, mtp_enabled))
    # --spec-type draft-mtp builds a SECOND llama_context on the same model, with
    # its own KV cache (usually a full duplicate) and its own compute buffer.
    mtp_kv_bytes = (mtp_kv_cache_bytes(info, args.ctx, None, args.parallel,
                                       args.ubatch, args.kv_unified, args.swa_full)
                    if mtp_enabled else 0)
    measured, measured_mtp = {}, {}
    if args.compute_buf is not None:
        compute_bytes = args.compute_buf * MiB
        compute_src = "given"
    elif args.measure:
        measured, measured_mtp = measure_compute_buffers(
            args, args.model, expert_layers, expert_suffixes, list(lookup_bytes),
            mtp_enabled=mtp_enabled)
        compute_bytes = measured["CUDA0"]
        compute_src = "measured"
    else:
        compute_bytes = estimate_compute_buffer(info, args.ubatch, len(expert_layers))
        compute_src = "estimated"

    if not mtp_enabled:
        mtp_compute_bytes = 0
    elif measured_mtp:
        mtp_compute_bytes = measured_mtp.get("CUDA0", 0)
    else:
        mtp_compute_bytes = MTP_COMPUTE_ESTIMATE

    gpu0_budget = gpus[0].budget_bytes - args.reserve0 * MiB
    gpu1_budget = gpus[1].budget_bytes - args.reserve1 * MiB
    fixed0 = kv_bytes + recr_bytes + compute_bytes + mtp_kv_bytes + mtp_compute_bytes
    # CUDA1 holds only expert weights, but it still takes part in the graph and
    # so allocates its own compute buffer. Measured 328.06 MiB (Ling) and
    # 150.07 MiB (LongCat), so a fraction of CUDA0's with a floor.
    fixed1 = max(int(compute_bytes * args.compute_buf1_frac), 256 * MiB)
    if "CUDA1" in measured:
        fixed1 = max(measured["CUDA1"], 256 * MiB)
    # The MTP graph only touches the nextn block, whose weights are core tensors
    # pinned to CUDA0, so llama.cpp reserves nothing on CUDA1 for it (measured
    # 0.00 MiB on LongCat). Trust a measurement if we have one, add nothing if not.
    fixed1 += measured_mtp.get("CUDA1", 0)

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
        if mtp_enabled:
            e(f"  MTP draft context (KV + compute)".ljust(38)
              + f"{fmt_mib(mtp_kv_bytes + mtp_compute_bytes):>12}\n")
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
        if mtp_enabled:
            e(f"  --no-mtp            drops the second context "
              f"({fmt_mib(mtp_kv_bytes + mtp_compute_bytes)})\n")
        e(f"  --reserve0 512      reclaims allocator headroom (may OOM at graph reserve)\n")
        raise SystemExit(2)

    g0, g1, cpu_layers, lk_gpu1, lk_cpu, used0, used1 = packed
    cpu_bytes = (sum(expert_bytes[i] for i in cpu_layers)
                 + sum(lookup_bytes[n] for n in lk_cpu))
    plan = Plan(g0, g1, cpu_layers, lk_gpu1, lk_cpu, used0, used1, cpu_bytes,
                gpu0_budget, gpu1_budget, core_bytes, kv_bytes, recr_bytes, compute_bytes,
                mtp_kv_bytes, mtp_compute_bytes)

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
            "mtp_enabled": mtp_enabled,
            "bytes": {
                "weights_total": weights_bytes,
                "core": plan.core_bytes,
                "experts_total": total_expert_bytes,
                "kv_cache": kv_bytes,
                "recurrent_state": recr_bytes,
                "compute_buffer": compute_bytes,
                "mtp_kv_cache": mtp_kv_bytes,
                "mtp_compute_buffer": mtp_compute_bytes,
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
        dup = "a full duplicate" if info.arch not in MTP_KV_FILTERED_ARCHS \
            else f"{info.n_nextn} nextn layer(s)"
        e(f"MTP           : ENABLED -- {info.n_nextn} nextn block(s) present, so the command\n"
          f"                carries --spec-type draft-mtp. That builds a SECOND llama_context\n"
          f"                on the same weights: +{fmt_mib(mtp_kv_bytes)} KV ({dup}) and\n"
          f"                +{fmt_mib(mtp_compute_bytes)} compute on CUDA0. Use --no-mtp to reclaim it.\n")
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
    if info.swa_layers:
        base_cells, swa_cells, _ = kv_cell_counts(info, args.ctx, args.parallel,
                                                  args.ubatch, args.kv_unified,
                                                  args.swa_full)
        e(f"      {len(info.main_kv_layers) - len(info.swa_layers)} full layers "
          f"@ {base_cells} cells, {len(info.swa_layers)} SWA "
          f"@ {swa_cells} (n_swa {info.n_swa})\n")
    e(f"  recurrent state ({args.parallel} seq)".ljust(33) + f"{fmt_mib(recr_bytes):>12}\n")
    e(f"  compute buffer  ({compute_src})".ljust(33) + f"{fmt_mib(compute_bytes):>12}\n")
    if compute_src == "estimated":
        e("      ^ a coarse upper bound; re-run with --measure for the real value\n")
    if mtp_enabled:
        e(f"  MTP draft context".ljust(33)
          + f"{fmt_mib(mtp_kv_bytes + mtp_compute_bytes):>12}\n")
        e(f"    second KV cache".ljust(33) + f"{fmt_mib(mtp_kv_bytes):>12}\n")
        e(f"    second compute buffer".ljust(33) + f"{fmt_mib(mtp_compute_bytes):>12}"
          + ("   (measured)\n" if measured_mtp else "   (estimated)\n"))
    e("\n")

    for i, (gpu, budget, used) in enumerate((
            (gpus[0], gpu0_budget, plan.gpu0_used),
            (gpus[1], gpu1_budget, plan.gpu1_used))):
        reserve = args.reserve0 if i == 0 else args.reserve1
        e(f"  CUDA{i} ({gpu.name})\n")
        basis = "memory.total" if args.use_total_vram else "memory.free"
        e(f"    {basis:<24}     {fmt_mib(gpu.budget_bytes):>12}\n")
        if not args.use_total_vram and gpu.total_bytes > gpu.free_bytes:
            e(f"      ({fmt_mib(gpu.total_bytes)} total, "
              f"{fmt_mib(gpu.total_bytes - gpu.free_bytes)} already held)\n")
        e(f"    - reserved (CUDA ctx)        {fmt_mib(reserve * MiB):>12}\n")
        e(f"    = budget                     {fmt_mib(budget):>12}\n")
        lk_here = ([n for n in lookup_bytes
                    if n not in plan.lookup_gpu1 and n not in plan.lookup_cpu]
                   if i == 0 else plan.lookup_gpu1)
        if i == 0:
            label = "KV + recurrent + compute" + (" + MTP" if mtp_enabled else "")
            e(f"      {label}".ljust(33) + f"{fmt_mib(fixed0):>12}\n")
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
