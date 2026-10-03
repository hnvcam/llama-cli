# Adaptive Heterogeneous MoE Execution for llama.cpp

## 1. Project Objective

Design and implement a **model-generic, hardware-adaptive MoE execution system for llama.cpp**.

The system should accelerate Mixture-of-Experts models that are too large to fit completely in GPU VRAM by intelligently distributing expert computation across:

- primary GPU,
- one or more secondary GPUs,
- CPU + system RAM,

while allowing those devices to execute different routed experts **concurrently**.

The design should borrow the important concepts demonstrated by specialized runtimes such as Strata, but must not depend on one specific model architecture such as Qwen3.8-Flash-Next.

The runtime must discover:

1. the MoE structure of the loaded model,
2. which experts are actually being selected,
3. the performance characteristics of the available hardware,
4. the available VRAM and RAM,
5. the cost of executing or transferring each expert,

and dynamically determine the optimal expert placement and execution strategy.

The objective is therefore not simply:

> cache hot experts on GPU.

The objective is:

> build an adaptive heterogeneous MoE scheduler capable of deciding where each selected expert should execute, and execute independent expert work in parallel across all available compute devices.

---

# 2. Core Design Principle

For every MoE layer, the model router already determines which experts are required.

The runtime must treat this router decision as a **dynamic scheduling event**.

Conceptually:

Router produces selected experts

→ determine current residency of those experts

→ partition the work by execution device

→ execute independent experts concurrently

→ collect results

→ apply routing weights / aggregation

→ continue the transformer layer.

The desired execution shape is:

GPU0 expert work  
GPU1 expert work  
GPU2 expert work  
CPU expert work

all occurring concurrently whenever possible.

There is a synchronization point only when the MoE outputs must be combined.

Therefore, ideal layer latency becomes approximately:

**maximum(device execution times) + transfer/merge overhead**

instead of:

**GPU0 time + GPU1 time + CPU time**

The design must attempt to minimize the slowest branch.

---

# 3. Important Constraint: Do Not Change Model Routing

This optimization must initially be **lossless with respect to the model's MoE routing**.

Do not:

- remove experts,
- alter top-k,
- modify router probabilities,
- approximate expert selection,
- skip low-weight experts,
- alter shared-expert behavior.

The model chooses exactly the same experts as normal llama.cpp.

Only their **placement and execution location** changes.

Small numerical differences caused by CPU versus CUDA arithmetic are acceptable under the same rules currently applicable to llama.cpp backend changes, but model semantics must remain unchanged.

Lossy routing optimization can be considered separately and must not be part of this project.

---

# 4. Why Tensor Override Is Not Sufficient

Existing llama.cpp mechanisms such as tensor override, CPU-MoE placement, layer splitting, row splitting, and tensor splitting primarily control where weights are stored and where graph operations execute.

Current documented multi-GPU modes include pipelined layer splitting and parallel row/tensor splitting. Tensor overrides can assign specific tensor patterns to specific buffer types.

This project requires an additional concept:

**one logical MoE operation must be capable of executing on several backends simultaneously.**

For example, if one token selects eight experts:

- four may already be resident on GPU0,
- two may be resident on GPU1,
- two may remain in RAM.

The runtime should not move the entire MoE operation to one backend.

It should partition that operation:

- GPU0 calculates its four experts,
- GPU1 calculates its two experts,
- CPU calculates its two experts,

concurrently.

The results are then merged.

This means expert execution cannot be solved purely through static tensor placement.

---

# 5. Normalized MoE Architecture Description

The first requirement is to separate **MoE scheduling** from individual model architecture implementations.

llama.cpp currently contains architecture-specific graph builders, while MoE execution eventually converges around expert matrix operations such as `MUL_MAT_ID`. Current OpenAI-MoE support, for example, creates gate/up/down expert tensors explicitly, while normal llama-derived architectures use shared MoE graph-building paths.

Introduce or identify an internal normalized description of every MoE layer.

For each MoE layer, the runtime should be able to determine at minimum:

- number of routed experts,
- number of experts selected per token,
- whether shared experts exist,
- expert hidden/intermediate dimensions,
- expert tensor layout,
- gate/up/down tensor grouping,
- expert quantization type,
- expert biases/scales,
- router type,
- routing normalization behavior,
- whether routing count is static or dynamic,
- whether experts are structurally identical,
- approximate bytes per expert,
- backend compatibility for that expert representation.

Define an **expert instance** logically as:

(layer ID, expert ID)

not merely expert ID.

Expert 17 in layer 5 and expert 17 in layer 20 are different expert instances.

The scheduler must work using this normalized representation rather than model-specific tensor-name regexes wherever possible.

Architecture-specific code should provide metadata/adapters only where the architecture genuinely differs.

---

# 6. Expert Bundle Concept

Treat all tensors required to calculate one expert as one logical **expert bundle**.

For a conventional gated FFN this might include:

- gate projection,
- up projection,
- down projection,
- related scales,
- related biases.

The residency manager should reason about the bundle as a unit.

This avoids situations where half of an expert is resident on one backend and another part requires an unexpected transfer.

For each bundle maintain:

- size,
- source/backing location,
- current resident locations,
- execution capability per backend,
- measured execution cost per backend.

---

# 7. Expert Routing Telemetry

Add routing telemetry at the point where selected expert IDs become known.

For every `(layer, expert)` record statistics such as:

- selection count,
- recent selection count,
- exponentially weighted selection frequency,
- last-used position,
- reuse distance,
- routing probability/weight statistics where useful,
- cache hit count,
- cache miss count,
- execution device,
- estimated CPU cost avoided when GPU resident.

Do not define "hotness" as frequency alone.

A better concept is:

**expected latency saved per byte of VRAM consumed.**

An expert may be frequently used but very large.

Another expert may be slightly less frequent but much cheaper to keep resident.

The cache policy should eventually account for:

- probability of use,
- expert size,
- CPU execution cost,
- GPU execution cost,
- transfer cost,
- device contention,
- available VRAM.

---

# 8. Separate Prefill and Decode Policies

Prefill and token decoding should not automatically use the same cache strategy.

Recent llama.cpp expert-cache experiments observed that decode routing may be highly skewed while prompt/prefill routing can be much flatter, causing cache churn.

Therefore maintain separate behavior for:

### Decode

Primary target for adaptive hot-expert residency.

Routing history from the current conversation should influence expert placement.

### Prefill

Favor large-batch throughput and avoid unnecessary expert-cache churn.

Possible policies to research:

- freeze the decode cache during large prefill,
- maintain separate prefill statistics,
- use conventional CPU/GPU execution for very wide prefill routing,
- update hotness at a reduced rate,
- use temporary batch-specific placement.

The implementation should benchmark rather than assume one strategy works for both phases.

---

# 9. Persistent Expert Residency Manager

Create a persistent residency layer above the backing model weights.

System RAM should normally remain the authoritative backing store when the full expert set cannot fit in VRAM.

GPU caches contain additional resident copies of selected expert bundles.

For each expert, track states equivalent to:

- RAM resident,
- GPU0 resident,
- GPU1 resident,
- GPU-N resident,
- transfer pending.

The system must support different cache capacities on different GPUs.

Do not assume GPUs are identical.

Do not simply divide expert count according to VRAM capacity.

Placement should be performance-aware.

---

# 10. Primary GPU Role

When possible, designate one GPU as the **primary execution device**.

The primary GPU should preferably contain persistent non-expert computation such as:

- attention,
- router,
- normalization,
- shared dense weights,
- shared experts when beneficial,
- output layers,
- KV/state relevant to the architecture,
- other frequently executed non-MoE tensors.

The remaining VRAM can contain routed expert bundles.

This avoids repeatedly transferring the transformer activation between CPU and GPU for unavoidable dense portions of every layer.

The fastest/largest GPU will often become primary, but this must be chosen through hardware evaluation rather than hard-coded assumptions.

For the reference machine:

- RTX 5070 Ti 16 GB,
- RTX 4060 8 GB,
- i5-13500,
- 64 GB RAM,

the likely initial topology worth benchmarking is:

5070 Ti  
→ primary dense execution + router + KV + hottest experts

4060  
→ secondary expert cache / expert compute

CPU + RAM  
→ complete expert backing store + cold expert execution

But the system must not assume this arrangement is universally optimal.

---

# 11. Secondary GPU Role

Secondary GPUs should be capable of acting primarily as **expert accelerators** rather than requiring full transformer layers.

This is especially important for heterogeneous systems.

A secondary GPU may have:

- less VRAM,
- slower compute,
- slower PCIe connectivity,
- no direct P2P path to the primary GPU.

The scheduler should therefore evaluate whether using that GPU actually reduces layer latency.

A secondary GPU should be able to maintain its own persistent expert cache.

Default policy should favor **exclusive expert ownership** between GPU caches to maximize aggregate coverage.

Replication of the same hot expert across multiple GPUs may be allowed when measurements indicate that replication reduces contention or communication cost.

Strata has experimented with this type of ownership model: each expert can belong to CUDA0, a secondary GPU, or the CPU pool, with GPU and CPU work proceeding concurrently before a layer-level join.

---

# 12. Heterogeneous Expert Executor

The critical new component is a heterogeneous executor for expert operations.

Once routing IDs are known, classify selected expert work into groups:

- primary-GPU resident hits,
- secondary-GPU resident hits,
- CPU-resident misses,
- optionally experts worth transferring immediately.

Launch the independent groups asynchronously.

Do not serialize device execution unless a dependency requires it.

The executor must support:

- multiple CUDA streams/devices,
- CPU thread pool execution,
- asynchronous host/device copies,
- device/device copies where beneficial,
- pinned host staging buffers,
- asynchronous result return,
- one final synchronization/aggregation barrier.

The CPU should not sit idle merely because some experts are cached on GPU.

Likewise GPUs should not wait while cold experts execute on CPU when their own routed experts are already available.

---

# 13. CPU Miss Versus GPU Transfer Decision

A cache miss should not automatically mean:

copy expert to GPU and execute it there.

For one token, transferring an entire expert across PCIe may take longer than calculating that expert directly using CPU + RAM bandwidth.

For every miss the scheduler should conceptually compare:

### Option A

execute expert directly on CPU from RAM.

### Option B

transfer expert to GPU, then execute.

### Option C

transfer to another GPU.

The decision should consider:

- expert byte size,
- quant type,
- measured RAM bandwidth,
- measured PCIe bandwidth,
- pinned/pageable memory,
- CPU kernel throughput,
- GPU kernel throughput,
- expected near-future reuse,
- current backend queue load.

If the expert is unlikely to be reused soon, CPU execution may be preferable.

If reuse probability is high, paying the transfer once and promoting it into GPU residency may be worthwhile.

---

# 14. Hardware Auto-Profiler

Do not write policies specifically for 12 GB, 16 GB, 24 GB, etc.

At initialization, obtain or benchmark relevant hardware properties.

For each GPU:

- available VRAM,
- effective VRAM bandwidth where practical,
- representative expert MMV/GEMM throughput by quant family,
- host-to-device bandwidth,
- device-to-host bandwidth,
- GPU-to-GPU bandwidth,
- P2P availability,
- launch overhead.

For CPU/system memory:

- memory bandwidth,
- thread scaling,
- relevant quantized `MUL_MAT_ID` performance,
- NUMA topology where applicable,
- pinned-memory behavior.

The scheduler should use actual measured capability rather than device model names.

The same runtime should therefore adapt differently to:

- 24 GB + weak CPU,
- 12 GB + strong CPU,
- 16 GB + 8 GB dual GPU,
- 4 × 3090,
- workstation GPUs,
- high-bandwidth server CPUs.

---

# 15. Cost Model

Introduce an internal cost model used for scheduling.

It should estimate at minimum:

- CPU expert execution latency,
- GPU expert execution latency,
- H2D expert-transfer latency,
- activation-transfer latency,
- output-return latency,
- expected synchronization delay,
- device queue load.

The important optimization target is not maximum GPU utilization.

It is minimum **MoE layer completion latency**.

For example, assigning too many experts to GPU0 could make GPU0 the bottleneck while GPU1 and CPU finish early.

A better partition may move some expert work to CPU even if GPU execution is individually faster.

Therefore work assignment should attempt to balance:

GPU0 completion time  
GPU1 completion time  
CPU completion time

so they finish approximately together.

---

# 16. Dynamic Hotness / Cache Policy

Support both:

### Initial placement

Based on:

- optional saved routing profile,
- model-independent defaults,
- available VRAM,
- possibly a short warm-up.

### Online adaptation

As inference continues, continuously update routing statistics.

Expert residency should gradually adapt to:

- coding workloads,
- conversation,
- language,
- mathematical tasks,
- individual conversation context,
- long-running server workload.

Do not rely exclusively on a static profile generated by one benchmark dataset.

The cache policy should also avoid excessive churn.

Candidate policy research should include:

- LFU,
- LRU,
- segmented LRU,
- frequency + recency combinations,
- weighted frequency per byte,
- latency-saved-per-byte scoring,
- admission thresholds,
- hysteresis,
- conversation-local plus global statistics.

The exact algorithm should be benchmark-driven.

---

# 17. Conversation and Multi-User Behavior

llama-server introduces another problem: several sequences may use the same model simultaneously.

Routing statistics therefore need clearly defined scope.

Potential structure:

- long-term model-global prior,
- short-term context/session statistics,
- shared physical expert cache.

A single unusual request should not immediately evict experts valuable to every other active request.

The agent should specifically research cache pollution and fairness under continuous batching.

---

# 18. llama.cpp Areas to Investigate

The implementation research should focus on several existing subsystems.

## A. MoE graph construction

Investigate shared MoE graph construction and architecture-specific graph builders under `src/models`.

Identify the common path through which expert matrix operations reach `MUL_MAT_ID`.

Recent llama.cpp expert-cache work describes `build_lora_mm_id` as an important common funnel for MoE expert matrix multiplication.

The goal is to attach generic scheduling metadata here rather than implement unrelated solutions in every model architecture.

## B. `MUL_MAT_ID`

This should be treated as one of the main abstraction points.

`MUL_MAT_ID` already represents expert-indexed matrix multiplication: the expert ID tensor determines which expert matrix participates in each operation.

Investigate whether heterogeneous expert execution is best implemented primarily inside/around this operation.

## C. ggml backend scheduler

Study:

- `ggml-backend.h`,
- `ggml-backend.cpp`,
- backend graph splitting,
- tensor backend assignments,
- async graph execution,
- cross-backend tensor copies.

The scheduler already supports multiple backend devices, tensor assignment and backend-to-backend copying.

Determine what must change to support a logical MoE operation with multiple simultaneously active backends.

## D. CPU MoE implementation

Investigate CPU `MUL_MAT_ID` and quant-specific expert kernels.

CPU execution remains fundamental because cold expert misses should often execute directly from RAM.

Do not make GPU availability a requirement for correctness.

## E. CUDA MoE implementation

Investigate CUDA `MUL_MAT_ID`, relevant quantized MMV/GEMM paths, stream usage and grouped expert execution.

CUDA expert execution should support efficient operation from compact cached expert slots rather than assuming the original full tensor layout.

## F. Memory allocation / tensor loading

Investigate how expert tensors can retain a full CPU backing representation while also exposing persistent compact copies on accelerator devices.

Avoid requiring a full GPU-shaped allocation when only a small expert subset is cached.

## G. Auto-fit / placement system

Adaptive MoE placement should eventually cooperate with llama.cpp auto-fit instead of fighting it.

VRAM must first account for mandatory working memory, context/KV needs and dense tensors before assigning the remainder to expert caches.

---

# 19. Operator-Level Versus General Scheduler Design

Research two implementation strategies.

## Strategy 1 — MoE-Specific Hybrid Operator

Treat heterogeneous execution as a specialized implementation of expert operations such as `MUL_MAT_ID`.

Internally the operation partitions selected experts across devices.

Advantages:

- smaller modification surface,
- naturally aligned with MoE semantics,
- easier to preserve ordinary graph execution,
- easier first implementation.

Existing llama.cpp expert-cache experiments use a similar philosophy: CPU remains capable of processing misses while cached hits are dispatched to accelerators.

This should probably be the first implementation path.

## Strategy 2 — General Fork/Join Backend Scheduler

Extend ggml graph scheduling so one logical region can explicitly fork into independent backend subgraphs and later join.

Advantages:

- more general,
- potentially useful beyond MoE.

Disadvantages:

- much larger scheduler change,
- greater regression risk,
- more complex memory lifetime and synchronization.

Do not begin with the general scheduler rewrite unless research demonstrates the operator-level approach fundamentally cannot support multi-device expert parallelism.

---

# 20. Compact Expert GPU Slots

Do not require GPU expert cache tensors to preserve the full original expert dimension.

A GPU containing 500 of 10,000 expert instances should ideally allocate space only for those 500.

Maintain mapping between:

global `(layer, expert)` identity

and

device-local expert slot.

On a routing hit, translate global expert identity to the local slot.

On admission:

- choose/allocate slot,
- asynchronously transfer the complete expert bundle,
- update mapping only when transfer is safe/complete.

Persistent compact slot pools have already been explored in llama.cpp expert-cache proposals and are an important direction to investigate.

---

# 21. Transfers and Staging

Host expert storage intended for frequent GPU promotion should support pinned memory when it produces measurable benefit.

Pinned memory enables asynchronous transfers and better overlap.

However, pinning very large model allocations has system-level costs.

The runtime should support configurable budgets rather than blindly pinning hundreds of gigabytes.

For secondary GPUs, choose among:

- direct P2P transfer,
- NCCL where applicable,
- pinned-host staging.

Selection should depend on measured topology.

Do not assume NVLink or even good PCIe bandwidth.

---

# 22. Quantization Awareness

The scheduler must be quantization-aware.

An expert cache should only place an expert on a backend if that backend can execute the quantization efficiently.

Different formats may have dramatically different:

- bytes per expert,
- CPU throughput,
- CUDA throughput,
- conversion requirements.

The cost model should therefore key performance data by:

- expert tensor shape,
- quantization family,
- backend.

Do not hard-code policy such as:

IQ quant → CPU

or

K quant → GPU.

Measure it.

---

# 23. Shared Experts

Many architectures combine routed experts with one or more shared experts.

Shared experts are effectively always hot.

They should normally be handled separately from the dynamic expert cache.

If memory permits, shared experts should generally remain resident on the device executing the dense portion of the layer.

The normalized MoE metadata must explicitly identify them.

---

# 24. Dynamic / Unusual MoE Architectures

The design must anticipate:

- different expert count per architecture,
- different top-k values,
- shared experts,
- dynamic expert counts,
- expert-choice versus token-choice routing,
- different activation functions,
- fused expert representations,
- different router weighting,
- repeated expert IDs,
- architecture-specific state.

Do not build assumptions around Qwen's exact dimensions.

Backend test coverage is especially important because current `MUL_MAT_ID` semantics and corner cases such as repeated expert IDs have already required investigation in llama.cpp.

Unsupported architectures should fall back to the existing llama.cpp path rather than fail.

---

# 25. Runtime Fallback and Self-Protection

Adaptive mode must be capable of deciding that optimization is not beneficial.

Examples:

- too little spare VRAM,
- old/slow GPU,
- very fast CPU memory subsystem,
- poor PCIe link,
- low expert reuse,
- extremely flat routing,
- high cache churn,
- secondary GPU communication overhead exceeding its compute benefit.

Recent experimental llama.cpp cache testing has shown exactly this hardware dependence: some systems improved significantly while smaller/slower GPUs could regress.

Therefore periodically compare observed optimized execution cost against estimated baseline cost.

If the optimization is consistently slower:

- shrink cache,
- stop using a device,
- disable transfers,
- revert to normal CPU expert execution,
- or disable adaptive mode.

Automatic mode must prioritize "do no significant harm."

---

# 26. Observability

Expose enough telemetry to understand why performance is good or bad.

At minimum report:

- expert cache capacity per GPU,
- cache occupancy,
- expert hit rate,
- misses per token,
- experts executed per device,
- cache admissions/evictions,
- bytes transferred H2D,
- bytes transferred GPU-to-GPU,
- CPU expert execution time,
- GPU expert execution time,
- per-layer join wait,
- device utilization where available,
- routing entropy/skew,
- cache churn rate,
- prefill TPS,
- decode TPS.

For debugging, allow per-layer statistics.

This telemetry is essential because a poor result must be distinguishable between:

- bad placement,
- weak routing locality,
- PCIe bottleneck,
- CPU bottleneck,
- GPU bottleneck,
- synchronization overhead,
- poor quant kernel.

---

# 27. Correctness Requirements

For every supported architecture verify output against normal llama.cpp with adaptive scheduling disabled.

Tests should cover:

- one expert selected,
- multiple experts selected,
- repeated expert IDs if architecture permits them,
- shared experts,
- batch size 1,
- multiple tokens,
- prompt evaluation,
- decode,
- continuous batching,
- cache hit,
- cache miss,
- cache eviction,
- simultaneous CPU/GPU execution,
- secondary GPU execution,
- quantized experts,
- LoRA interaction where supported,
- context reset,
- model unload/reload.

Cache movement must never modify model weights.

Race conditions around slot reassignment must be specifically tested.

---

# 28. Performance Acceptance Criteria

Do not define success as "must achieve Strata TPS."

Performance varies dramatically by architecture and hardware.

Instead establish these goals:

### Baseline safety

Adaptive mode disabled must reproduce current llama.cpp behavior and performance within measurement noise.

### Automatic-mode safety

After warm-up, auto mode should detect and abandon configurations that remain materially slower than baseline.

### Single-GPU + CPU regime

Demonstrate improvement where dense tensors fit in GPU memory but a substantial portion of expert weights remain in RAM.

### Multi-GPU regime

Demonstrate that GPU0 and GPU1 can perform independent expert work during the same MoE layer instead of merely storing additional model weights.

### Heterogeneous concurrency

Demonstrate measurable overlap between:

- GPU0 expert execution,
- secondary GPU expert execution,
- CPU expert execution.

### Generic architecture support

Demonstrate the same scheduler mechanism on multiple structurally different MoE families rather than implementing separate runtime logic for each model.

Initial test candidates should include models representing significantly different MoE configurations, for example:

- Qwen MoE,
- GPT-OSS / OpenAI MoE,
- Tiel-Coder or comparable Qwen-derived hybrid MoE,
- Mixtral-style MoE,
- another large expert-count architecture available in llama.cpp.

---

# 29. Suggested Development Phases

## Phase 1 — Instrumentation

Implement architecture-independent expert identification and routing statistics.

No optimization yet.

Prove that routing can be observed correctly across several MoE architectures.

## Phase 2 — One GPU + CPU

Implement persistent GPU expert slots.

GPU calculates cache hits.

CPU calculates misses concurrently.

This is the simplest architecture that validates the central idea.

## Phase 3 — Dynamic Cache

Add online hotness tracking, admission and eviction.

Compare policies.

Separate decode and prefill behavior.

## Phase 4 — Secondary GPU Expert Backend

Add independent expert cache to GPU1.

Partition router-selected experts between:

- GPU0,
- GPU1,
- CPU.

Ensure concurrent execution and one aggregation barrier.

## Phase 5 — Hardware Cost Model

Add automatic profiling and scheduling.

Allow heterogeneous GPUs without fixed assumptions.

## Phase 6 — Architecture Expansion

Validate against all llama.cpp MoE architectures.

Move model-specific information into descriptors/adapters rather than scheduler branches.

## Phase 7 — Advanced Optimizations

Only after the fundamental scheduler works, investigate:

- expert prefetch,
- routing prediction,
- cache profile persistence,
- expert replication,
- NUMA-aware expert placement,
- grouped GEMM,
- CUDA graph compatibility,
- smarter continuous-batching cache policy,
- speculative/MTP interaction.

These are secondary optimizations and should not block the core architecture.

---

# 30. Explicit Non-Goals for Initial Implementation

Do not initially attempt to solve:

- MTP speculative decoding,
- SSD expert streaming,
- lossy expert pruning,
- router modification,
- KV cache redesign,
- distributed multi-machine inference,
- general tensor parallelism replacement,
- general ggml fork/join scheduling unless required.

Keep the first project focused on:

**intelligent expert residency + heterogeneous parallel expert execution.**

---

# 31. Key Architectural Rule

The scheduler must never assume:

> expert not in VRAM = CPU only

or:

> expert on GPU = always execute there.

Instead it should ask:

> Given the experts selected for this layer, the current cache state, device load, transfer costs, and hardware topology, which assignment minimizes the time until all required expert outputs are ready?

That question should be the central scheduling abstraction.

---

# 32. Target End State

The desired llama.cpp execution model should ultimately look like this:

Model router selects experts.

The generic MoE scheduler receives the selection.

It knows:

- which experts live on each GPU,
- which experts remain only in RAM,
- how fast every backend is,
- transfer costs,
- current routing history,
- current device load.

It creates an execution plan for that MoE layer.

GPU0, GPU1...GPU-N and CPU execute their independent portions concurrently.

Outputs return to the aggregation device.

The layer continues.

Hot experts gradually migrate toward the devices where they provide the greatest performance benefit.

Cold experts remain in RAM and are calculated efficiently by the CPU.

The process adapts automatically to both the loaded model and the user's hardware.

The implementation should therefore provide the **generalized mechanism behind Strata-style expert acceleration without making Qwen3.8-Flash-Next, a particular GPU size, or a static routing profile part of the architecture.**