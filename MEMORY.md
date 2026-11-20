# Memory requirements

| Operation/profile | GPU count | Calculated VRAM per GPU (GiB) | Calculated host RAM (GiB) | Input limits / precision |
|---|---:|---:|---:|---|
| Native encoder cache preparation / decision | 0 | 0 | 8 | CPU FP32; SDPA; batch 2; bounded text≤ 512 tokens including rendered choices; two choices only |
| Native full-context encoder decision upper envelope | 0 | 0 | 22 | CPU FP32; batch 2; SDPA math fallback; 7991 tokens+8special-token allowance≤ 7999 |
| Cached readout PG training / inference | 0 | 0 | 4 | CPU FP32; encoder unloaded; ≤ 512 cached rows; batch 2; rank8 native 1024→255 readout |

These are **calculated planning budgets, not measured peaks or hard runtime ceilings**. GiB means 2³⁰ bytes. The table bounds one operation at a time; do not add sequential phases, and do add concurrently resident models/jobs. Longer contexts, larger batches, reference audio or different backends require recalculation; the package does not have a universal finite maximum for every accepted request. No benchmark, model construction, forward, GPU query or training run was performed for this document.

Frozen BF16/FP16 weights cost `2P`; FP32 costs `4P`. LoRA costs `r × Σ(in + out)`. The root particle bank is counted once (128 × 4 FP32 values). Rank-8 shared RoutedMLP branches have 6,476 parameters each: router 8→16→16→16→4 and network 12→48→48→48→8; graph variants below use this conservative branch allowance. Critic allocations are included. Frozen weights have no parameter gradients or optimizer moments.

For ParticleGAN 0.9.5 `bispf_optimized`, DualNorm has zero momentum: do **not** add Adam's two full-sized moments. We reserve **64 bytes per trainable scalar** for FP32 parameters/gradients, old/displaced parameters and normalization/constraint temporaries, including protected-normal `.double()` buffers; small scalar step/constraint state fits the workspace allowance. This is a conservative transient envelope, not 64 bytes of persistent optimizer state. No model-weight EMA copy is allocated in the supported RAW policy. A separately identified supervised AdamW comparator uses its actual two FP32 moments.

Activation estimates deliberately include FP32 routed branches and two FP32 attention-sized buffers for eager/math attention. With `L` layers, width `H`, FF width `I`, query heads `N`, tokens `T` and effective accumulated batch `B`, the uncheckpointed training allowance is `64BLTH + 12BLTI + 8BLNT²` bytes. With block checkpointing it is `8BLTH + 64BTH + 12BTI + 8BNT²`; hybrid attention uses only its full-attention layer count in the quadratic term, with recurrent state added separately. These are explicit planning allowances for saved tensors/recomputation, not a promise about every allocator/kernel. Sampling prefill uses `16BTH + 8BTI + 8BNT²`, plus actual KV/recurrent state and any materialized logits. Runtime/workspace allowances below cover Python/framework/CUDA workspaces, allocator fragmentation and the small game state; an executor must still enforce live resource checks.

Host budgets count the stated CPU loading peak, tensor caches and serialization copies plus workspace. They exclude an optional **reclaimable OS file cache** of the complete checkpoint: add the checkpoint file size if deliberately retaining that cache in RAM. Model files remaining on disk are not resident GPU weights. Training duration/update count changes checkpoint disk consumption, not this live memory calculation. RAW/FULL serialization can add CPU copies of trainables; those are included in the host allowances.

Model: **fastino/GLiDE-0.4B**, revision **`6bda7ab91aeb36dcad6d27828a8871e548c7fc71`**.

The table rounds each calculated phase sum upward to the next2 GiB. The byte ledger below gives the unrounded calculation; per-GPU split estimates are listed in device order.

| Phase | Allocation ledger (GiB before rounding) |
|---|---|
| Native encoder cache preparation / decision | frozen_CPU: 1.4716; checkpoint_load_buffer: 0.7354; activation_upper: 0.0981; workspace: 4.0000 |
| Native full-context encoder decision upper envelope | frozen_CPU: 1.4716; checkpoint_load_buffer: 0.7354; full_context_attention_and_activation_upper: 15.7808; workspace: 4.0000 |
| Cached readout PG training / inference | readout: 0.0010; trainable_transient: 0.0020; cache: 0.0024; workspace: 2.0000 |

Detailed arithmetic and assumptions:

- **FP32 encoder and readout bytes**: `1580171264`.
- **LoRA elements**: `10232`.
- **generator plus critic allowance elements**: `33733`.
- **cache bytes**: `2623488`.

Scope and limits:

- The actual native cohort is CPU-only and refuses CUDA. Zero VRAM is intentional; this is a text decision encoder, not an image encoder or text generator.
- Only the frozen encoder is used for preparation and live decisions. Training consumes exact cached1024-wide encoder states and the original 255-output readout; it does not keep or backpropagate through the encoder.
- 512 tokens is this planning profile, not the full 7999-token model limit. At7999 tokens the two FP32 attention-sized batch 2 buffers alone are15.26 GiB; recalculate the CPU budget before a full-context request. No quantized/GPU route is claimed.

Source facts (read as text; no model execution):

- [src/glide_particle_sliders/native.py](src/glide_particle_sliders/native.py)
- [src/glide_particle_sliders/binding.py](src/glide_particle_sliders/binding.py)
- [native-model.lock.json](native-model.lock.json)
- [src/glide_particle_sliders/__init__.py](src/glide_particle_sliders/__init__.py)
- [Pinned model metadata](https://huggingface.co/fastino/GLiDE-0.4B/tree/6bda7ab91aeb36dcad6d27828a8871e548c7fc71) (architecture/parameter metadata or exact file-size lock as described above).
- ParticleGAN 0.11.0, `recipes.py:bispf_optimized`, `optim/dualnorm.py`, `optim/direction_blend.py` and `optim/constraint_geometry.py`; shared core `routed_projection.py`, `reference.py:RoutedMLP`, `audio.py:AudioResponseCritic`. These optimizer formulas are shared dependencies, not a new optimizer implementation.

Calculated 2026-11-14. Counts, live runtime qualification and memory requirements are separate facts.
