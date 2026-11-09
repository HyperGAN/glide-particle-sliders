# Concision-choice LR decision: four full recipes tie

The existing thirteen-arm RAW512 comparison retains **arm04/05/06/07** as an
exact tie. All four use **G0.001, batch8, horizon512 and seed7**, but their
D/bank pairs differ. Each scores **1** on the single intended-effect metric.
No unique D/bank/full recipe or package default is selected, and no index,
SSE, batch, preservation or fresh-confirmation tiebreak is applied.

## One metric, original first counts

Intensity is binary:1 if the native chosen option is the original concise
slot, otherwise0. Compute intended-effect ON at strength1 minus exact matched
OFF at strength0, averaged equally across eight original main scenarios and
their two opposite option orders. With two equally weighted orders per
scenario, this is exactly `(ON_concise_count - OFF_concise_count) / 16`.
Correctness-control and unrelated rows, pooled all-row accuracy, probabilities,
confidence, teacher/student SSE and preservation are excluded from selection.

These are original authored synthetic-label DEV choices, not a new model
judgment, human preference/generalization result or untouched holdout. Reuse the
saved native first counts; do not rerun training, inference, proofs or ratings.
Per-case HPO transcripts are unavailable. Counts determine the exact linear
mean but cannot identify which12 rows were concise for a0.75 arm; no per-row
judgments are invented. Legitimate saved zeros remain zero, negatives would
remain negative, exact ties remain tied and technical failures remain missing.

| Arm | G | Actual D | Actual bank | Batch | ON concise | OFF concise | Mean ON−OFF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| arm-00 | 0.0003 | 0.003 | 0.003 | 8 | 0/16 | 0/16 | 0.0 |
| arm-01 | 0.0003 | 0.003 | 0.03 | 8 | 0/16 | 0/16 | 0.0 |
| arm-02 | 0.0003 | 0.018 | 0.003 | 8 | 0/16 | 0/16 | 0.0 |
| arm-03 | 0.0003 | 0.018 | 0.03 | 8 | 0/16 | 0/16 | 0.0 |
| arm-04 | 0.001 | 0.003 | 0.003 | 8 | 16/16 | 0/16 | 1.0 |
| arm-05 | 0.001 | 0.003 | 0.03 | 8 | 16/16 | 0/16 | 1.0 |
| arm-06 | 0.001 | 0.018000000000000002 | 0.003 | 8 | 16/16 | 0/16 | 1.0 |
| arm-07 | 0.001 | 0.018000000000000002 | 0.03 | 8 | 16/16 | 0/16 | 1.0 |
| arm-08 | 0.003 | 0.003 | 0.003 | 8 | 12/16 | 0/16 | 0.75 |
| arm-09 | 0.003 | 0.003 | 0.03 | 8 | 12/16 | 0/16 | 0.75 |
| arm-10 | 0.003 | 0.018 | 0.003 | 8 | 12/16 | 0/16 | 0.75 |
| arm-11 | 0.003 | 0.018 | 0.03 | 8 | 12/16 | 0/16 | 0.75 |
| arm-12 | 0.001 | 0.018000000000000002 | 0.03 | 32 | 12/16 | 0/16 | 0.75 |

This was a joint G/D/bank recipe comparison; arm12 also changed batch. It does
not isolate a causal G or batch effect or establish a global/joint optimum.
Original normalized/residual64 source, CPU FP32, RAW512/h512/seed7 and all
saved source/checkpoint identities remain authoritative. The official scalar
optimizer statistics are unchanged; model-weight EMA stays disabled for
training, selection, exports and inference.

## Exact tied options and existing API

| Arm | lr | d_lr_mult | prior_lr_mult | ema_decay |
| --- | ---: | ---: | ---: | --- |
| arm04 | 0.001 | 3.0 | 3.0 | null |
| arm05 | 0.001 | 3.0 | 30.0 | null |
| arm06 | 0.001 | 18.0 | 3.0 | null |
| arm07 | 0.001 | 18.0 | 30.0 | null |

The common names mean G=`lr`, D=`lr*d_lr_mult`, bank=`lr*prior_lr_mult`.
Actual D for multiplier18.0 is0.018000000000000002, rather than a reconstructed
rounded literal. Copy the original tuples. All four full option fragments in
[the portable summary](finished-output-rate.json) also preserve batch8,
steps512, seed7, critic residual64, CPU FP32, continuous penalties and the
release-0.9.5 profile. Only the deployment-specific `cache_file` path is omitted
explicitly; these fragments are not substituted full source-bound configs.

The existing [normalized configuration factory](../src/glide_particle_sliders/binding_normalized_v3.py)
returns a common SessionConfig for backend
`glide-normalized-readout-particles-v3`, cohort
`glide-04b-normalized-adapter-pg095-raw-v3`. For example, this shows **arm04's
accepted option shape**, not a command to repeat completed work:

```python
from glide_particle_sliders.binding_normalized_v3 import config

cfg = config(cache_file=caller_owned_original_cache_path,
             lr=0.001, d_lr_mult=3.0, prior_lr_mult=3.0,
             batch=8, steps=512, seed=7, critic="residual64",
             device="cpu", base_precision="float32", ema_decay=None,
             penalty_input_domain="continuous", pg_source_profile="release-0.9.5")
```

The local cache must retain its original model/data/source identity.
[`scripts/train_normalized.py`](../scripts/train_normalized.py) exposes lr,
batch, steps, seed and critic but hardcodes multipliers18/30; it cannot express
all four tied recipes. The historical experiment/HPO CLI scripts retain their
SSE/index selection behavior and are not the current intended-effect selector.
No new script, default or execution protocol is introduced here.

The historical experiment/helper scripts that call `cli.source_receipt()`
require this product's source checkout/editable installation: the helper looks
for repository-root `src`, `pyproject.toml` and `native-model.lock.json` relative
to its Source file. A normal wheel-installed experiment CLI lacks that layout.
The pinned shared core remains non-editable. Registered response/inference
bindings do not call this CLI receipt helper; copying the scalar options is
still not independent wheel/native qualification or migration of old RAWs.
See [the installation scope](../README.md#install-and-historical-reproduction-workflow).

## Historical SSE and separate fresh104 scope

The historical SSE-selected **arm12/batch32/G0.001/D0.018000000000000002/
bank0.03** recipe has current main-choice gain **0.75**, not the tied maximum1.
Keep all original SSE values, reports, source locks and presets unchanged as
diagnostics; do not use SSE or declared arm index to choose among the four ties.
The original fresh104 results cover the historical anchor and selected seed7/
seed11 models only. They remain separate from this original thirteen-arm DEV
ranking and do not confirm all four tied full recipes. No additional fresh
confirmation is required. See [the historical study](HYPERPARAMETERS.md).

The retained model/cache/RAW/FULL artifacts are not public downloads bundled
here. The summary provides first counts, exact options, original observation
pointers and source hashes for scalar arithmetic review. This decision does
not establish a newly Root-adopted native bundle, current-core qualification,
sampler publication or ordinary Generate availability. Implementation merge
and product integration remain separate work; this documentation authorizes
no new execution or repeated completed search.
