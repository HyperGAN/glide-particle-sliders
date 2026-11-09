# Historical GLiDE SSE search and separate fresh104 results

Current intended-effect selection is the [four-way batch8 RAW512 full-recipe tie](finished-output-learning-rates.md), not the SSE-selected batch32 recipe below. The current selector uses only equal-case concise-choice ON−OFF on16 original main DEV rows; preservation/control accuracy, SSE, probability and declared index are not selectors or tiebreaks. The reports, measurements and commands below retain their original historical scope. Reuse completed outputs/first counts; no new search, proof or fresh confirmation is required.

Bounded tuning produced a better measured preference/control tradeoff. On a new frozen set, RAW relative SSE fell from **1.113 before tuning to0.327**, versus OFF 1.000; a second training seed scored 0.475. Both tuned runs kept every tested correctness-control and unrelated-task choice correct. The tuned models make fewer concise selections than the aggressive pre-tuning controller, and option-order sensitivity remains substantial.

## What changed

All experiments retain the native frozen GLiDE encoder/readout, shared rank 8 / alpha 8 HNDL projection, real 128×4 routed bank, exact official PG 0.9.5 mechanics and no weight EMA. No GAN loss, optimizer, penalty or checkpoint implementation was replaced.

Earlier measurements found a large common offset in native representations and very little row variation inside the routed branch. A separate cohort therefore normalizes **only the adapter branch input** by TRAIN-only mean and global centered RMS; the native pretrained readout still receives the original representation. Both fixed statistics are raw network buffers and replay through native-only exports. A response-only 2→64→64→1 HNDL critic distinguishes the shared core's Gaussian reference from Gaussian-plus-paired-residual. Original and modified source-bound cohorts remain separate.

A fixed 2×2 comparison at 512 updates gave development errors .96096 (raw adapter/conditional16), .96733 (raw/residual64), 1.34865 (normalized/conditional16), and **.87985(normalized/residual64)**. The combination was selected on development relative SSE alone. This does not establish normalization alone as the cause.

## Fixed search

The earlier generator-LR sweeps kept multipliers constant, coupling generator, critic and bank rates. The new search varies independent **starting absolute** rates:

- Generator: .0003,.001,.003.
- Critic: .003,.018.
- Particle bank: .003,.03.

The historical search used12 combinations at batch8 and one additional batch32 configuration. Official multiplier inputs were D/G and bank/G, with actual group rates logged. All used seed7, horizon512 and RAW strength1; every finite declared budget completed and one exact historical anchor was reused. That search selected minimum FP64 operand-first equal-example development relative SSE with declared candidate-index tie handling. This preserves the old diagnostic procedure, not the current intended-effect selector or a future finish-worse-arms rule.

| G | D | Bank | Batch | Dev relative SSE |
|---:|---:|---:|---:|---:|
|.0003|.003|.003|8|.92550|
|.0003|.003|.03|8|.92315|
|.0003|.018|.003|8|.91387|
|.0003|.018|.03|8|.91101|
|.001|.003|.003|8|.95319|
|.001|.003|.03|8|.94497|
|.001|.018|.003|8|.89467|
|.001|.018|.03|8|.87985(reused)|
|.003|.003|.003|8|.26639|
|.003|.003|.03|8|.25737|
|.003|.018|.003|8|.23358|
|.003|.018|.03|8|.22355|
|**.001**|**.018**|**.03**|**32**|**.18218**|

The historical SSE-selected recipe was **G=.001,D=.018,bank=.03,batch 32,512 updates**, with multipliers 18 and 30. A predeclared seed 11 repeat changed only the seed and scored **.25858** on development data. It was not used to reselect a model. Higher generator LR also helped batch8 SSE fits in that study. The fresh set did not measure every batch8 recipe; no new confirmation requirement follows for the current four-way intended-effect tie.

Batch32 processes four times the example exposures per update as batch 8 and takes about 2.6 times as long per step in the benchmark. That was the stronger SSE-tested fixed-update recipe, not a compute-matched demonstration that batch size alone caused improvement or the current main-choice winner.

## Separate historical fresh104 measurements

The confirmation dataset contains 40 new preference scenarios (32 main, 8 correctness controls), each in both option orders, plus 12 unrelated binary classification/fact cases in both orders. Dataset SHA256 is`995871d69e3c1b7e0b8ee052d5e2fa68c9a472f8455408d39ba74cf2bf1b9059`. Artifact selection, both seeds, strengths and metrics were frozen before any native predictions on it. There was no confirmation-set training, filtering or subsequent model/strength selection.

Preference targets are explicitly synthetic:`T=N+delta`, preferred slot+1.5/other−1.5. Unrelated targets are exactly T=N; those 24 teacher-zero-energy rows are excluded from the primary relative SSE, while all student fields must still be finite. Their accuracy, flips, probability changes and KL are reported separately.

| Strength1 metric | OFF | Pre-tuning anchor | Tuned seed 7 | Tuned seed 11 |
|---|---:|---:|---:|---:|
|Raw relative SSE|1.000|1.113|**.327**|**.475**|
|Main preferred choices|0/64|60/64|40/64|43/64|
|Main scenarios correct in both orders|0/32|29/32|16/32|17/32|
|Correctness controls|16/16|14/16|**16/16**|**16/16**|
|Unrelated choices correct|24/24|23/24|**24/24**|**24/24**|
|Unrelated mean absolute probability change|0|.15788|**.00199**|**.00157**|

The tuned raw-error95%scenario-bootstrap intervals are **[.215,.458]** and **[.321,.643]**, both below OFF 1.000. They measure variation over this synthetic scenario set, not training-seed uncertainty or arbitrary user preferences.

Main mean P(preferred) rises from .2055 OFF to .5852(seed 7) and .5984(seed 11). The anchor reaches .6461 but introduces correctness/unrelated errors. Tuning therefore does not dominate every metric: it reduces the aggressive brevity bias while improving the prescribed raw metric and the measured control behavior.

Controls staying correct does not mean zero confidence cost. Mean correctness probability is .9787 OFF, .9857 for seed 7 and .9440 for seed 11. Main option-order disagreement remains8/32 scenarios for seed 7 and9/32 for seed 11. These are experimental results on synthetic two-choice judgments, eight correctness scenarios, twelve unrelated cases and two seeds; broad personalization and general non-interference are unqualified.

Every model also replayed through the full native public API on one main, one correctness and one unrelated scenario, each in both orders, at strengths 0,.5,1. Maximum probability discrepancy was below 1.4e-6. Zero strength reproduced the matching native readout exactly. RAW native-only exports require no training cache, critic or optimizer. Selected seed 7 and seed 11 full-state 256→257 packets matched exactly, with no prefix replay.

## Throughput, resources and accounting

[The CPU benchmark](THROUGHPUT.md) measured full native inference at 1.65–1.72 requests/sec, cached active-head API around 1 ms, and shared training at 32.7 updates/sec(batch 8) or 12.4 updates/sec(batch 32). The selected 512-update cached-head fit took about 43 seconds in the search. Encoder cost is excluded from cached-head measurements.

The search added 6144 unique updates; the second seed added 512. Two selected/replica next-update comparisons add two deliberate physical repeats. The separate speed benchmark added 192 updates. Across the complete investigation, genuine native-data training totals **10,304unique /10,311physical updates**, including seven deliberate next-update comparisons. Synthetic unit fixtures are excluded. All earlier checkpoints, failed launcher attempts, recovered sources and provenance corrections remain intact.

Current workloads were inspected before new work. Both GPUs were busy and the session's Babel mailbox remained unprepared, so there were no GPU leases or launches. The CPU search used at most two workers with four Torch threads each; the benchmark encoder was released before it began. No foreign workload, credentials or permissions were changed.

Machine evidence: [search plan](../reports/hpo-v5-plan.json), [search results](../reports/hpo-v5-results.json), [seed confirmation](../reports/hpo-v5-seed11.json), [frozen confirmation plan](../reports/hpo-v5-fresh-plan.json), and [fresh results](../reports/hpo-v5-fresh-results.json).

## Historical SSE-selected source-cohort reproduction

The following original commands describe the historical batch32 SSE-selected cohort, not all four current tied recipes. They are retained for provenance, not an instruction to rerun completed work. From the repository root, they required the pinned dependencies and locked local model:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 .venv/bin/glide-particle-sliders prepare \
  --model-dir artifacts/native-model --cache artifacts/native-cache.pt --readout artifacts/readout.pt
.venv/bin/python scripts/build_label_cache.py \
  --native-cache artifacts/native-cache.pt --output artifacts/label-cache.pt
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/train_normalized.py \
  --cache artifacts/label-cache.pt --output artifacts/selected-source-run \
  --critic residual64 --steps 512 --batch 32 --lr .001 --seed 7
```

This uses the original synthetic labels, not a learned model of Mikkel's personal preferences. The exact historical search also has immutable plans and source capsules under`artifacts/`; do not overwrite them when reproducing. The source supports shared registered training/inference backend`glide-normalized-readout-particles-v3`. The convenience script hardcodes D/G18 and bank/G30; use [the actual config option guide](finished-output-learning-rates.md) to understand all four tied tuples without repeating them.
