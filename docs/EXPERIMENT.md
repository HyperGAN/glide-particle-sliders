# GLiDE readout-particle experiments — 2026-11-06

The native integration works, including real 0.4B inference, shared routed-particle updates, exact continuation and live raw-export replay. **A useful concision preference has not been demonstrated.** The explicit-label probe shows a small context-dependent numerical edit, while a much simpler linear residual control fits the target substantially better.

## Model and integration

The official [GLiDE 0.4B repository](https://huggingface.co/fastino/GLiDE-0.4B/tree/6bda7ab91aeb36dcad6d27828a8871e548c7fc71) contains an Ettin/ModernBERT-derived bidirectional encoder, 28 layers / 1024 hidden dimensions / 16 heads, and a separate bias-free 1024→255 decision readout. The native path uses a render tokenizer plus the encoder tokenizer, `[CLS]`/`[SEP]`, head/tail truncation and the last real token before `[SEP]`. It returns option probabilities and accepts text/JSON, with no image encoder or generation. The released inference wrapper freezes weights and uses inference mode; its underlying readout is differentiable. Fastino's decision-training dataset and complete training recipe are not supplied by the release.

This product freezes all pretrained weights and adapts only that native readout with the shared rank 8 / alpha 8 HNDL routed branch and a 128×4 bank. It observes the two active native option logits. It does not approximate the encoder, project hidden states to satisfy a gate, fine-tune 400M parameters, or establish transfer to 255-option questions. Unused adapter output rows 2–254 remain exactly zero after actual training.

Shared core 0.13.0 Git`3dd672a7f9b8c169daef90fd63e4bb42b9ddc423`, official PG 0.9.5, HyperGAN Git`5f368a3770cf5bdae0448995c9f76e29af8049f7`, HNDL 0.10.0. CPU FP32 execution uses Python 3.12.15 / Torch 2.9.1+cu128 / Transformers 5.3.0 / Tokenizers 0.22.2, with four CPU threads. This differs from the vendor's newer GPU/BF16 cohort and is recorded independently.

## Data and metric

Original synthetic instructor-authored scenarios define concision when accuracy and completeness are equal. Quality controls make the shorter reply incorrect or incomplete. Every scenario is evaluated in both option orders with opaque`first`/`second` keys. There are 54 groups / 108 ordered examples: 26 train groups / 52 rows, 10 development groups / 20 rows, 18 test groups / 36 rows. Main test cases comprise 16 groups / 32 rows from two new domains; two additional control groups contribute four rows.

The sole ranking metric is RAW strength1 relative SSE. Student,teacher and neutral operands are cast to FP64 before subtraction; both native logit coordinates are pooled within each ordered example, then examples receive equal weight. Teacher-only exact-zero exclusions are fixed independently of the student; no examples were excluded in either run. OFF has an ideal score 1, exact teacher 0. Semantic probabilities, swaps and control choices are separate diagnostics and never reject a finite candidate.

The native cache uses batch2, while shared response evaluation uses one-row readout GEMM. Maximum neutral-logit roundoff is`9.5367431640625e-7`. Actual OFF scores are measured as well as the analytic baseline:1.00000064 for the instruction target and 1.00000002 for the label target. Strength 0 is bit-exact to the original native readout at the same one-row geometry.

## Native instruction-distillation result

Neutral and positive requests contain identical state/options; only the question instruction adds the concision preference. The student uses neutral frozen encoder inputs, with native positive-instruction logits as its teacher. Three generator learning rates completed 128 updates each, common horizon 128, batch 8, seed 7, critic multiplier 18 and prior multiplier 30. Weight EMA is None throughout. Final 128-step checkpoints were ranked only on development error.

| Generator LR | Completed updates | Development relative SSE |
|---|---:|---:|
|0.0001|128|1.0151435482|
|0.001|128|1.7250604888|
|0.01|128|4.9291773228|

The selected 0.0001 candidate scored**1.0035299780** on the test cache,95%group-bootstrap interval`[0.9268421,1.1058634]`. A train-fitted constant slot-logit edit scored0.9888786. The particle result does not improve on OFF.

The teacher also failed to express the intended preference: test mean P(concise) falls from native 0.2453213 to teacher 0.2307562. The native baseline and learned slider both choose the concise candidate 3/32 times; the teacher does so 1/32 times. This is a negative instruction-distillation result, not evidence that GLiDE cannot represent concision.

**Provenance correction:** initial encoding overlapped with finalization of one development control. Its exact original rows are recovered from the cache; metadata's dataset hash incorrectly identified the later file. The initial result used the old return-window control; no train/test rows changed. Original cache/results remain intact. [The correction record](../reports/experiment-v1-provenance-correction.json) identifies both development rows and the old task-family concern. The new builder hashes captured input bytes before encoding and refuses any concurrent file change. The follow-up re-encoded the two changed rows in four real native forwards, retaining all unaffected tensors bit-for-bit.

## Explicit synthetic-label capacity probe

This distinct diagnostic replaces the teacher with a declared synthetic edit:`T=N+delta`, preferred slot`+1.5`, other slot`−1.5`. Labels include the accuracy controls. It preserves the native logit common mode and gives nominal edit energy 4.5. These targets are **not** claimed to come from GLiDE instructions or from Mikkel's personal preferences.

The same three rates completed 512 updates each, common horizon 512, with the same seed,batch and optimizer multipliers. Selection used only final 512-step development error. This test set had already been viewed during the first experiment; all subsequent test measurements are **exploratory**, with no untouched confirmation claim.

| Generator LR | Completed updates | Development relative SSE |
|---|---:|---:|
|0.0001|512|1.0004367572|
|0.001|512|0.9609596242|
|0.01|512|1.0002387492|

The selected 0.001 candidate scored**0.9564043774**,95%group-bootstrap interval`[0.9173749,1.0044939]`. Constant slot bias scores 1.0000; scrambled input-dependent edits score 1.0562469. This supports a small contextual effect, with substantial uncertainty. Strength 0/.25/.5/.75/1 produces errors 1.0000000/.9881340/.9769127/.9663361/.9564044 and mean P(concise) .2453213/.2493313/.2534432/.2576581/.2619771. Main choice rate remains 3/32; controls remain 4/4. The target itself reaches mean P(concise) .8302838 and 32/32 main choices. This is far from a useful preference controller.

For the selected particle model, the actual bank's L2 change is 1.426042, active up-weight L2 is .9414903, inactive up-weight maximum is 0, pretrained readout bytes remain identical, and EMA shadow count is 0.

A separate closed-form ridge residual baseline uses the same native frozen features, train-only centering, and seven fixed regularization values selected on development data. Selected lambda .01 yields development error .2467681 and exploratory test error**.1968958**, mean P(concise) .6813774 and 23/32 main choices. Controls remain 4/4. However its option-reversal agreement is only 50%, with mean probability gap .2283, so it is not a robust preference product either. Its much lower numerical error suggests this frozen representation contains useful target information and motivates work on particle optimization/conditioning rather than assuming the base model is unsuitable. That is an inference from this bounded comparison.

## Mechanical evidence and accounting

- The real pinned0.4B model encoded 216 initial neutral/positive requests, with four more native forwards for the corrected development control. Initial direct/native public probability agreement was within 5.5e-8.
- Full-state next-update packets match exactly at 32→33 for the first run and 128→129 for the label probe, using fresh factories. Each repeats only one next update; neither replays a committed prefix. This qualifies the exact PG 0.9.5 cohort, not a separate upstream checkpoint audit or future fix.
- Raw native-only exports reload without a training cache, critic or optimizer. Fresh-process full-encoder public`Glide.decide_batch` replay checks four original/swapped test requests at strengths 0,.5,1. Maximum probability discrepancies are 3.26e-7 and 3.33e-7; the encoder remains frozen and OFF is exact at matching geometry.
- Actual genuine shared training: 384+1536=**1920unique committed updates**. Including the two deliberate next-update comparisons gives**1922physical updates**. Unit-test fixtures and the closed-form ridge solve are separate from this count.
- Final CPU suite: 21 tests plus 324 subtests pass. A wheel builds and its contents contain no model weights. Training process peak RSS was about 674 MiB; fresh full-model live-check peak RSS was about 3.1 GiB. There were zero GPU launches and no changes to existing workloads.

Machine-readable [first-run results](../reports/experiment-v1-results.json), [label-probe results](../reports/label-probe-v1-results.json), [ridge control](../reports/ridge-label-control-results.json), [live proof](../reports/label-probe-live-check.json) and [cache correction derivation](../reports/label-probe-input-derivation.json) accompany this report. Local checkpoints, raw adapters, exact source snapshots and downloaded assets remain under`artifacts/` and are excluded from publication.

## Remaining limits

Native model execution and faithful shared-particle mechanics are demonstrated. Useful concision control, broad preference learning, encoder-level adaptation, 255-option transfer and GPU behavior are unqualified. The source is a local candidate; no repository/PR, deployment or model publication was performed. The native Babel mailbox check reported that this delegated session was not prepared by its startup hook; no mailbox or credential state was created. CPU execution avoided the need for a new GPU lease, preserved Vox priority and never launched or resumed Glimmer.
