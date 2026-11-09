# GLiDE particle sliders

Native text-decision readout particles for [`fastino/GLiDE-0.4B`](https://huggingface.co/fastino/GLiDE-0.4B), using shared `particle-sliders-core` 0.13.0 and official ParticleGAN 0.9.5. Built with Fastino GLiDE.

**Current intended-effect decision:** four existing full recipes **arm04/05/06/07 tie**, all **G0.001/batch8/RAW512/horizon512/seed7**, with D/bank combinations retained separately. Each has mean binary concise-choice ON−OFF **1** over the original16 main ordered DEV rows. No unique D/bank recipe or default is selected. See [current rates and method](LEARNING-RATES.md), [all13 first-count results and exact options](docs/finished-output-learning-rates.md), and [portable metadata](docs/finished-output-rate.json).

The historical SSE-selected batch32 run and fresh104 results remain [separate diagnostic evidence](docs/HYPERPARAMETERS.md): relative SSE1.113→0.327 versus OFF1.000, second seed0.475, with original control/choice measurements unchanged. They do not select the current four-way tie or confirm all four recipes. Reuse completed searches, RAWs, outputs and first counts; no new confirmation is required. [Throughput](docs/THROUGHPUT.md) and the [initial negative experiment](docs/EXPERIMENT.md) retain their original scopes.

The reviewed implementation from [PR1](https://github.com/HyperGAN/glide-particle-sliders/pull/1) merged on 2026-11-09 at [`e058591d`](https://github.com/HyperGAN/glide-particle-sliders/commit/e058591d1ac12042609dfe58504cebc8d9235ae0); its reviewed Source head `29075f79269d1976071575e9d0ce8d569458fc3b` is retained as an ancestor. The [public repository](https://github.com/HyperGAN/glide-particle-sliders) publishes source and weight-free measurements. This source merge publishes no model weights, server or deployment and does not change captured native identities.

**Campaign status, 2026-11-09:** the existing frozen core0.13.0/PG0.9.5 native
bundle is Root-adopted, and twelve saved fresh-anchor OFF/ON decision previews
are published in the campaign sampler. These are separate from the original
main16 LR comparison. [Portable status evidence](docs/finished-output-rate.json)
records the actual adoption/publication artifact identities. Ordinary Generate
remains unavailable and published weights remain zero; no new native proof,
training, inference or rating was required for this status update.

## Architecture

GLiDE is a bidirectional Ettin/ModernBERT encoder with 28 layers, width 1024 and a native bias-free `Linear(1024,255)` decision readout. It has no image encoder and does not generate text. Its 255 outputs are option positions, not semantic classes.

This product freezes the complete pretrained model and attaches the shared rank 8 / alpha 8 HNDL particle projection to its existing readout. A single real `128×4` trainable particle bank participates in the conditional PG game. The shared router conditions its readout changes on the native pooled representation. Only the first two active choice logits receive supervision; other adapter output rows remain exactly zero. This product is scoped to **two-choice text decisions**.

The selected v3 cohort applies TRAIN-only centering/global RMS to the adapter branch alone and uses a response-only HNDL residual critic. The fixed statistics serialize as network buffers; the pretrained readout still receives unchanged native inputs. The historical SSE-selected recipe was G=.001,D=.018,bank=.03,batch32,horizon512, no weight EMA; it is not the current intended-effect recommendation. The four tied batch8 recipes and exact multiplier literals are recorded above. Original source-bound cohorts remain separate.

The encoder is cached for head training. Caches come from the bundled native `Glide` runtime, using its dual tokenizers, chat template, truncation, left padding, positional IDs and final `(` pooling before `[SEP]`. The public inference API runs under `inference_mode`, so training calls the differentiable native readout with ordinary frozen tensors instead. A fresh-process live check attaches the same exported head to the real encoder and exercises `Glide.decide_batch`.

Dependency direction is product → shared sliders → official PG/HG/HNDL. The product contains no replacement GAN loss, optimizer, checkpoint engine or inference serialization. `binding.py` supplies only native geometry, construction, data rows, attachment and cleanup. Shared code performs updates, full-state continuation, evaluation, raw export and native-only reload. EMA is explicitly disabled. Execution is CPU-only in this tested cohort and rejects CUDA device options.

## Install and historical reproduction workflow

Tested: Python3.12.15, Torch2.9.1+cu128 running on CPU, Transformers5.3.0, Tokenizers0.22.2, HNDL0.10.0. This differs from Fastino's published Torch2.14/Transformers5.17/BF16 GPU environment; results qualify this explicit CPU FP32 cohort only.

The commands below record the original installation/experiment/live-check
workflow. The `experiment` command retains its historical SSE/index selection;
it is not the current intended-effect selector or an instruction to repeat
completed work. Current option placement and artifact limitations are in the
[scoped guide](docs/finished-output-learning-rates.md).

**Historical CLI limitation:** run the receipt-producing experiment/helpers from
this source checkout with this product installed editable (`pip install -e`).
Their existing `source_receipt()` locates `src`, `pyproject.toml` and
`native-model.lock.json` relative to the product Source file; an ordinary wheel
installation does not supply that layout and the experiment command fails when
it constructs the receipt. The pinned shared core remains a non-editable Git
installation. The example below shows the product-editable placement only;
it adds no packaging fix or instruction to repeat completed experiments.

```bash
python3.12 -m venv .venv
.venv/bin/pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -e '.[test]'
python3 scripts/download_model.py
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/glide-particle-sliders prepare \
  --model-dir artifacts/native-model \
  --cache artifacts/cache.pt --readout artifacts/readout.pt
CUDA_VISIBLE_DEVICES='' .venv/bin/glide-particle-sliders experiment \
  --cache artifacts/cache.pt --readout artifacts/readout.pt \
  --output artifacts/run-1 --steps 128 --rates 0.0001 0.001 0.01
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/glide-particle-sliders live-check \
  --model-dir artifacts/native-model --cache artifacts/cache.pt \
  --experiment artifacts/run-1
.venv/bin/python -m pytest -q
```

Model revision: `6bda7ab91aeb36dcad6d27828a8871e548c7fc71`. Every required asset is SHA256-locked. Core Git pin: `3dd672a7f9b8c169daef90fd63e4bb42b9ddc423`; runtime validates immutable Git origin and the shared native-source checks. Downloading is an explicit separate preparation step; training and live checks load local files only.

In that historical workflow, output directories were immutable. Each candidate wrote per-update logs, checkpoints at32/64/128 and a RAW inference artifact. The original128-update search selected minimum RAW FP64 equal-example relative SSE, then evaluated test results; its32→33 proof deliberately compared one next update. Those completed results and code remain unchanged, including the historical finish-every-finite-arm policy. They are not future search/proof/confirmation requirements. Current selection uses only equal-case main concise-choice ON−OFF, retains the four full ties, and reuses completed work without replaying updates.

The original workspace's `.venv` used previously installed pinned dependencies
through local `.pth` entries; that is historical execution provenance, not a
portable environment promise. The receipt-producing CLI requires the source
layout described above. Registered response/inference bindings are separate
package APIs and do not call that CLI receipt helper; this distinction is not
new wheel/native qualification. All saved Source, model and RAW identities
remain unchanged.

## Scope and license

The dataset is original synthetic preference examples, not Mikkel's actual preference labels. Candidate pairs, reversal groups and domains are split explicitly. This is neutral→concise instruction distillation; negative strength is extrapolation, and broader255-option transfer or encoder fine-tuning has not been established.

The independent integration source is [MIT licensed](LICENSE); [the top-level NOTICE](NOTICE) explains its scope and shared-code attribution. Fastino's [Community License](licenses/GLiDE-LICENSE) applies to its contributions and derivative model weights. It includes commercial attribution and revenue-dependent production terms. The [GLiDE NOTICE](licenses/GLiDE-NOTICE) and [Ettin MIT license](licenses/Ettin-encoder-400m-LICENSE) are preserved. This repository does not relicense the pretrained model or publish any weights.
