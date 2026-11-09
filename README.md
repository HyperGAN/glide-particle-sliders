# GLiDE particle sliders

Native text-decision readout particles for [`fastino/GLiDE-0.4B`](https://huggingface.co/fastino/GLiDE-0.4B), using shared `particle-sliders-core` 0.13.0 and official ParticleGAN 0.9.5. Built with Fastino GLiDE.

Bounded tuning improved fresh synthetic preference/control behavior: relative error fell from 1.113 before tuning to **0.327**, versus OFF 1.000; a second seed scored 0.475. Both tuned runs kept all 16 correctness-control and 24 unrelated judgments correct. They selected concise replies in 62.5%/67.2% of main cases versus 0% OFF, with remaining option-order sensitivity. See [hyperparameters and fresh confirmation](docs/HYPERPARAMETERS.md), [measured throughput](docs/THROUGHPUT.md), and the preserved [initial negative experiment](docs/EXPERIMENT.md).

The [public repository](https://github.com/HyperGAN/glide-particle-sliders) contains the source and measurements for review. The implementation branch is `codex/glide-particle-sliders`; it is kept in a draft PR until reviewed. No model weights, server or deployment are published.

## Architecture

GLiDE is a bidirectional Ettin/ModernBERT encoder with 28 layers, width 1024 and a native bias-free `Linear(1024,255)` decision readout. It has no image encoder and does not generate text. Its 255 outputs are option positions, not semantic classes.

This product freezes the complete pretrained model and attaches the shared rank 8 / alpha 8 HNDL particle projection to its existing readout. A single real `128×4` trainable particle bank participates in the conditional PG game. The shared router conditions its readout changes on the native pooled representation. Only the first two active choice logits receive supervision; other adapter output rows remain exactly zero. This product is scoped to **two-choice text decisions**.

The selected v3 cohort applies TRAIN-only centering/global RMS to the adapter branch alone and uses a response-only HNDL residual critic. The fixed statistics serialize as network buffers; the pretrained readout still receives unchanged native inputs. The measured starting recipe is G=.001,D=.018,bank=.03,batch 32,horizon 512, no weight EMA. Original source-bound cohorts remain available separately.

The encoder is cached for head training. Caches come from the bundled native `Glide` runtime, using its dual tokenizers, chat template, truncation, left padding, positional IDs and final `(` pooling before `[SEP]`. The public inference API runs under `inference_mode`, so training calls the differentiable native readout with ordinary frozen tensors instead. A fresh-process live check attaches the same exported head to the real encoder and exercises `Glide.decide_batch`.

Dependency direction is product → shared sliders → official PG/HG/HNDL. The product contains no replacement GAN loss, optimizer, checkpoint engine or inference serialization. `binding.py` supplies only native geometry, construction, data rows, attachment and cleanup. Shared code performs updates, full-state continuation, evaluation, raw export and native-only reload. EMA is explicitly disabled. Execution is CPU-only in this tested cohort and rejects CUDA device options.

## Install and reproduce

Tested: Python3.12.15, Torch2.9.1+cu128 running on CPU, Transformers5.3.0, Tokenizers0.22.2, HNDL0.10.0. This differs from Fastino's published Torch2.14/Transformers5.17/BF16 GPU environment; results qualify this explicit CPU FP32 cohort only.

```bash
python3.12 -m venv .venv
.venv/bin/pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install '.[test]'
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

Output directories are immutable. Each candidate writes actual per-update logs, checkpoints at 32/64/128, and a raw inference artifact. The search selects the minimum RAW FP64 operand-first equal-example relative SSE on development examples at the same 128 completed updates. Test results are computed after the selection file is saved. A history-bearing 32→33 full-packet resume proof repeats just that next update, with no prefix replay. All finite candidates complete their budgets even if worse than OFF.

The current workspace's `.venv` reads previously installed, pinned dependencies through local `.pth` entries. It does not modify those environments. A clean installation can use the commands above. Actual interpreter and distribution provenance is recorded in each run's source receipt.

## Scope and license

The dataset is original synthetic preference examples, not Mikkel's actual preference labels. Candidate pairs, reversal groups and domains are split explicitly. This is neutral→concise instruction distillation; negative strength is extrapolation, and broader255-option transfer or encoder fine-tuning has not been established.

The independent integration source is [MIT licensed](LICENSE); [the top-level NOTICE](NOTICE) explains its scope and shared-code attribution. Fastino's [Community License](licenses/GLiDE-LICENSE) applies to its contributions and derivative model weights. It includes commercial attribution and revenue-dependent production terms. The [GLiDE NOTICE](licenses/GLiDE-NOTICE) and [Ettin MIT license](licenses/Ettin-encoder-400m-LICENSE) are preserved. This repository does not relicense the pretrained model or publish any weights.
