# Explicit cached fresh95 normalized-readout opt-in

`glide_particle_sliders.fresh95` is a separate initialization profile. Existing
registered readout/residual/normalized backends, defaults, learning-rate ties and
old FULL/RAW checkpoints keep their original routes. Completed cohorts are not
reset or migrated.

Use a genuinely installed `glide-particle-sliders==0.1.0` in its compatible
original core0.13.0 Git3dd672a7/PG0.9.5/HNDL/HyperGAN runtime. Original Source
and noneditable core-origin guards remain active. Generated RECORD must own
the four captured original modules and all seven fresh95 files. A Source
prepend, editable checkout or counterfeit metadata cannot replace that owner.
No dependency/version/entrypoint or core upgrade is added here.

The caller supplies the actual frozen CPU FP32 bias-free1024→255 readout,
its immutable model/revision/readout-weight identity, and completed original
PreparedReadout records with splits and whole output/Source origins. The owner
authenticates those inputs. This API never runs an encoder, `prepare`, a teacher
or native cache reconstruction. Targets retain their exact declared semantics:
`native-positive-logits`, `declared-synthetic-preference-logits`, or
`synthetic-software-only`. Synthetic preference logits must not be relabeled as
native positive-instruction teacher outputs. No native cache/encoder availability
is supplied or implied by this guide.

The original NormalizedReadoutSlider and ResidualCritic64 are retained. Down QR
uses the original HNDL Kaiming(a=sqrt5) variance1/(3*fan_in), with the same
Parameter objects. UP remains zero. The128×4 bank uses declared R2Normal(0,1);
router/bridge/critic constructor values remain KEEP. Only added trainables are
initialized, once before optimizers, with a receipt in the immutable signature.
TRAIN-only center/global coordinate-RMS buffers remain RAW state, not EMA. The
frozen readout sees its original input; response_scale remains1. Weight EMA is
refused and inference uses raw current weights.

These are library calls with caller-supplied data, paths and functions.
`component_factory(request)` returns the authenticated original readout;
`owner_admission(request)` authenticates origins and admits the operation. A
fresh cohort and owner directory must be new. `config` has explicit
seed/steps/batch/lr/d_lr_mult/prior_lr_mult/model_identity, `ema_decay=None`,
`device="cpu"`, `pg_version="0.9.5"`, `pg_source_profile="release-0.9.5"`,
and `penalty_input_domain="continuous"`. This namespace selects no rate.

```python
from glide_particle_sliders.fresh95 import (
    save_prepared_cache, open_training, open_inference,
)

cached = save_prepared_cache(
    new_cache_path, prepared=completed_original_rows, splits=original_splits,
    teacher_origins=authenticated_origins, model_identity=model_identity,
    cache_id=cache_id, owner_admission=owner_admission,
)
with open_training(
    config, cached, new_directory, mode="fresh", cohort_id=new_cohort,
    component_factory=component_factory, owner_admission=owner_admission,
    ema_decay=None,
) as run:
    # Entry FULL0/RAW0 and initialization receipt are already durable.
    # Call run.update() only for the new work your application intends to run.
    full = run.save(new_FULL_path)
    raw = run.export(new_RAW_path)

with open_training(
    config, cached, resume_directory, mode="resume", resume=full,
    cohort_id=new_cohort, component_factory=component_factory,
    owner_admission=owner_admission, ema_decay=None,
) as resumed:
    pass  # Whole game/network/critic/optimizers/cursors/RNG restored.

with open_inference(
    raw, component_factory=component_factory, owner_admission=owner_admission,
) as inference:
    logits = inference.response(already_encoded_hidden, strength=1., seed=7)
```

Resume requires this new profile's exact typed FULL, receipt and original cache;
it skips PG initialization and compares the whole restored state directly. RAW
inference strictly restores added tensors and normalization buffers, while
skipping cache payload, critic/game/optimizers and PG initialization. Original
added-architecture constructors still run before RAW values are restored.
The original finite-strength capability remains unchanged. This response API
returns the original active logits; it does not load an encoder, reproduce
`Glide.decide_batch`, or add an ordinary Generate command.

Closing blocks work immediately, checks exact marker/Parameters/callback/
normalization-handle ownership, retries only pending cleanup, preserves foreign
changes on refusal, and removes the marker last. Native model lifetime and
resource admission stay with the caller.

Four genuine installed-owner software cases passed with one total tiny update.
They establish the tested software lifecycle, not pretrained encoder/native
quality, public default qualification or a new intended-effect LR result. The
existing four scoped RAW512 ties and all earlier LR/native evidence remain
unchanged; no old test/proof/search/teacher/output is replayed by this addition.

## Explicit installed-owner software check

The retained four-case fixture is published as `tests/fresh95_owner_check.py`.
That filename is outside pytest’s default `test_*.py` and `*_test.py` discovery,
so ordinary repository CI keeps collecting its existing tests. The fixture body
is unchanged from the retained installed-owner software observations. Publication
routing adds no new software pass or native qualification.

This check requires the genuine installed wheel’s generated RECORD to own both
the original namespace and all seven fresh95 modules, with the frozen compatible
core, ParticleGAN and other runtime dependencies described above. An editable
checkout or a Source-tree import is not that installed owner. For an explicitly
requested owner check, use that compatible installed environment, an independent
working directory, an empty PYTHONPATH and a neutral pytest config:

```sh
cd /absolute/path/to/neutral-working-directory
PYTHONPATH= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /absolute/path/to/compatible-env/bin/python -m pytest \
  -q -c /dev/null \
  /absolute/path/to/checkout/tests/fresh95_owner_check.py
```

The paths are caller-supplied. The explicit file argument selects these four
software cases; `-c /dev/null` avoids repository configuration and its Source
pythonpath. The fixture includes one total tiny update. This proposal does not
execute or replay it, and the earlier completed software observations remain
separate from public CI and pretrained/native qualification.
