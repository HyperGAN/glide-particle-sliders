# GLiDE-9B recurrent sliders

This opt-in package steers the final input token in the pinned text-only GLiDE-9B publisher runtime. The shared `PromptSteering` probe computes the coefficient from each visit's own residual. The first edit feeds into the second visit.

## Block 23 calibrated replay

Install `pip install ./native9b` in the pinned Python 3.11 and CUDA environment. The entry point accepts a local, verified publisher snapshot, a completed author fit held as RAW and FULL checkpoints, and the selected strength from that fit's calibration. A replay plan binds these files and their digests.

```sh
glide-recurrent-sliders --model-dir "${GLIDE_9B_SNAPSHOT}" \
  --request request.json --plan replay.json --plan-sha256 "${REPLAY_SHA256}" \
  --raw raw.pt --full full.pt --full-receipt full-receipt.json \
  --calibration calibration.json --mode adaptive
```

The request uses the publisher's native `state` and one two-option choice question in `questions`. The response includes the publisher's decision and a trace with both resolved coefficients. Use `--mode off` for the stock baseline. Use `--mode matched --reference adaptive.json` with a saved adaptive result for the same request and replay. The shared binary64 dose matcher gives both passes the mean coefficient when that mean is exactly representable. Dose here refers to the coefficients before the native BF16 residual cast.

The `glide-recurrent-replay-v1` JSON includes `identity` from `glide_recurrent_sliders.identity.model_identity()`, `raw_sha256`, `full_sha256`, `fit_recipe_sha256`, `completed_updates`, `calibration_sha256` and `ema_decay` set to null. The `glide-recurrent-calibration-v1` JSON includes `raw_sha256`, the measured `selected_strength`, and `selection_metric` set to `mean_matched_intended_effect_on_minus_off`. The holdout evaluation belongs to the experiment consuming this replay.

Use the root package and its existing entry point for GLiDE-0.4B readout sliders.

## Block 17 development replay

Block 17 visits 18 and 24 in the exact 44-call stock route. At each visit the runtime scales the shared FP32 probe coefficient, masks to the final input token, performs the FP32 outer product and add, and casts the edit once to BF16. All six co-leading strengths in the [verified development comparison](evidence/block17-development-20261123.json) produced the same choices. The development plan leaves `selected_strength` null.

Supply the exact RAW80, FULL80 and FULL receipt from the registered fit. The [development evidence](evidence/block17-development-20261123.json) binds the eligible trials and those artifact digests. The installation does not distribute model or fitted weights.

```python
from glide_recurrent_sliders import write_block17_development_plan

digest = write_block17_development_plan(
    "block17-development.json",
    trial_strength=7.5,
    raw_path="raw80.pt",
    full_path="full80.pt",
    full_receipt_path="full80-receipt.json",
    development_evidence_path="native9b/evidence/block17-development-20261123.json",
)
```

Pass the returned digest as `--plan-sha256` and supply the same evidence file using `--development-evidence`. The existing `adaptive`, `off` and `matched` modes use this plan. For `matched`, provide the adaptive trace for the same request and plan. The shared binary32 matcher chooses the nearest representable constant by default. Use `--match-side lower` or `--match-side upper` to reproduce either bracket control. The trace gives the exact rational dose of each arm, the signed constant-minus-adaptive residual, and whether the match is exact at the coefficient boundary before BF16 writeback.

### Third development panel

Each of the six registered strengths scored 14 of 16 intended choices on
the same order-balanced cases. All 16 choices per strength matched the
stock OFF choices. The nonzero coefficients changed the BF16 activation
on all 192 visits across the six runs. The equal-case gain was zero for
each strength, so the registered rule left strength selection unset.
The matched-constant timing study remains a separate comparison.
[Third-panel results and source digests](evidence/block17-third-development-20261124.json)
