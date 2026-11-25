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

### Full-suffix fit check

One original fit case exercised the authored full-suffix probe at Block 23 on
the final RAW80 checkpoint. The run used the original fit's global strength
of 1.0 and the complete 123-position aligned suffix. The live coefficient
varied by position at both visits in the native 44-call route.

| Native visit | Nonzero positions | Distinct applied coefficients | Signed dose |
| --- | ---: | ---: | --- |
| 30 | 119 of 123 | 120 | `64384529523/134217728` |
| 36 | 115 of 123 | 116 | `613403837177/1073741824` |

The signed dose sums the applied coefficients before the BF16 residual cast.
The complete canonical reply, including the choice and reported probabilities,
matched the saved stock OFF reply for the same fit request on the same physical
GPU. The comparison used the two retained records.
The [fit-only evidence](evidence/site23-full-suffix-fit-runtime-20261125.json)
records the binary64 constants immediately below and above each exact mean.
At this width the two means lie between representable values. Those endpoints
are available for sensitivity comparisons. Development and held-out effect
measurements are pending.

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

This example reproduces the registered 7.5 development arm.

Pass the returned digest as `--plan-sha256` and supply the same evidence file using `--development-evidence`. The existing `adaptive`, `off` and `matched` modes use this plan. For `matched`, provide the adaptive trace for the same request and plan. The shared binary32 matcher chooses the nearest representable constant by default. Use `--match-side lower` or `--match-side upper` to reproduce either bracket control. The trace gives the exact rational dose of each arm, the signed constant-minus-adaptive residual, and whether the match is exact at the coefficient boundary before BF16 writeback.

### Matched constant controls

The registered 16-case development panel has 144 native constant controls
across the six strengths. Dose is the sum of the two binary32 scalar
coefficients before the native BF16 residual writeback. The 48 exact-dose
comparisons use one constant each. The remaining 48 use the two binary32
coefficients on either side of the adaptive mean, repeated at both visits.

At 7.5, adaptive steering chose the intended answer on 10 of 16 cases and
each constant control arm did so on 9. At each other strength, adaptive and
constant controls chose the intended answer on 10 of 16 cases. The
registered ranking uses the equal-case difference between adaptive ON and
matched OFF. All six strengths tied at +2 of 16 over OFF, so strength
selection and the default preset remain unset.
[Control counts and source digests](evidence/block17-development-constant-controls-20261124.json)

### Third development panel

Each of the six registered strengths scored 14 of 16 intended choices on
the same order-balanced cases. All 16 choices per strength matched the
stock OFF choices. The nonzero coefficients changed the BF16 activation
on all 192 visits across the six runs. The equal-case gain was zero for
each strength, so the registered rule left strength selection unset.
The matched-constant timing study remains a separate comparison.
[Third-panel results and source digests](evidence/block17-third-development-20261124.json)

## Layer 15 confirmation

The separate Layer 15 study tested the frozen nominal strength of 5 on 16
fresh text-choice cases from eight public facts, with both answer orders.
The stock baseline chose the intended concise answer in 10 of 16 cases.
Stepped adaptive steering and its casewise exact-dose constant control each
did so in 15 of 16, choosing the same answer on every case. Steering
improved the score by 5 of 16 over stock. Timing added zero over the
constant control, so the registered criterion left the default strength
unset.

Both steering arms ran on the same GPU at native visits 16 and 22. The
binary64 constant repeated the exact mean of the two resolved adaptive
coefficients. This matches their signed sum before BF16 writeback. The
first BF16 edit still differed between the two arms in all 16 cases.
[Per-case choices, coefficient doses, and source digests](evidence/layer15-exact-dose-confirmation-20261124.json)
