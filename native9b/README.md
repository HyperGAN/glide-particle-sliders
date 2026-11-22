# GLiDE-9B recurrent sliders

This opt-in package uses the final token at layer 23 on both Euler passes in the pinned text-only GLiDE-9B publisher runtime. The shared `PromptSteering` probe computes the coefficient from each pass's own input. The first edit feeds into the second pass.

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
