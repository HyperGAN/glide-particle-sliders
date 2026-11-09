# Current GLiDE learning-rate guidance

The completed saved RAW512 comparison yields four tied full recipes:
**arm04/05/06/07**, all G0.001/batch8/horizon512/seed7, with D0.003 or
0.018000000000000002 and bank0.003 or0.03. There is no unique full recipe,
D/bank choice or default change. See [exact options and all13 first counts](docs/finished-output-learning-rates.md)
and [portable metadata](docs/finished-output-rate.json).

Select only equal-case binary concise-choice ON−exact matched OFF on the
original sixteen main ordered DEV rows (eight scenarios, two orders each).
Retain negatives, zeros and exact ties; failures are missing evidence, not
zero. Exclude correctness-control/unrelated rows, probability/confidence,
SSE, preservation, beauty and candidate index from selection and tiebreaks.
Use RAW weights with explicit `ema_decay=None`; refuse weight EMA.

Reuse completed training, source-bound RAW/FULL checkpoints, native outputs
and first observations. No rerun, new confirmation or worse-arm completion is
required. Historical SSE-selected batch32 and fresh104 results retain their
original separate scopes. The original experiment/HPO scripts still implement
historical SSE/index selection; they are not current ranking instructions.
The existing normalized configuration factory accepts the exact scalar tuples;
the convenience training script hardcodes one multiplier pair and does not
expose every tied recipe. The existing frozen core0.13.0/PG0.9.5 native bundle
is now Root-adopted, and twelve saved fresh-anchor decision previews are
published, separately from this accepted arithmetic decision. Ordinary Generate
remains unavailable and published weights remain zero. See the portable status
evidence; no proof, search, output or score was repeated.
