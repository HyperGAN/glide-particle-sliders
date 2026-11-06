# Repository location and publication status

The existing project is:

`/ml2/codex-storage/future-workspaces/2026-11-06/task/hypergan/glide-particle-sliders`

Local duplicate checks under `/ml2/hypergan`, `/ml2/particlesliders-working` and this task's `hypergan/` directory found this one GLiDE project. No second checkout was created in the canonical `/ml2/hypergan` tree, which is outside this task's writable roots.

The authenticated CLI at `/home/mikkel/.local/bin/gh` verified that `HyperGAN/glide-particle-sliders` does not exist within the account's public/private access. The account has active organization admin membership, and organization policy allows public and private creation. Existing model-specific slider repositories are public; this convention alone did not satisfy automatic approval review's requirement for explicit visibility authorization.

The code and weight-free reports are available locally for review. The exact remaining decision is whether the requested repository should be public or private. A public creation attempt was denied before execution by automatic approval review; a subsequent authenticated check confirmed no remote repository was created. No push, PR, deployment or merge occurred. No model weights were published, and no credentials or permissions were changed. The local [draft PR description](DRAFT-PR.md) is review text only.

The older [repository-publication-status-v2.json](../reports/repository-publication-status-v2.json) records the earlier PATH-only/unauthenticated limitation and remains historical. The current local publication readiness and denial record is retained under `artifacts/publication-v3/`; it is excluded from source publication. CPU CI is configured, but a genuine GitHub run cannot be verified until remote creation and the draft PR are authorized.
