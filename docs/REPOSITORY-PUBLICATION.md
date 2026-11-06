# Repository location and publication status

The existing project is:

`/ml2/codex-storage/future-workspaces/2026-11-06/task/hypergan/glide-particle-sliders`

Local duplicate checks under `/ml2/hypergan`, `/ml2/particlesliders-working` and this task's `hypergan/` directory found this one GLiDE project. No second checkout was created in the canonical `/ml2/hypergan` tree, which is outside this task's writable roots.

The authenticated CLI at `/home/mikkel/.local/bin/gh` checked public/private access and verified that the requested repository did not already exist. After explicit owner approval of public visibility, [HyperGAN/glide-particle-sliders](https://github.com/HyperGAN/glide-particle-sliders) was created and verified as PUBLIC. The initial denied attempt had not executed; the approved creation was retried once.

The code and weight-free reports are published on `codex/glide-particle-sliders` for draft-PR review. The `main` bootstrap contains only `.gitignore`; no implementation merge or deployment is performed. Pretrained and trained weights, caches, environments and local checkpoints remain excluded. No credentials or access permissions were changed. The local [draft PR description](DRAFT-PR.md) records the implementation and validation scope.

The older [repository-publication-status-v2.json](../reports/repository-publication-status-v2.json) records the earlier PATH-only/unauthenticated limitation and remains historical. Local publication, retry and CI receipts are retained under `artifacts/publication-v3/` and excluded from source publication. CPU CI checks source-only synthetic mechanics and provenance; it does not repeat the pretrained encoder or preference-quality experiments.
