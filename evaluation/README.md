# StructureFirst reconstruction evaluation

This directory contains reproducible, non-image manifests and result summaries.
Operator-owned source pixels remain under the ignored `data/` directory.

Run the real local four-view acceptance case:

```powershell
python services/reconstruction/evaluate.py `
  --manifest evaluation/datasets/bedroom-four-view.manifest.json `
  --registration data/evaluation/bedroom-four-view-joint-v2/registration.json `
  --verify-inputs `
  --output evaluation/results/bedroom-four-view-joint-v2.json
```

The evaluator measures registered-frame precision/recall, room completeness,
joint-camera acceptance, rotation agreement, sparse pose-graph improvement,
reciprocal dense-surface improvement, input hashes, Gaussian-count
preservation, cross-view support, and GPU provenance. A passing report is an
engineering milestone, not a claim that unseen geometry or a tactical route is
verified.

Full-detail projection acceptance results are also tracked:

- `results/panorama-360-detail.json` records the original licensed 2:1
  full-sphere panorama, source hash, Gaussian count, and streaming checks.
- `results/panorama-180-detail.json` records the lossless 1:1 hemisphere
  fixture, its partial-panorama observed fraction, Gaussian count, and streaming
  checks.

The source pixels and generated splats stay under ignored `data/`; the tracked
JSON results contain enough provenance and sizing checks to audit the run
without redistributing the large artifacts.

Two independent ETH3D indoor scenes expand the evaluation beyond the local
bedroom capture:

- `eth3d-lounge-ten-view.manifest.json` covers a ten-image capture set with two
  disconnected room components. The primary component must remain separate.
- `eth3d-lounge-secondary-four-view.manifest.json` reconstructs the verified
  four-view secondary component as its own full-resolution room splat.
- `eth3d-door-seven-view.manifest.json` covers seven portrait-oriented DSLR
  views with fine architectural detail.

Run either dataset through the same full-resolution production reconstruction
code and compile a browser-ready splat:

```powershell
python services/reconstruction/run_dataset.py `
  --manifest evaluation/datasets/eth3d-lounge-ten-view.manifest.json `
  --output data/evaluation/results/eth3d-lounge
```

The ETH3D source archives and extracted pixels remain under ignored `data/`.
Their archive and per-image SHA-256 digests are pinned in the manifests.
Tracked summaries for the verified RTX runs are in `results/eth3d-*.json`;
the large browser splats remain under ignored `data/evaluation/results/`.
