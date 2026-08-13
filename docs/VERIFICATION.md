# Verification record

Last verified: 2026-08-01 on Windows 11, Node 24.12, Python 3.11.9,
and an NVIDIA GeForce RTX 4080 Laptop GPU.

## Automated checks

```text
npm.cmd run check  -> contracts, server, web, and desktop passed
npm.cmd test       -> 57 server and 2 web tests passed
npm.cmd run build  -> production server, Vite, and Electron builds passed
python pytest      -> 25 reconstruction tests passed
python py_compile -> worker modules passed
```

The 84 tests cover address ranking, exact U.S. Census matches, OpenStreetMap
footprint selection, KartaView sequencing, Wikimedia/Openverse licensing,
browser result extraction, source policy, encrypted settings, no-cost chat
catalog filtering, JSON model verification, streamed photo batches, property deletion, reconstruction
fallbacks, metric Gaussian registration, joint pose refinement, two-view
artifact cleanup, VLM scene classification, disconnected room grouping,
projection-aware photo/panorama uploads, canonical 1:1 half-sphere rejection and
acceptance, VGGT metric calibration, retained-point optimization,
reciprocal dense-surface refinement, source-ray footprint regularization,
cross-view support reporting, overlap-derived appearance calibration,
scale-count preservation, and evaluation precision/recall.

The manual whole-house pass also verified that 13 uploads become two bounded
jobs of 12 and 4 images with three exact bridge frames, distinct bedrooms do
not collapse through a corridor label, Electron `43.2.0` starts successfully,
and the local operations/settings views render without console errors. The
temporary live UI property was deleted after the visual test.

The machine-wide Python installation also contains unrelated Browser Use,
Gradio, LibreTranslate, and image-editing packages with mutually incompatible
pinned versions, so a global `pip check` is not clean. StructureFirst's worker,
tests, and CUDA acceptance run pass in that environment. The README now uses a
project `.venv` for clean future installs instead of changing those unrelated
global tools.

## Joint four-view room milestone

The four original full-resolution responder JPEGs were rerun after adding
EXIF-normalized VGGT shared cameras, measured SHARP metric calibration, and
global pose-graph refinement. The hash-verified evaluator at
`evaluation/datasets/bedroom-four-view.manifest.json` produced:

- registered frames: `4/4`, up from the previous `2/4`;
- Gaussian count: `4,709,048`;
- joint-camera acceptance: passed;
- median joint/measured rotation disagreement: `2.0349°`;
- pose-graph RMSE: `0.0644981` before and `0.0498477` after;
- reconstruction GPU: NVIDIA GeForce RTX 4080 Laptop GPU;
- peak VGGT VRAM: `6,765.1 MB`;
- source hash verification: `4/4`.

The shared-camera stage does not independently establish room scale or truth.
It is accepted only after a verified SIFT/LoFTR correspondence edge lifted into
SHARP metric geometry calibrates it and the camera rotations agree. The local
responder dataset covers one bedroom and one phone. The separate panorama
fixtures validate projection handling only. The ETH3D tests below add several
calibrated DSLR room components, but multi-floor, varied-device, and
current-condition datasets remain required before operational claims.

## Independent ETH3D room acceptance

Two official ETH3D archives added three full-resolution room-component tests
outside the original bedroom. Archive and per-image SHA-256 hashes are pinned
in `evaluation/datasets/`; source pixels and generated splats remain in ignored
`data/evaluation/`.

The seven-view door/hall component produced:

- registration: `7/7`;
- browser Gaussians after conservative cleanup: `8,222,085`;
- retained sparse 3D point constraints: `4,084`;
- pose-graph RMSE: `0.1008696` to `0.0783907`;
- sparse point RMSE: `0.1198081 m` to `0.1189119 m`;
- reciprocal dense constraints: `25,200`;
- dense surface RMSE: `0.0196839 m` to `0.0196301 m`;
- sparse point RMSE after dense refinement: `0.1189119 m` to
  `0.1187621 m`;
- maximum accepted dense-refinement camera change: `4.5577 mm` and
  `0.04968°`;
- overlap color log-RMSE: `0.0064901` to `0.0052965` from `2,730`
  matched samples;
- source-footprint tangent adjustment: `5,712,671` of `8,243,583`
  pre-cleanup Gaussians, capped at `1.15x`, with the original count preserved;
- cross-view depth support: `10.208037%`;
- splat bytes: `263,106,720`, exactly `8,222,085 x 32`;
- GPU: NVIDIA GeForce RTX 4080 Laptop GPU.

The renderer also applies a `0.12` pixel-space covariance floor and retains the
full three-standard-deviation footprint. This targets small raster pinholes;
it does not fill disoccluded surfaces that no source camera observed.

The ten-image lounge capture set contains two internally connected groups with
zero shared ETH3D point tracks across the groups. StructureFirst correctly kept
them separate instead of inventing one transform. The primary six-view group
produced `7,028,387` Gaussians and improved transform RMSE from `0.447636` to
`0.4005716`, point RMSE from `0.4020168 m` to `0.3927892 m`, and overlap color
log-RMSE from `0.102767` to `0.0515219`.

The independently reconstructed four-view secondary lounge component passed:

- registration: `4/4`;
- Gaussians: `4,692,223`;
- retained 3D point constraints: `2,345`;
- transform RMSE: `0.2170004` to `0.1532958`;
- point RMSE: `0.2681613 m` to `0.2481378 m`;
- overlap color log-RMSE: `0.2090309` to `0.097614` from `2,272`
  matched samples;
- scale artifacts adjusted: `1`, with `0` Gaussians deleted;
- splat bytes: `150,151,136`, exactly `4,692,223 x 32`.

All three reports passed input hashes, registration precision/recall, joint
camera acceptance, pose and retained-point improvement, overlap-color
improvement, full Gaussian-count preservation, and RTX provenance. Visual
inspection loaded all `8,222,369` door Gaussians and all `4,692,223` secondary
lounge Gaussians. The embedded browser again selected the Radeon 610M, so it
rendered the full sets at about `2` and `5` FPS respectively; quality was not
reduced. The door source files omit EXIF orientation and are stored sideways,
which led to the non-destructive Rescue View `Roll 90°` control.

The Windows per-app DirectX preference was set to `GpuPreference=2` for Chrome
and Edge. The repository launcher also starts an isolated Chrome process with
Chromium's high-performance GPU switch. Rescue View now reports live FPS and
the actual WebGL adapter; the live browser confirmation is recorded below
rather than inferred from CUDA reconstruction.

The final API-driven artifact was
`artifact_b47c0378df114e088d5d6290ad2efdbe`. The production path, rather than
the evaluation-only function call, compiled a 150.69 MB splat with `4,709,044`
Gaussians. The ready artifact retained four calibrated camera poses, recorded
CUDA 12.8 and the RTX 4080, reported `1.9375°` median rotation disagreement,
and improved pose RMSE from `0.0650462` to `0.0502429`.

LucidFrame remained a clean, read-only dependency at revision
`f48777638d6275b32bc01039168b5e4bd0f52d13`; all product and integration changes
were made inside the StructureFirst repository.

The in-app browser loaded all `4,709,044` Gaussians with
`activeSplats=4709044` and `detailMode=full`. It rendered at roughly 5.3 FPS on
the Radeon 610M, which is an embedded-browser constraint rather than a
reconstruction fallback. Clicking the Closet room selected calibrated source
camera 4 and moved the view to `-0.0633,-0.0291,-0.0849`; free-flight movement
then changed the position to `-0.1735,-0.0291,0.0820`. The room index contained
only the current bedroom and closet nodes after superseded 2/4 artifact nodes
were removed. The console contained no errors; only the known Three Clock
deprecation and a Direct3D signed/unsigned shader warning were present.

`npm.cmd run rescue:rtx` launched a separate Chrome GPU process with
`--force_high_performance_gpu` and the isolated StructureFirst profile.
`nvidia-smi pmon` reported that exact Chrome GPU PID as a `C+G` workload on the
RTX 4080, confirming that the high-performance demo path uses NVIDIA graphics.

## Full-detail 180° and 360° panorama acceptance

A licensed NOIRLab hotel-room equirectangular panorama was retained at its
original `12,892 x 6,446` resolution and reconstructed through the production
LucidFrame worker in `detail` mode:

- source bytes: `13,491,081`;
- source SHA-256:
  `0e70e2c748a71983601890bcf622206706a23ab359e9563a8bc36d155ab8f96f`;
- projection profile: `full_equirectangular`;
- Gaussian count: `5,801,327`;
- splat bytes: `185,642,464`, exactly `5,801,327 x 32`;
- range streaming: `206 Partial Content`.

The left hemisphere was then encoded losslessly as a `6,446 x 6,446` PNG,
without resizing, to exercise canonical 1:1 180° input. That run produced:

- projection profile: `partial_panorama`;
- measured/observed fraction: `0.49844`;
- pole caps: disabled;
- Gaussian count: `5,767,405`;
- splat bytes: `184,556,960`, exactly `5,767,405 x 32`;
- range streaming: `206 Partial Content`.

This acceptance run exposed and fixed an adapter-boundary bug: LucidFrame's
partial-panorama normalizer supported the 1:1 layout, but its earlier landscape
guard rejected it first. StructureFirst now replaces only that in-process input
guard and delegates normalization and reconstruction to LucidFrame's unchanged
`reconstruct_sharp360` function. LucidFrame remains clean at revision
`f48777638d6275b32bc01039168b5e4bd0f52d13`.

## Live address-to-scene test

The production server and worker were restarted from the final build. A new
property was submitted with:

```text
350 Fifth Avenue, New York, NY 10118
```

The complete live result was:

- U.S. Census Geocoder resolved `350 5th Ave, New York, NY, 10118`;
- Overpass selected OpenStreetMap way `34633854`, the Empire State Building,
  with 102 levels and 443.2 m height;
- 50 evidence records were retained, including 39 image candidates;
- six modification-safe images were downloaded with attribution and hashes;
- 26 unknown-rights results remained link-only metadata;
- the three KartaView frames shared one measured sequence and overlap set;
- LucidFrame registered all 3/3 captures with confidence `0.6493`;
- the final scene contains 3,536,159 Gaussians and 113,157,088 bytes;
- a byte-range request returned `206 Partial Content`, 32 requested bytes, and
  the correct total scene size.

The saved case is left in the local workspace as one ready property. The earlier
duplicate verification case and its files were deleted.

## Earlier address-to-scene LucidFrame boundary

That earlier address-to-scene run used the local LucidFrame revision
`a7a1e2840a9005dba7a954649fd71b47190f297e` and the official Apple SHARP
checkpoint:

```text
sharp_2572gikvuh.pt
2,809,738,232 bytes
SHA-256 94211a75198c47f61fca7d739ba08a215418d8d398d48fddf023baccc24f073d
```

Full-resolution frame outputs were 1,178,223, 1,178,894, and 1,179,042
Gaussians. The compiler's 32-byte record check passed exactly:

```text
3,536,159 x 32 = 113,157,088 bytes
```

The browser loaded the 113.16 MB scene with Spark, set the canvas ready marker,
showed 3/3 captures and 65% registration, and displayed no scene error. Drag-look
and keyboard movement changed the rendered frame, and Reset restored the view.
There were no browser errors; Spark's Three.js dependency emitted one non-fatal
`THREE.Clock` deprecation warning.

The final run was slow because the laptop GPU remained power-limited near 50 W:
the three SHARP passes took 278.6, 699.2, and 678.8 seconds. This is a real local
performance constraint, not a stalled or simulated job.

## Live settings test

The saved NVIDIA NIM key returned 119 raw entries spanning chat, embeddings,
retrieval, safety, detection, and other API families. StructureFirst reduced
that mixed list to 36 NVIDIA-documented chat or image-understanding prototype
candidates; five were marked image-capable. The provider dropdown changed to
Groq and back, the native model dropdown changed to `openai/gpt-oss-20b` and
back, and `meta/llama-3.2-11b-vision-instruct` passed the real JSON-format test.
The final browser **Verify & save** run completed in 431 ms, left NVIDIA enabled
with vision support, and displayed a stable verified-and-active confirmation.
No browser errors were reported.

Groq, Cerebras, and OpenRouter were not live-called because no keys for those
accounts were configured. Their endpoints, filters, dropdown contracts, and
success/failure behavior are covered with mocked provider responses in the
server test suite. The UI links directly to each provider's current access or
limit documentation.

## Product boundary

The live scene is exterior street evidence. StructureFirst does not claim that
online exterior photos reveal unseen rooms. Interior geometry requires
overlapping responder, owner-authorized, or otherwise reusable interior images
or authoritative plans. Failed multi-photo registration falls back to the exact
first source image and records the reason instead of merging unrelated geometry.

The Vite build still reports large Three/Spark chunks as a performance warning;
it is not a compile or runtime failure. This remains a prototype rather than a
certified dispatch or life-safety system.

## Anonymous mixed-room test

Seven unordered uploads contained four photos of one bedroom, two unrelated
hotel rooms, and one cat. No filename or known-input rule was used. The live
upload endpoint produced this result:

- geometric core: three bedroom photos;
- recognized candidate: the fourth bedroom angle from capture continuity and
  DINOv2 affinity;
- rejected before SHARP: both hotel rooms and the cat;
- registration: only cross-image SIFT/LoFTR correspondences lifted into SHARP
  metric geometry may assign a camera transform;
- the fourth bedroom angle is recorded as `same_scene_unregistered`, because
  SuperPoint + LightGlue found only seven matches and zero fundamental-matrix
  inliers against its best candidate view;
- the previous point-cloud-shape placement was removed. Its transform disagreed
  with a joint VGGT pose diagnostic by `59.19 degrees`, confirming that its
  apparent ICP score did not establish camera correspondence;
- final artifact: `3,530,992` Gaussians and `112,991,744` bytes;
- live case result: `3/7` inputs connected, one same-room view left unplaced,
  and both unrelated rooms plus the cat excluded.

The source-color repair path previously reopened portrait JPEGs without applying
EXIF orientation. It therefore sampled unrelated pixels and could turn dark
Gaussians white. Applying the same EXIF transpose used by SHARP reduced repaired
pixels from `19,616`, `59,141`, `138,584`, and `32,473` to `947`, `2,147`,
`6,923`, and `363` for the four bedroom photographs.

The in-app browser used an AMD Radeon 610M rather than the laptop RTX 4080. The
viewer kept all `3,530,992` Gaussians active in `full` detail while moving the
camera from `z=0.0000` to `z=0.6000`; there were no runtime errors. It rendered
at about `6.1 FPS` on that integrated adapter. Python/CUDA separately detected
the RTX 4080, so the remaining viewer performance limit is Windows' GPU choice
for the embedded Chromium process rather than reconstruction or an LoD cap.

## Current four-view cleanup and automation test

The four original responder JPEGs were uploaded together through the live API.
NVIDIA NIM classified all four without a filename rule: three bedroom views and
one closet view. All floor labels became `unknown`, because no pixels or summary
supported a basement, grade-level, upper-floor, or attic claim. The exact case
address is no longer included in the VLM prompt; placeholders such as
`possible`, `unknown`, and `null` are normalized before storage. NVIDIA's legacy
vision model ignored JSON-schema mode on some calls, so the validated fallback
also parses its labeled prose without allowing it to set address truth.

The final fresh-worker reconstruction was
`artifact_d6d13954a9b3451380f6d65598a08096`:

- 2/4 images registered with `0.9724` transform confidence;
- the accepted pair contained 274/280 metric inliers with `0.02072 m` RMSE;
- the two remaining bed/window angles had no verified SIFT or LoFTR geometry and
  were left unplaced instead of receiving guessed rotations;
- two-view depth consistency checked 37,010 cross-view Gaussians and removed
  946 front-surface contradictions;
- the cleaned artifact contains 2,353,576 Gaussians;
- Rescue View loaded all 2,353,576 with `detailScale=1.000` and LoD disabled;
- free-flight moved the camera and Reset returned it to `0,0,0`;
- the embedded browser rendered about 10.1 FPS on the Radeon 610M and reported
  no runtime errors. Windows still must assign the browser executable to the
  RTX 4080 for higher viewer FPS.

The strict live browser query for `221B Baker Street, London, UK` returned only
results whose title or URL contained `221B` plus the submitted street terms.
Previously observed unrelated accounting pages no longer pass the filter.
Zillow and Redfin are searched only through result metadata; StructureFirst
does not visit their pages or copy listing images.

## Live agentic-browser test

The enabled NVIDIA vision model drove a normal headed Chrome session against
the existing `221B Baker Street, London, UK` test case. The agent completed its
full 20-step budget after the response adapter normalized both `name`-based and
`type`-based action dialects. It did not navigate to Zillow, Redfin, Google
imagery, login pages, or social sources; those destinations are independently
blocked in the prompt, click handler, navigation-request interceptor, and
import policy.

The run found no permitted, exact-address, full-size image that also carried an
established reusable license, so it imported zero bytes. This is the correct
result: the agent cannot turn search visibility into media rights or safe
multi-view geometry. KartaView, Wikimedia, Openverse, responder uploads, and
future licensed property feeds remain the byte-producing paths. Unknown-rights
property pages are preserved only as source links.
