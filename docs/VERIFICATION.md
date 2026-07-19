# Verification record

Last verified: 2026-07-19 on Windows 11, Node 24.12, Python 3.11.9,
and an NVIDIA GeForce RTX 4080 Laptop GPU.

## Automated checks

```text
npm.cmd run check  -> contracts, server, and web passed
npm.cmd test       -> 39 server and 2 web tests passed
npm.cmd run build  -> production server and Vite build passed
python pytest      -> 13 reconstruction tests passed
python py_compile -> worker modules passed
```

The 54 tests cover address ranking, exact U.S. Census matches, OpenStreetMap
footprint selection, KartaView sequencing, Wikimedia/Openverse licensing,
browser result extraction, source policy, encrypted settings, no-cost chat
catalog filtering, JSON model verification, streamed photo batches, property deletion, reconstruction
fallbacks, metric Gaussian registration, joint pose refinement, two-view
artifact cleanup, VLM scene classification, and disconnected room grouping.

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

## Exact LucidFrame boundary

The worker used the local LucidFrame revision
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
