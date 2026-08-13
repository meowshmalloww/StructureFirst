# StructureFirst

StructureFirst turns connected photographs of a property into a navigable
**Rescue View** for emergency teams. The current MVP accepts a manual capture
set, verifies which photographs actually overlap, organizes connected views
into rooms and floors, and uses local LucidFrame reconstruction to build the 3D
view.

The product is intentionally simple:

1. Enter an address.
2. Add one ordered capture walk through the property.
3. Only visually connected photographs enter reconstruction.
4. Large capture sets become overlapping GPU-safe jobs with shared doorway,
   corridor, or stair bridge frames.
5. Open Rescue View and move between verified room and floor segments.

## What works

- Address resolution and OpenStreetMap building footprints
- Manual whole-house capture sets up to 50 files and 1 GB
- Natural filename ordering plus geometry-verified view matching
- LucidFrame SHARP reconstruction with SIFT, indoor LoFTR, and
  correspondence-verified metric smart connect
- Capture-set coverage gating before VGGT or SHARP: one verified geometric
  core must contain at least half the supplied photos, so a small cluster of
  repeated fixtures cannot masquerade as the requested room or building
- EXIF-normalized VGGT shared-camera inference calibrated against measured
  LucidFrame/SHARP constraints; the joint estimate is rejected when its camera
  rotation disagrees with measured geometry
- Correspondence-aware Sim(3) pose-graph refinement when three or more verified
  views create loop-closing constraints; shared SHARP 3D inlier points remain
  in the joint solve instead of being reduced to pair-transform summaries
- Reciprocal dense-surface camera refinement after the verified SIFT/LoFTR
  pose graph; it accepts only bounded camera changes that improve measured
  depth agreement without degrading the original feature constraints
- Source-ray footprint regularization that closes small observed pinholes by
  growing only undersized tangent axes up to 15%, while preserving resolution,
  Gaussian count, and the thin surface axis
- Verified-overlap RGB calibration, cross-view depth artifact cleanup,
  cross-view support measurement, and conservative single-axis needle
  regularization with no Gaussian-count loss
- Automatic VLM scene, room, doorway, and evidence-floor classification with
  unsupported floor claims forced back to unknown
- GPU-safe reconstruction windows of at most 12 frames with three exact bridge
  frames shared between adjacent windows
- Separate room nodes for same-type rooms unless measured overlap joins them;
  shared bridge frames link reconstruction segments
- Byte-for-byte SHA-256 verification from the saved photo into LucidFrame
- Multiple JPEG, PNG, or WebP uploads: up to 50 files and 1 GB total per batch
- One manual whole-house capture workflow; plans, room photos, exterior views,
  and unrelated media are separated before reconstruction
- A full-detail navigable Rescue View with mouse, keyboard, wheel, on-screen
  controls, a non-overlapping room scene navigator, an always-visible
  multi-view camera layout with calibrated source bookmarks, live FPS, visible
  GPU reporting, measured
  cross-view support, and an outside-capture warning
- A separate, pannable and zoomable House Map that preserves the supplied plan
  image and marks plan-linked scenes in green, position-unverified matches in
  amber, and unavailable rooms in gray
- Local YOLO26 Nano detection starts automatically over the live Rescue View,
  with real 2D object boxes and clearly unverified tactical tags; rendered
  frames stay on the local machine and the overlay can be switched off. The
  official pretrained model is AGPL-3.0 (or requires an Ultralytics Enterprise
  License), and its boxes are not persistent 3D hazard geometry
- Artifact-local metric room/floor coordinates and conservative candidate
  doorway, corridor, or stair edges
- Photo-only observed-level separation from calibrated camera up vectors and
  metric elevation; it labels `Observed level 0/1/...` without claiming a
  ground floor or basement that the imagery does not verify
- Hash-verified real-room reconstruction evaluators under `evaluation/`,
  including a phone bedroom and three ETH3D room components
- Saved-property deletion from both the home page and property page
- One AI connection area for Groq, Cerebras, OpenRouter, or NVIDIA NIM
- Server-side encrypted API keys and provider connection tests
- Light and dark themes
- A locked-down Electron/TypeScript desktop shell that requests the
  high-performance GPU before Chromium starts

AI is optional. Address lookup, maps, uploads, geometric matching, and local
reconstruction do not require an AI provider key. A vision-capable provider
adds room, floor, doorway, corridor, stair, and outlier classification.

Floorplans are optional. Without one, room and floor placement comes only from
the measured camera path: keep 60-80% overlap, include 2-3 bridge views through
every doorway, and photograph stairs continuously. Unseen rooms and missing
connections remain unknown rather than being inferred from labels.

## Run

Requirements: Node.js 24+, Python 3.11+, an NVIDIA CUDA GPU, and the local
`LucidFrame` folder beside this repository.

```powershell
npm.cmd install
Copy-Item .env.example .env
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r services/reconstruction/requirements.txt
npm.cmd run desktop:dev
```

The StructureFirst desktop window opens after the worker, server, and renderer
are ready. Use `npm.cmd run dev:full` only when a browser tab is useful for
development.

On this hybrid-GPU Windows laptop, apply the per-browser high-performance
preference once and open an isolated RTX Rescue View:

```powershell
npm.cmd run rescue:rtx:configure
npm.cmd run rescue:rtx
```

For the production build:

```powershell
npm.cmd run build
npm.cmd run worker
```

Then, in a second terminal:

```powershell
npm.cmd run start -w @structurefirst/server
```

Open [http://127.0.0.1:8787](http://127.0.0.1:8787).

## Settings

Open **Settings** to choose one optional AI provider. After a key is entered,
StructureFirst loads only no-cost choices that fit its chat/JSON workflow:
Groq Free Plan chat models, Cerebras public chat models, OpenRouter zero-price
text models, or NVIDIA's documented developer-prototype chat and multimodal
models. **Verify & save** sends one short compatibility request before the
connection is activated. Provider quotas and availability can still change.
Keys are encrypted in the local data directory and are never returned to the
browser.

Online listing discovery is paused and hidden from the normal workflow. Its
source-policy implementation remains in the repository for later work.

Important `.env` values:

| Variable                            | Use                                      |
| ----------------------------------- | ---------------------------------------- |
| `LUCIDFRAME_ROOT`                   | Local LucidFrame repository path         |
| `STRUCTUREFIRST_ACCESS_KEY`         | Required when exposed beyond localhost   |
| `STRUCTUREFIRST_DATA_DIR`           | Optional database and property-file path |
| `STRUCTUREFIRST_BROWSER_EXECUTABLE` | Optional Chrome or Edge path             |

## Better multi-photo results

Choose **Whole house** and upload adjacent views in capture order with roughly
60-80% visual overlap. Choose **180°** only for one 1:1 half-sphere
equirectangular image, or **360°** for one 2:1 full-sphere equirectangular
image. StructureFirst validates those layouts and preserves the original
uploaded bytes; panorama reprojection and SHARP's normal model preprocessing
occur inside LucidFrame.

For canonical 1:1 half-sphere input, StructureFirst adapts only LucidFrame's
in-process landscape guard so LucidFrame's existing partial-panorama normalizer
can run. The LucidFrame repository, `reconstruct_sharp360` implementation, and
official Apple SHARP checkpoint are not modified.

StructureFirst keeps every uploaded photo. It naturally sorts filenames, then
divides a large continuous capture walk into jobs of at most 12 frames. Adjacent
jobs share three exact source frames so later house-level alignment has measured
bridge evidence. Photos that cannot be matched are never silently treated as
connected geometry. A disconnected subset with its own verified overlap is
queued as a separate room scene; isolated images are not reconstructed as a
room.

The selector first forms a geometrically verified overlap core. DINOv2 scene
descriptors and EXIF capture continuity may nominate another angle from the
same room, but recognition never places that image. VGGT then estimates all
nominated cameras jointly from EXIF-normalized views. That shared solution can
place a low-texture angle only after at least one verified image/SHARP metric
edge calibrates the room scale and the joint camera rotations agree with the
measured edge. Conflicting shared-camera estimates are rejected. Unrelated room
photos, objects, and animals remain excluded.

Rescue View represents only surfaces visible in the source images. It does not
infer a collision-safe route or expose an unseen interior. To reconstruct an
interior, capture each room and doorway with continuous overlap so the images
form one connected visual path.

The pinhole pass is not generative inpainting: it fills only sub-pixel gaps
inside a source camera's measured footprint. Moving behind a bed, shelf, wall,
or other occluder can still expose genuinely unseen space. Rescue View reports
the fraction supported by another measured view and warns after the free camera
moves far beyond the calibrated capture envelope.

Rescue View displays the WebGL adapter it actually received. On a hybrid-GPU
laptop, `npm.cmd run rescue:rtx:configure` sets Windows' per-app preference for
Chrome and Edge, while `npm.cmd run rescue:rtx` starts a separate Chrome profile
with Chromium's high-performance GPU switch. Confirm that the in-view badge
names the discrete NVIDIA GPU. The web page requests high performance but
cannot override Windows' per-process adapter assignment by itself.

## Image-only scope

The current product accepts JPEG, PNG, and WebP captures only. Video upload and
frame extraction are intentionally disabled so the multi-image registration,
room grouping, provenance, and Rescue View behavior can be validated first.

## Checks

```powershell
npm.cmd run check
npm.cmd run test
npm.cmd run build
python -m pytest services/reconstruction
python services/reconstruction/evaluate.py --manifest evaluation/datasets/bedroom-four-view.manifest.json --registration data/evaluation/bedroom-four-view-joint-v2/registration.json --verify-inputs
python services/reconstruction/run_dataset.py --manifest evaluation/datasets/eth3d-door-seven-view.manifest.json --output data/evaluation/results/eth3d-door
```

## Paused online-discovery module

StructureFirst retains the source-policy and visible-browser discovery code,
but it is not started by the address pipeline and its controls are hidden. The
manual capture workflow is the current product milestone.

Requirements: a vision-capable model on the active AI provider, Chrome or
Edge installed locally (or `STRUCTUREFIRST_BROWSER_EXECUTABLE` pointing at
one), and a step budget between 4 and 60.

Automatic listing-site collection remains paused while whole-house manual
capture and reconstruction are evaluated.

The agent starts from several exact-address searches, clicks permitted source
pages, and accepts a candidate only when the full-size image and exact address
are both visible on that page. It cannot trust a model-supplied URL by itself.
Unknown-rights candidates are retained as metadata links only. Image bytes are
downloaded and passed to the VLM/LucidFrame pipeline only when source policy
establishes an open or public-domain license; imported bytes retain their
origin URL, license state, SHA-256 hash, and exact byte size.

## Source and model boundaries

- Unknown-rights web and real-estate results remain links to the original site
  by default. StructureFirst's automatic collectors (KartaView, Wikimedia,
  Openverse, keyless Bing) do not crawl Zillow or Redfin or copy their listing
  media.
- The local browser agent does not copy unknown-rights or restricted listing
  images. Those discoveries remain address-check links and never enter
  reconstruction. Export and sharing paths must honor those flags.
- Only modification-safe public-domain and Creative Commons images are
  eligible for redistribution or reuse outside this machine. Automatic
  reconstruction additionally requires exact address text support or an
  operator-supplied case assignment.
- StructureFirst passes the unchanged saved image file to LucidFrame. Apple
  SHARP performs its required decode and 1536 x 1536 model preprocessing inside
  the reconstruction engine.
- Every reconstruction manifest records the input hashes, exact LucidFrame
  backend hashes, official Apple checkpoint hash, selected entry point, and
  whether multi-photo registration succeeded or an exact single-photo fallback
  was used, including the fallback reason.
- Indexed license metadata should still be verified at the original item page;
  StructureFirst keeps that link and attribution attached.
- A single-photo Gaussian predicts nearby appearance; it does not reveal unseen
  rooms or prove that conditions are current.
- Apple SHARP weights are restricted to noncommercial research use.
- This is a prototype, not a certified dispatch or life-safety system.

See the [operator guide](docs/OPERATOR_GUIDE.md),
[system design](docs/SYSTEM_DESIGN.md), and
[verification notes](docs/VERIFICATION.md).
