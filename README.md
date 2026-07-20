# StructureFirst

StructureFirst turns connected photographs of a property into a navigable
**Rescue View** for emergency teams. The current MVP resolves an address,
searches for reusable images, verifies which photographs actually overlap, and
uses local LucidFrame reconstruction to build the 3D view.

The product is intentionally simple:

1. Enter an address.
2. StructureFirst searches automatically.
3. Openly reusable photos are imported with their source and license.
4. Only visually connected photographs enter the multi-image reconstruction.
5. Open Rescue View and navigate the resulting Gaussian scene.
6. Add overlapping responder photos at any time for a better result.

## What works

- Address resolution and OpenStreetMap building footprints
- Keyless KartaView, Wikimedia Commons, Openverse, and Chrome/Edge search
- Optional Brave Search for a structured image index
- Automatic import of public-domain, CC0, CC BY, and CC BY-SA photos
- LucidFrame SHARP reconstruction with SIFT, indoor LoFTR, and
  correspondence-verified metric smart connect
- Joint Sim(3) pose-graph refinement when three or more verified views create
  loop-closing constraints, plus cross-view depth artifact cleanup
- Automatic VLM scene, room, doorway, and evidence-floor classification with
  unsupported floor claims forced back to unknown
- Separate reconstruction jobs and floor/room entries for disconnected photo
  groups that contain their own verified overlap
- Strict street-number/address-term filtering before browser results can become
  property evidence; Zillow and Redfin results remain link-only
- Byte-for-byte SHA-256 verification from the saved photo into LucidFrame
- Multiple JPEG, PNG, or WebP uploads: up to 50 files and 1 GB total per batch
- A full-detail navigable Rescue View with mouse, keyboard, wheel, on-screen
  controls, and visible GPU reporting
- Saved-property deletion from both the home page and property page
- One AI connection area for Groq, Cerebras, OpenRouter, or NVIDIA NIM
- Server-side encrypted API keys and provider connection tests
- Light and dark themes

AI is optional. Address lookup, maps, keyless online image search, uploads, and
local reconstruction do not require an AI provider key.

## Run

Requirements: Node.js 24+, Python 3.11+, an NVIDIA CUDA GPU, and the local
`LucidFrame` folder beside this repository.

```powershell
npm.cmd install
Copy-Item .env.example .env
python -m pip install -r services/reconstruction/requirements.txt
npm.cmd run dev:full
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173).

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

Automatic online photo search needs no key. A Brave Search key is optional and
can be added under **Optional Brave Search connection**.

Important `.env` values:

| Variable                            | Use                                      |
| ----------------------------------- | ---------------------------------------- |
| `LUCIDFRAME_ROOT`                   | Local LucidFrame repository path         |
| `STRUCTUREFIRST_ACCESS_KEY`         | Required when exposed beyond localhost   |
| `STRUCTUREFIRST_DATA_DIR`           | Optional database and property-file path |
| `STRUCTUREFIRST_BROWSER_EXECUTABLE` | Optional Chrome or Edge path             |

## Better multi-photo results

Upload adjacent views of the same space with roughly 60-80% visual overlap.
StructureFirst sends up to 12 photos into one smart-connect job and keeps every
uploaded photo in the property. Photos that cannot be matched are not silently
treated as connected geometry. If disconnected photos form another internally
verified overlap group, the backend queues that group as a separate room scene;
isolated images are not reconstructed as a room.

The selector first forms a geometrically verified overlap core. DINOv2 scene
descriptors and EXIF capture continuity may nominate another angle from the
same room, but recognition never places that image. It enters the splat only
when cross-image correspondences produce a stable transform in SHARP's metric
geometry. A same-room image without enough overlap remains explicitly
unregistered instead of being snapped onto similar walls or furniture.
Unrelated room photos, objects, and animals remain excluded.

Rescue View represents only surfaces visible in the source images. It does not
infer a collision-safe route or expose an unseen interior. To reconstruct an
interior, capture each room and doorway with continuous overlap so the images
form one connected visual path.

Rescue View displays the WebGL adapter it actually received. On a hybrid-GPU
laptop, set the executable hosting StructureFirst under Windows **Settings >
System > Display > Graphics** to **High performance**, restart it, and confirm
that the badge names the discrete NVIDIA GPU. The web page requests a
high-performance adapter but cannot override Windows' per-process assignment.

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
```

## Local browser agent (optional)

StructureFirst can drive a **visible** local Chrome or Edge with your
configured AI provider acting as the operator. When enabled under
**Settings > Local browser agent**, pressing "Search" launches a headed
browser, sends each step's screenshot plus a short list of interactive
elements to the model, and executes one JSON action per turn (`goto`,
`type`, `click`, `scroll`, `collect_image`, `done`, ...).

Requirements: a vision-capable model on the active AI provider, Chrome or
Edge installed locally (or `STRUCTUREFIRST_BROWSER_EXECUTABLE` pointing at
one), and a step budget between 4 and 60.

Every image the agent collects is downloaded into the case directory and
recorded with:

- `rights: "restricted"` and `redistributable: false`
- provenance tags `automated-discovery`, `browser-agent`, `local-only`,
  `not-redistributable`
- the original page URL kept as `originUrl` and the direct image URL as
  `downloadUrl`
- a SHA-256 hash and the exact byte size

These files stay on the machine that ran the agent, feed only the local
LucidFrame reconstruction, and must not be exported or shared. If reuse is
needed, replace the agent-collected image with an operator upload or a
result from an open-license source (KartaView, Wikimedia, Openverse).

## Source and model boundaries

- Unknown-rights web and real-estate results remain links to the original site
  by default. StructureFirst's automatic collectors (KartaView, Wikimedia,
  Openverse, keyless Bing) do not crawl Zillow or Redfin or copy their listing
  media.
- The optional local browser agent may download restricted images for
  local-only reconstruction. Those files are flagged `rights: "restricted"`,
  `redistributable: false`, and stay on the machine that captured them. Export
  and sharing paths must honor those flags.
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
