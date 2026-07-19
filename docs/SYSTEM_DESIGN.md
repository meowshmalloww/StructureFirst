# System design

StructureFirst is a persistent property workspace, not a chat session. The
operator-facing product keeps the address, source photos, map, reconstruction,
and preparation history together. The internal case schema retains provenance
and confidence fields without exposing that implementation complexity in the UI.

## Data flow

1. The Fastify server uses the U.S. Census structure-address service when
   applicable, otherwise ranks Nominatim candidates against the submitted
   address.
2. Overpass candidates are ranked by exact address tags, polygon containment,
   and distance to the footprint rather than the first geometry vertex.
3. The discovery coordinator searches KartaView, Wikimedia Commons, Openverse,
   Bing through a real Chrome/Edge session, and optional Brave Search.
4. Browser results must contain the submitted street number and address terms.
   Zillow and Redfin matches stay link-only and are never crawled for media.
5. Modification-safe public-domain and Creative Commons images are imported
   automatically. Only exact address-text matches or operator uploads are
   eligible for automatic reconstruction.
6. A configured vision model classifies permitted local images. The server,
   not the model, determines address match and rejects unsupported floor claims.
7. Responder images are stored inside the property directory with a
   generated filename, byte count, MIME type, and SHA-256 digest.
8. The local Python worker automatically reconstructs eligible local images
   with LucidFrame.
9. AI providers may create source-linked hazard candidates. They cannot create
   verified geometry, a confirmed hazard, or a route by themselves.
10. The browser shows one map/3D canvas, real preparation progress, and the photo
    sources. Operational review data remains an internal safety boundary.

Pipeline stages are persisted and streamed to the browser. A provider outage is
reported as limited or failed; it is never rewritten as success.

## Confidence and provenance

Every material claim carries:

- a score from 0 to 1;
- a band: verified, reconstructed, estimated, or unknown;
- a state: observed, derived, inferred, or unknown;
- a plain-language rationale;
- the number of supporting sources.

These fields describe the quality of knowledge, not the physical level of risk.
An intelligence gap can be verified while the building geometry remains unknown.

## Discovery and source policy

| Source                         | Local bytes           | Case metadata | Redistribution                 |
| ------------------------------ | --------------------- | ------------- | ------------------------------ |
| Responder upload               | Allowed               | Allowed       | Off until rights are confirmed |
| KartaView CC BY-SA street view | Automatic import      | Allowed       | Attribution/share-alike apply  |
| PDM, CC0, CC BY, or CC BY-SA   | Automatic import      | Allowed       | Only when item terms permit    |
| Rights unknown                 | No automatic download | Allowed       | No                             |
| Restricted social source       | No automatic download | Allowed       | No                             |
| Zillow or Redfin listing       | No crawl or download  | Link only     | No                             |
| Google imagery                 | Prohibited            | Link only     | No                             |
| YouTube media                  | Prohibited            | Link only     | No                             |

The importer blocks private-network targets, checks redirects and DNS results,
limits downloads to 20 MB, verifies image signatures, and retains attribution.
Indexed license metadata and address relevance still require item-level review.

## AI providers

The same OpenAI-compatible adapter supports Groq, Cerebras, OpenRouter, and
NVIDIA NIM. Settings presents one connection form and disables previously active
providers when the operator chooses another one.

Catalogs are filtered before they reach the browser. Groq is restricted to its
documented Free Plan general-chat set, Cerebras uses its public chat catalog,
OpenRouter requires zero prompt/completion pricing and text-only output, and
NVIDIA's mixed `/v1/models` list is intersected with its documented chat and
multimodal endpoints. The chosen model must pass a real JSON-format chat request
before the UI saves and activates it.

API keys are encrypted with AES-256-GCM in the local data directory. Plaintext
keys are never returned to the browser. Provider endpoints are allowlisted;
NVIDIA additionally permits an HTTPS or localhost self-hosted NIM endpoint.

Image classification uses structured JSON when the provider supports it and a
validated text-normalization fallback for legacy vision models. Exact case
addresses are withheld from the vision prompt so a model cannot copy them and
pretend they were visible. New findings are stored as pending candidates for
human review.

## LucidFrame boundary

The worker imports LucidFrame's `reconstruct_sharp`, `reconstruct_sharp360`, and
`compile_splat` functions. Jobs are serialized on one GPU and publish a splat
only after confirming:

```text
file size = Gaussian count x 32 bytes
```

For 2 to 12 unordered photos, smart connect:

1. tests every pair with SIFT and an indoor LoFTR matcher plus robust geometry;
2. forms the strongest verified overlap core before spending GPU time;
3. lets DINOv2 place affinity and EXIF capture continuity nominate likely
   same-room angles, without assigning them a position;
4. excludes unrelated outliers and reconstructs only selected frames with SHARP;
5. lifts verified image matches into SHARP's metric 3D points and estimates a
   RANSAC similarity transform;
6. leaves recognized views without cross-image geometry unregistered instead
   of placing them from similar point-cloud shape;
7. builds a maximum-confidence graph from transforms that pass metric gates;
8. jointly refines camera similarities against every verified edge when a loop
   closure exists;
9. transforms Gaussian positions, scales, and covariance rotations before merge;
10. removes only Gaussians that contradict measured front surfaces in other
    registered views;
11. queues a disconnected group as another room only when that group has its
    own verified overlap edge.

The registration report records match counts, inliers, scale, error, confidence,
and excluded frames. The system does not concatenate unrelated splats and call
them one room. When an explicitly grouped photo set fails registration, the
worker publishes an exact single-photo LucidFrame reconstruction and records the
fallback and reason in the manifest.

Rescue View keeps and renders the complete compiled splat by default. Spark's
LoD hierarchy is disabled for evidence inspection, pre-blur is disabled, and
the normal full Gaussian footprint is retained. The viewer reports the actual
WebGL adapter so an integrated-GPU browser is not mistaken for RTX rendering.

## Visual evidence is not navigation truth

A photorealistic splat does not prove hidden rooms, floor connections,
load-bearing systems, current smoke conditions, or safe passage. Therefore:

- unknown space stays unknown;
- no structure edge is derived from appearance alone;
- route generation requires a connected evidence-backed graph;
- every AI hazard and route remains pending until reviewed.

## Local security boundary

- The worker binds to localhost.
- A non-loopback server bind is refused without `STRUCTUREFIRST_ACCESS_KEY`.
- Sessions use signed, HTTP-only, same-site cookies.
- Cookie-authenticated mutations require an explicit intent header.
- Batch uploads stream to disk, accept at most 50 supported images and 1 GB in
  total, and never trust client filenames.
- Static case assets use the same authorization hook as the API.

TLS, enterprise identity, retention policy, audit export, approved data
agreements, and agency validation remain production requirements.

## Repository layout

```text
apps/web/                  React + Vite operator workspace
apps/server/               Fastify case, evidence, AI, and discovery service
packages/contracts/        Shared Zod schemas and TypeScript types
services/reconstruction/   Local FastAPI LucidFrame worker
docs/                      Design, operator, and verification records
data/                      Local database and case assets (Git-ignored)
```
