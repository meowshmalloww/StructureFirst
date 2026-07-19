# Operator guide

## 1. Enter the property

On **Properties**, enter the most complete street address available and choose
**Prepare view**. No role or incident form is required.

The property opens immediately. The progress strip reports the real current
task: address lookup, map data, online photo search, or LucidFrame processing.

## 2. Let the online scan run

StructureFirst automatically checks:

- the U.S. Census Geocoder or ranked Nominatim results for the location;
- OpenStreetMap for the address-matched building geometry;
- KartaView for nearby, sequenced street captures;
- Wikimedia Commons and Openverse for reusable property-photo matches;
- Bing Images and web results through the installed Chrome or Edge browser;
- Brave Image Search when its optional key is configured.

Public-domain, CC0, CC BY, and CC BY-SA results may be downloaded automatically.
Other web results stay as links because a search result does not grant rights to
copy the image.

Use **Scan again** to repeat the search. StructureFirst uses the submitted and
resolved addresses plus a mapped building name when one is available.

## 3. Add responder photos

Choose **Add responder photos** and select multiple JPEG, PNG, or WebP files.
One batch can contain up to 50 photos and 1 GB total. Files stream to local
storage instead of being buffered in browser or server memory.

- One photo starts a single-image LucidFrame job.
- Two or more photos start smart connect with up to 12 photos.
- Any photos beyond the first 12 are still saved to the property.

For reliable smart connect, capture adjacent views of the same space with about
60–80% overlap, similar lighting, and visible texture. Unrelated web photos are
not merged. If a selected overlap set cannot register, StructureFirst clearly
labels an exact single-photo LucidFrame fallback instead of inventing a joined
scene.

## 4. Use the property view

- **Map** shows the resolved address and any mapped footprint.
- **3D scene** opens the latest completed LucidFrame Gaussian scene.
- **Photos** shows both saved reusable images and link-only search results.
- **Delete** permanently removes the property record and its local photo folder.

A single-image Gaussian provides a nearby-view visual reconstruction. It does
not establish unseen rooms, doors, structural integrity, current hazards, or a
safe interior route.

## 5. Optional AI connection

Online search does not need an AI key. In **Settings**, choose one provider only
if AI-assisted summaries are required later. Supported providers are Groq,
Cerebras, OpenRouter, and NVIDIA NIM.

Enter or save the key, load the filtered no-cost catalog, select a model, then
choose **Verify & save**. StructureFirst activates the connection only after the
model completes a short JSON-format chat request. Keys remain server-side and
are encrypted in the local data directory.

The dropdown policy is provider-specific:

- Groq shows current general-chat models listed under official Free Plan limits.
- Cerebras shows public chat models returned for the key.
- OpenRouter shows only zero-price, text-output models allowed by the key's
  preferences.
- NVIDIA intersects the key's mixed API catalog with NVIDIA's documented chat
  and image-understanding prototype endpoints.

Provider quotas and availability are controlled by the providers and can
change. A catalog entry is a candidate; **Verify & save** is the access and
format check for the selected model.

## Failure states

- **Address lookup failed:** return to Properties and enter a fuller address.
- **No property photos yet:** continue with the map or add responder photos.
- **Photos need more overlap:** add adjacent views with more shared detail.
- **Reconstruction failed:** check the worker log, GPU memory, LucidFrame path,
  and supported image type.
- **Browser search unavailable:** set `STRUCTUREFIRST_BROWSER_EXECUTABLE` to a
  local Chrome or Edge executable; KartaView, Wikimedia Commons, and Openverse
  can still run without it.
