import {
  Box,
  ChevronLeft,
  CircleAlert,
  Crosshair,
  ExternalLink,
  Image as ImageIcon,
  Layers3,
  LoaderCircle,
  Map as MapIcon,
  MapPin,
  RefreshCw,
  ShieldAlert,
  Trash2,
  Upload,
} from "lucide-react";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
} from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import type {
  CaseWorkspace,
  EvidenceAsset,
  PipelineStageName,
  RoomType,
  SpatialNode,
  StageStatus,
} from "@structurefirst/contracts";
import { api } from "../lib/api";
import { StructureMap } from "../components/StructureMap";
import { SplatViewer } from "../components/SplatViewer";
import {
  StructuralViewer,
  type StructuralRoomSelection,
} from "../components/StructuralViewer";
import {
  artifactForSpace,
  registeredEvidenceIds,
  resolvePlanRoom,
  roomTypeTitle,
  type PlanRoomCandidate,
} from "../lib/room-availability";
import {
  artifactSceneLabels,
  maximalReadyArtifacts,
} from "../lib/scene-labels";

type PropertyView = "rescue" | "structure" | "map";

const PREPARATION_STAGES: PipelineStageName[] = [
  "address_resolution",
  "building_records",
  "reconstruction",
];

export function CasePage() {
  const { caseId = "" } = useParams();
  const navigate = useNavigate();
  const fileInput = useRef<HTMLInputElement>(null);
  const refreshTimer = useRef<number | undefined>(undefined);
  const [workspace, setWorkspace] = useState<CaseWorkspace>();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>();
  const [message, setMessage] = useState<string>();
  const [view, setView] = useState<PropertyView>("rescue");
  const [uploading, setUploading] = useState(false);
  const [uploadingCount, setUploadingCount] = useState(0);
  const [deleting, setDeleting] = useState(false);
  const [selectedArtifactId, setSelectedArtifactId] = useState<string>();
  const [selectedSpaceId, setSelectedSpaceId] = useState<string>();
  const [inputsOpen, setInputsOpen] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setWorkspace(await api.workspace(caseId));
      setError(undefined);
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "The property could not be loaded.",
      );
    } finally {
      setLoading(false);
    }
  }, [caseId]);

  useEffect(() => {
    void refresh();
    const stream = new EventSource(
      `/api/cases/${encodeURIComponent(caseId)}/events`,
    );
    stream.onmessage = () => {
      window.clearTimeout(refreshTimer.current);
      refreshTimer.current = window.setTimeout(() => void refresh(), 120);
    };
    return () => {
      stream.close();
      window.clearTimeout(refreshTimer.current);
    };
  }, [caseId, refresh]);

  const readyArtifacts = useMemo(() => {
    if (!workspace) return [];
    const eligiblePhotoIds = new Set(
      workspace.evidence
        .filter(
          (evidence) =>
            evidence.kind === "image" &&
            !evidence.tags.includes("reconstruction-excluded"),
        )
        .map((evidence) => evidence.id),
    );
    return maximalReadyArtifacts(workspace.artifacts).filter(
      (artifact) =>
        artifact.mode === "floorplan" ||
        registeredEvidenceIds(artifact).some((id) => eligiblePhotoIds.has(id)),
    );
  }, [workspace]);
  const reconstructionArtifacts = useMemo(
    () => readyArtifacts.filter((artifact) => artifact.mode !== "floorplan"),
    [readyArtifacts],
  );
  const structuralArtifact = useMemo(
    () => readyArtifacts.find((artifact) => artifact.mode === "floorplan"),
    [readyArtifacts],
  );
  const readyArtifact = useMemo(
    () =>
      reconstructionArtifacts.find(
        (artifact) => artifact.id === selectedArtifactId,
      ) ??
      reconstructionArtifacts.find(
        (artifact) =>
          !artifact.fallback &&
          (artifact.mode !== "multi_image" ||
            (artifact.registration?.connectedFrameCount ?? 0) >= 2),
      ) ??
      reconstructionArtifacts[0],
    [reconstructionArtifacts, selectedArtifactId],
  );
  const failedPhotoArtifact = useMemo(
    () =>
      workspace?.artifacts.find(
        (artifact) =>
          artifact.mode !== "floorplan" && artifact.status === "failed",
      ),
    [workspace?.artifacts],
  );
  const currentArtifact = useMemo(
    () =>
      workspace?.artifacts.find((artifact) =>
        ["queued", "running"].includes(artifact.status),
      ) ??
      workspace?.artifacts.find((artifact) => artifact.status === "failed"),
    [workspace?.artifacts],
  );

  useEffect(() => {
    const latest =
      reconstructionArtifacts.find(
        (artifact) =>
          !artifact.fallback &&
          (artifact.mode !== "multi_image" ||
            (artifact.registration?.connectedFrameCount ?? 0) >= 2),
      ) ??
      reconstructionArtifacts[0];
    if (latest) {
      setSelectedArtifactId((current) => {
        if (!current) return latest.id;
        const selected = readyArtifacts.find(
          (artifact) => artifact.id === current,
        );
        return selected &&
          sameEvidenceSet(selected.evidenceIds, latest.evidenceIds)
          ? latest.id
          : current;
      });
      setView("rescue");
    } else if (structuralArtifact) {
      setView("structure");
    }
  }, [reconstructionArtifacts[0]?.id, structuralArtifact?.id]);

  async function uploadPhotos(event: ChangeEvent<HTMLInputElement>) {
    const files = [...(event.target.files ?? [])].sort((left, right) =>
      left.name.localeCompare(right.name, undefined, {
        numeric: true,
        sensitivity: "base",
      }),
    );
    event.target.value = "";
    if (files.length === 0) return;
    if (files.length > 50) {
      setError("Select no more than 50 photos at once.");
      return;
    }
    const total = files.reduce((sum, file) => sum + file.size, 0);
    if (total > 1024 * 1024 * 1024) {
      setError("The selected photos exceed 1 GB in total.");
      return;
    }
    const form = new FormData();
    form.append("captureMode", "perspective");
    for (const file of files) form.append("files", file, file.name);
    setUploading(true);
    setUploadingCount(files.length);
    setMessage(undefined);
    setError(undefined);
    try {
      const result = await api.uploadPhotos(caseId, form);
      setMessage(result.note);
      setView("rescue");
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Upload failed.");
    } finally {
      setUploading(false);
      setUploadingCount(0);
    }
  }

  async function deleteProperty() {
    if (!workspace) return;
    if (
      !window.confirm(
        `Delete ${workspace.case.displayAddress} and all saved photos?`,
      )
    )
      return;
    setDeleting(true);
    setError(undefined);
    try {
      await api.deleteCase(caseId);
      navigate("/");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Delete failed.");
      setDeleting(false);
    }
  }

  const openPlanRoom = useCallback(
    (selection: StructuralRoomSelection, candidate?: PlanRoomCandidate) => {
      if (!workspace) return;
      if (candidate) {
        setSelectedSpaceId(candidate.space.id);
        setSelectedArtifactId(candidate.artifact.id);
        setView("rescue");
        if (
          !resolvePlanRoom(
            selection,
            workspace.nodes,
            reconstructionArtifacts,
            workspace.evidence,
          ).available
        ) {
          setMessage(
            `${selection.label}: opened a matching reconstructed scene; its exact floor-plan position is not verified.`,
          );
        }
        return;
      }
      const resolution = resolvePlanRoom(
        selection,
        workspace.nodes,
        reconstructionArtifacts,
        workspace.evidence,
      );
      if (!resolution.available || !resolution.space || !resolution.artifact) {
        setMessage(`${selection.label}: ${resolution.detail}`);
        return;
      }
      setSelectedSpaceId(resolution.space.id);
      setSelectedArtifactId(resolution.artifact.id);
      setView("rescue");
    },
    [reconstructionArtifacts, workspace],
  );

  const planRoomAvailability = useCallback(
    (selection: StructuralRoomSelection) => {
      if (!workspace)
        return {
          available: false,
          detail: "Property data is not loaded.",
          candidates: [],
        };
      return resolvePlanRoom(
        selection,
        workspace.nodes,
        reconstructionArtifacts,
        workspace.evidence,
      );
    },
    [reconstructionArtifacts, workspace],
  );

  if (loading)
    return (
      <div className="page-loading">
        <LoaderCircle className="spin" size={18} /> Loading property
      </div>
    );

  if (!workspace)
    return (
      <div className="fatal-state">
        <CircleAlert size={26} />
        <h1>Property unavailable</h1>
        <p>{error}</p>
        <Link to="/">Return to properties</Link>
      </div>
    );

  const current = workspace.case;
  const photos = workspace.evidence.filter(
    (item) =>
      item.kind === "image" &&
      Boolean(item.localUrl) &&
      !item.tags.includes("reconstruction-excluded") &&
      (item.tags.includes("operator-upload") ||
        item.tags.includes("address-text-match") ||
        item.tags.includes("listing-address-match") ||
        item.visualAnalysis?.addressMatch === "supported"),
  );
  const floorPlans = workspace.evidence.filter(
    (item) =>
      item.kind === "blueprint" &&
      Boolean(item.localUrl) &&
      !item.tags.includes("reconstruction-excluded") &&
      (item.tags.includes("operator-upload") ||
        item.visualAnalysis?.sceneType === "floor_plan"),
  );
  const excludedMedia = workspace.evidence.filter(
    (item) =>
      Boolean(item.localUrl) && item.tags.includes("reconstruction-excluded"),
  );
  const spaces = workspace.nodes
    .filter(
      (node) =>
        Boolean(artifactForSpace(node, reconstructionArtifacts)),
    )
    .filter((node, index, nodes) => {
      const artifact = artifactForSpace(node, reconstructionArtifacts);
      return (
        artifact &&
        nodes.findIndex(
          (candidate) =>
            artifactForSpace(candidate, reconstructionArtifacts)?.id ===
            artifact.id,
        ) === index
      );
    })
    .sort(compareSpatialScenes);
  const selectedSpace =
    spaces.find((space) => space.id === selectedSpaceId) ?? spaces[0];
  const registeredPhotoIds = new Set(
    reconstructionArtifacts.flatMap(registeredEvidenceIds),
  );
  const unavailableCaptureGroups = groupUnavailableCaptures(
    photos.filter((photo) => !registeredPhotoIds.has(photo.id)),
  );
  const sceneLabels = artifactSceneLabels(readyArtifacts, photos);
  const progress = preparationProgress(workspace);
  const status = preparationStatus(
    workspace,
    readyArtifact,
    currentArtifact,
    failedPhotoArtifact,
  );

  return (
    <div className="property-page">
      <header className="property-header">
        <div className="property-title">
          <Link to="/" className="back-link" aria-label="Back to properties">
            <ChevronLeft size={19} />
          </Link>
          <div>
            <span className="property-breadcrumb">
              Operations / Rescue View
            </span>
            <h1>{current.displayAddress}</h1>
          </div>
        </div>
        <div className="property-actions">
          <button
            type="button"
            className="secondary-button refresh-property"
            onClick={() => void refresh()}
          >
            <RefreshCw size={16} /> Refresh
          </button>
          <button
            type="button"
            className="icon-button danger-button"
            disabled={deleting}
            onClick={() => void deleteProperty()}
            aria-label="Delete property"
          >
            {deleting ? (
              <LoaderCircle className="spin" size={17} />
            ) : (
              <Trash2 size={17} />
            )}
          </button>
        </div>
      </header>

      <section className="case-status-panel" aria-live="polite">
        <div className="case-status-summary">
          <span className="case-status-icon" aria-hidden="true">
            {progress < 100 ? (
              <LoaderCircle className="spin" size={16} />
            ) : (
              <Box size={16} />
            )}
          </span>
          <span className="case-status-copy">
            <strong>{status.title}</strong>
            <small>{status.detail}</small>
          </span>
        </div>
        <div className="case-progress">
          <span>{progress}% complete</span>
          <progress max={100} value={progress} />
        </div>
        <ol className="pipeline-steps" aria-label="Preparation stages">
          {PREPARATION_STAGES.map((name) => {
            const stage = current.stages.find((item) => item.name === name);
            const hasPendingReconstruction = workspace.artifacts.some(
              (artifact) =>
                artifact.status === "queued" || artifact.status === "running",
            );
            const visibleStatus =
              name === "reconstruction" && hasPendingReconstruction
                ? "running"
                : name === "reconstruction" && readyArtifacts.length
                  ? failedPhotoArtifact
                    ? "limited"
                    : "complete"
                  : stage?.status;
            return (
              <li
                key={name}
                className={`pipeline-${visibleStatus ?? "pending"}`}
              >
                <span className="pipeline-icon" aria-hidden="true">
                  {stageIcon(name)}
                </span>
                <span>
                  <strong>{stageLabel(name)}</strong>
                  <small>{stageStatusLabel(visibleStatus)}</small>
                </span>
              </li>
            );
          })}
        </ol>
      </section>

      {error ? (
        <div className="property-notice error-notice" role="alert">
          <CircleAlert size={16} />
          <span>{error}</span>
          <button type="button" onClick={() => setError(undefined)}>
            Dismiss
          </button>
        </div>
      ) : null}
      {message ? (
        <div className="property-notice">
          <span>{message}</span>
          <button type="button" onClick={() => setMessage(undefined)}>
            Dismiss
          </button>
        </div>
      ) : null}

      <input
        ref={fileInput}
        className="sr-only"
        type="file"
        multiple
        accept="image/jpeg,image/png,image/webp"
        onChange={(event) => void uploadPhotos(event)}
      />

      <div className={`property-layout ${inputsOpen ? "inputs-open" : ""}`}>
        <section className="property-canvas">
          <div className="canvas-toolbar">
            <div
              className="view-switch"
              role="tablist"
              aria-label="Property view"
            >
              <button
                type="button"
                role="tab"
                aria-selected={view === "rescue"}
                onClick={() => setView("rescue")}
              >
                <Crosshair size={15} /> Rescue View
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={view === "structure"}
                disabled={!structuralArtifact}
                onClick={() => setView("structure")}
              >
                <Layers3 size={15} /> House Map
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={view === "map"}
                onClick={() => setView("map")}
              >
                <MapIcon size={15} /> Location
              </button>
            </div>
            <div className="canvas-toolbar-actions">
              <span className="canvas-source-count">
                <ImageIcon size={14} />
                {floorPlans.length
                  ? `${photos.length} photos · ${floorPlans.length} plans`
                  : captureCountLabel(photos.length, readyArtifact)}
              </span>
              <button
                type="button"
                className="capture-drawer-button"
                aria-expanded={inputsOpen}
                onClick={() => setInputsOpen((current) => !current)}
              >
                <Upload size={14} /> {inputsOpen ? "Close inputs" : "Inputs"}
                <span>{photos.length + floorPlans.length}</span>
              </button>
            </div>
          </div>
          {view === "rescue" ? (
            <div className="rescue-view-limit">
              <ShieldAlert size={15} />
              <span>
                <strong>Observed imagery only.</strong> Free-flight view does
                not verify collision-safe routes or unseen interior space.
              </span>
            </div>
          ) : null}
          {view === "rescue" &&
          failedPhotoArtifact &&
          reconstructionArtifacts.length === 0 ? (
            <div className="gaussian-unavailable" role="status">
              <CircleAlert size={16} />
              <span>
                <strong>
                  No Gaussian building was created from these photos.
                </strong>
                The plan model below is a separate evidence layer. The uploaded
                photographic views did not contain a verified connected overlap
                path.
              </span>
            </div>
          ) : null}
          <div
            className={`viewer-workspace ${view === "rescue" ? "has-scene-browser" : ""}`}
          >
          {view === "rescue" && readyArtifact ? (
            <aside
              className="space-index scene-browser"
              aria-label="Reconstructed floors and rooms"
            >
              <span className="space-index-label">
                <Layers3 size={14} /> 3D scenes
              </span>
              {spaces.length ? (
                spaces.map((space) => {
                  const artifact = artifactForSpace(
                    space,
                    reconstructionArtifacts,
                  );
                  const selected = space.id === selectedSpace?.id;
                  return (
                    <button
                      type="button"
                      key={space.id}
                      aria-pressed={selected}
                      onPointerEnter={() => prefetchScene(artifact?.splatUrl)}
                      onFocus={() => prefetchScene(artifact?.splatUrl)}
                      onClick={() => {
                        if (!artifact) return;
                        setSelectedArtifactId(artifact.id);
                        setSelectedSpaceId(space.id);
                      }}
                    >
                      <strong>{spaceFloorLabel(space)}</strong>
                      <span>
                        {artifact
                          ? sceneLabels.get(artifact.id)?.split(" · ")[0]
                          : space.label.split(" · ")[0]}
                      </span>
                    </button>
                  );
                })
              ) : (
                <span className="space-index-empty">
                  No room has a connected 3D scene yet
                </span>
              )}
              {unavailableCaptureGroups.length ? (
                <div
                  className="space-unavailable-groups"
                  aria-label="Captures without a connected 3D scene"
                >
                  <strong>Not reconstructed</strong>
                  {unavailableCaptureGroups.map((group) => (
                    <span
                      key={group.key}
                      title="These captures are not part of a calibrated Gaussian scene."
                    >
                      <span>{group.label}</span>
                      <small>
                        {group.count}{" "}
                        {group.count === 1 ? "capture" : "captures"} · No 3D
                      </small>
                    </span>
                  ))}
                </div>
              ) : null}
            </aside>
          ) : null}
          <div className="canvas-body">
            {view === "map" ? (
              <StructureMap cases={[current]} activeCaseId={current.id} />
            ) : view === "structure" ? (
              structuralArtifact ? (
                <StructuralViewer
                  artifact={structuralArtifact}
                  onRoomSelect={openPlanRoom}
                  roomAvailability={planRoomAvailability}
                />
              ) : (
                <div className="scene-placeholder">
                  <Layers3 size={28} />
                  <h2>No structural plan available</h2>
                  <p>Add a floor plan or blueprint to build the house map.</p>
                </div>
              )
            ) : readyArtifact ? (
              <SplatViewer
                key={readyArtifact.id}
                artifact={readyArtifact}
                {...(selectedSpace?.sourceIds[0] &&
                readyArtifact.id ===
                  artifactForSpace(selectedSpace, reconstructionArtifacts)?.id
                  ? { focusEvidenceId: selectedSpace.sourceIds[0] }
                  : {})}
              />
            ) : (
              <div className="scene-placeholder">
                {currentArtifact?.status === "failed" ? (
                  <CircleAlert size={28} />
                ) : currentArtifact ? (
                  <LoaderCircle className="spin" size={28} />
                ) : (
                  <ImageIcon size={28} />
                )}
                <h2>
                  {currentArtifact?.status === "failed"
                    ? "These photos did not connect"
                    : floorPlans.length
                      ? "Building structural floors"
                      : photos.length
                        ? "Preparing Rescue View"
                        : "Rescue View needs overlapping photos"}
                </h2>
                <p>
                  {currentArtifact?.status === "failed"
                    ? "Add overlapping photos of the same space and StructureFirst will try again."
                    : floorPlans.length
                      ? "The plans are being vectorized into floor plates and wall segments. Photographs are handled separately."
                      : photos.length
                        ? "LucidFrame is reconstructing ordered capture groups and verifying every connection."
                        : "Add a continuous, overlapping capture walk through the rooms, doorways, halls, and stairs."}
                </p>
                {currentArtifact?.error ? (
                  <small>{currentArtifact.error}</small>
                ) : null}
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => fileInput.current?.click()}
                >
                  <Upload size={15} /> Add photos
                </button>
              </div>
            )}
          </div>
          </div>
        </section>

        {inputsOpen ? (
        <aside className="photo-panel">
          <header>
            <div>
              <span className="panel-kicker">Rescue View input</span>
              <h2>Capture set</h2>
            </div>
            <span className="panel-count">
              {photos.length + floorPlans.length}
            </span>
          </header>

          <div className="capture-mode" aria-label="Capture type">
            <button type="button" aria-pressed="true">
              Whole house
            </button>
          </div>
          <p className="capture-guidance">
            Add plans and one continuous photo walk. Keep 60–80% overlap,
            capture 2–3 bridge views through every doorway, and photograph
            stairs continuously from bottom to top and top to bottom.
          </p>

          <button
            type="button"
            className="photo-upload"
            disabled={uploading}
            onClick={() => fileInput.current?.click()}
          >
            {uploading ? (
              <LoaderCircle className="spin" size={18} />
            ) : (
              <Upload size={18} />
            )}
            <span>
              <strong>
                {uploading
                  ? `Saving ${uploadingCount} ${uploadingCount === 1 ? "file" : "files"}…`
                  : "Add capture photos or plans"}
              </strong>
              <small>
                Saved first · organized in background · JPEG, PNG, or WebP · 1
                GB total
              </small>
            </span>
          </button>
          <div className="photo-grid">
            {photos.length === 0 ? (
              <div className="photo-empty">
                <ImageIcon size={23} />
                <strong>No usable captures yet</strong>
                <span>Add ordered, overlapping photos from the property.</span>
              </div>
            ) : (
              photos.map((photo) => <PhotoTile key={photo.id} photo={photo} />)
            )}
          </div>

          {floorPlans.length ? (
            <section className="plan-inputs">
              <header>
                <strong>Structural plans</strong>
                <span>{floorPlans.length} detected</span>
              </header>
              <div className="photo-grid">
                {floorPlans.map((plan) => (
                  <PhotoTile key={plan.id} photo={plan} />
                ))}
              </div>
            </section>
          ) : null}

          {excludedMedia.length ? (
            <section className="plan-inputs excluded-inputs">
              <header>
                <strong>Excluded from reconstruction</strong>
                <span>{excludedMedia.length}</span>
              </header>
              <div className="photo-grid">
                {excludedMedia.map((item) => (
                  <PhotoTile key={item.id} photo={item} />
                ))}
              </div>
            </section>
          ) : null}

          <p className="photo-policy">
            Original files stay local. Geometry decides which views connect;
            unverified gaps remain visibly separate.
          </p>
        </aside>
        ) : null}
      </div>
    </div>
  );
}

function PhotoTile({ photo }: { photo: EvidenceAsset }) {
  const preview = photo.localUrl ?? photo.thumbnailUrl;
  return (
    <article className="photo-tile">
      <div>
        {preview ? (
          <img src={preview} alt={photo.title} loading="lazy" />
        ) : (
          <ImageIcon size={22} aria-hidden="true" />
        )}
        <span>{photo.localUrl ? "Saved" : "Link only"}</span>
      </div>
      <footer>
        <span title={photo.title}>
          {photo.title}
          {photo.visualAnalysis ? (
            <small>
              {photo.kind === "blueprint"
                ? "Plan sheet · classified during vectorization"
                : photo.tags.includes("reconstruction-excluded")
                  ? "Excluded"
                  : `${roomTypeLabel(photo.visualAnalysis.roomType)} · ${floorHintLabel(photo.visualAnalysis.floorHint)}`}
            </small>
          ) : null}
        </span>
        {photo.originUrl ? (
          <a
            href={photo.originUrl}
            target="_blank"
            rel="noreferrer"
            aria-label={`Open source for ${photo.title}`}
          >
            <ExternalLink size={13} />
          </a>
        ) : null}
      </footer>
    </article>
  );
}

export function artifactSceneLabel(
  artifact: CaseWorkspace["artifacts"][number],
  photos: EvidenceAsset[],
): string {
  if (artifact.mode === "floorplan")
    return `House map · ${artifact.evidenceIds?.length ?? 1}`;
  const registeredIds = new Set(registeredEvidenceIds(artifact));
  const registeredPhotos = photos.filter((photo) =>
    registeredIds.has(photo.id),
  );
  const exteriorScene =
    registeredPhotos.length > 0 &&
    registeredPhotos.every(
      (photo) => photo.visualAnalysis?.sceneType === "exterior",
    );
  const roomTypes = [
    ...new Set(
      registeredPhotos
        .map((photo) => photo.visualAnalysis?.roomType)
        .filter((roomType): roomType is RoomType =>
          Boolean(roomType && roomType !== "unknown"),
        ),
    ),
  ];
  const viewCount = registeredIds.size || 1;
  const label = exteriorScene
    ? "Exterior"
    : roomTypes.length === 1 && roomTypes[0]
      ? roomTypeTitle(roomTypes[0])
      : artifact.fallback
        ? "Single-view scene"
        : "Connected scene";
  return `${label} · ${viewCount} ${viewCount === 1 ? "view" : "views"}`;
}

function groupUnavailableCaptures(photos: EvidenceAsset[]) {
  const groups = new Map<
    string,
    { key: string; label: string; count: number }
  >();
  for (const photo of photos) {
    const roomType = photo.visualAnalysis?.roomType ?? "unknown";
    const key = roomType;
    const current = groups.get(key);
    if (current) current.count += 1;
    else
      groups.set(key, {
        key,
        label: roomTypeTitle(roomType),
        count: 1,
      });
  }
  return [...groups.values()].sort(
    (left, right) =>
      right.count - left.count || left.label.localeCompare(right.label),
  );
}

function roomTypeLabel(
  value: NonNullable<EvidenceAsset["visualAnalysis"]>["roomType"],
) {
  if (value === "unknown") return "Space";
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function floorHintLabel(
  value: NonNullable<EvidenceAsset["visualAnalysis"]>["floorHint"],
) {
  if (value === "ground") return "Ground floor";
  if (value === "upper") return "Upper floor";
  if (value === "basement") return "Basement";
  if (value === "attic") return "Attic";
  return "Floor unknown";
}

function spaceFloorLabel(space: SpatialNode): string {
  if (space.kind === "exterior") return "Exterior";
  if (!space.floorLabel || space.floorLabel === "Unknown floor")
    return "Floor not verified";
  return space.floorLabel;
}

function compareSpatialScenes(left: SpatialNode, right: SpatialNode): number {
  const floorOrder = (node: SpatialNode) => {
    if (node.kind === "exterior") return 0;
    if (node.floorLabel === "Basement") return 1;
    if (node.floorLabel === "Ground floor") return 2;
    if (node.floorLabel === "Upper floor") return 3;
    if (node.floorLabel === "Attic") return 4;
    return 5;
  };
  return (
    floorOrder(left) - floorOrder(right) ||
    (left.roomType ?? left.kind).localeCompare(right.roomType ?? right.kind) ||
    left.label.localeCompare(right.label, undefined, {
      numeric: true,
      sensitivity: "base",
    })
  );
}

function prefetchScene(url?: string): void {
  if (!url || document.head.querySelector(`link[data-splat-prefetch="${CSS.escape(url)}"]`))
    return;
  const link = document.createElement("link");
  link.rel = "prefetch";
  link.as = "fetch";
  link.href = url;
  link.dataset.splatPrefetch = url;
  document.head.appendChild(link);
}

function sameEvidenceSet(left?: string[], right?: string[]): boolean {
  if (!left || !right || left.length !== right.length) return false;
  const expected = new Set(left);
  return right.every((id) => expected.has(id));
}

function stageIcon(name: PipelineStageName) {
  if (name === "address_resolution") return <MapPin size={15} />;
  if (name === "building_records") return <MapIcon size={15} />;
  if (name === "evidence_discovery") return <ImageIcon size={15} />;
  return <Crosshair size={15} />;
}

function stageLabel(name: PipelineStageName): string {
  if (name === "address_resolution") return "Address";
  if (name === "building_records") return "Map";
  if (name === "evidence_discovery") return "Connected photos";
  return "Rescue View";
}

function stageStatusLabel(status?: StageStatus): string {
  if (status === "running") return "In progress";
  if (status === "complete") return "Complete";
  if (status === "limited") return "Limited";
  if (status === "skipped") return "Not needed";
  if (status === "failed") return "Needs attention";
  return "Waiting";
}

function preparationProgress(workspace: CaseWorkspace): number {
  const stages = PREPARATION_STAGES.map((name) =>
    workspace.case.stages.find((stage) => stage.name === name),
  );
  const complete = stages.filter(
    (stage) =>
      stage &&
      ["complete", "limited", "skipped", "failed"].includes(stage.status),
  ).length;
  const pendingArtifacts = workspace.artifacts.filter(
    (artifact) => artifact.status === "queued" || artifact.status === "running",
  ).length;
  if (pendingArtifacts > 0) {
    const readyArtifacts = workspace.artifacts.filter(
      (artifact) => artifact.status === "ready",
    ).length;
    const reconstructionShare =
      readyArtifacts / Math.max(1, readyArtifacts + pendingArtifacts);
    const nonReconstructionComplete = stages
      .slice(0, 2)
      .filter(
        (stage) =>
          stage && ["complete", "limited", "skipped"].includes(stage.status),
      ).length;
    return Math.min(
      99,
      Math.round(((nonReconstructionComplete + reconstructionShare) / 3) * 100),
    );
  }
  const runningBonus = stages.some((stage) => stage?.status === "running")
    ? 0.45
    : 0;
  return Math.min(
    100,
    Math.round(((complete + runningBonus) / stages.length) * 100),
  );
}

function preparationStatus(
  workspace: CaseWorkspace,
  readyArtifact: CaseWorkspace["artifacts"][number] | undefined,
  currentArtifact: CaseWorkspace["artifacts"][number] | undefined,
  failedPhotoArtifact: CaseWorkspace["artifacts"][number] | undefined,
): { title: string; detail: string } {
  const hasReadyGaussian = workspace.artifacts.some(
    (artifact) =>
      artifact.status === "ready" &&
      artifact.mode !== "floorplan" &&
      Boolean(artifact.splatUrl),
  );
  const pendingArtifacts = workspace.artifacts.filter(
    (artifact) => artifact.status === "queued" || artifact.status === "running",
  );
  const readyGaussianCount = workspace.artifacts.filter(
    (artifact) =>
      artifact.status === "ready" &&
      artifact.mode !== "floorplan" &&
      Boolean(artifact.splatUrl),
  ).length;
  if (pendingArtifacts.length > 0)
    return {
      title: "Building remaining 3D scenes",
      detail: `${readyGaussianCount} ${readyGaussianCount === 1 ? "scene is" : "scenes are"} ready; ${pendingArtifacts.length} ${pendingArtifacts.length === 1 ? "scene is" : "scenes are"} still reconstructing automatically.`,
    };
  if (readyArtifact?.mode === "floorplan")
    return {
      title:
        failedPhotoArtifact && !hasReadyGaussian
          ? "Plan reference ready · Gaussian unavailable"
          : "Plan reference ready",
      detail: `${readyArtifact.evidenceIds?.length ?? 1} supplied plan ${
        (readyArtifact.evidenceIds?.length ?? 1) === 1 ? "sheet" : "sheets"
      } classified; labeled floor panels are separated and site plans remain reference-only${
        failedPhotoArtifact && !hasReadyGaussian
          ? "; no additional Gaussian scene registered from the remaining photos"
          : ""
      }`,
    };
  if (readyArtifact)
    return {
      title: "Rescue View ready",
      detail: readyArtifact.fallback
        ? `${readyArtifact.gaussianCount?.toLocaleString() ?? "Gaussian"} scene from one exact source photo; the other captures did not register`
        : `${readyArtifact.gaussianCount?.toLocaleString() ?? "Gaussian"} scene from ${readyArtifact.registration?.connectedFrameCount ?? 1} connected ${
            (readyArtifact.registration?.connectedFrameCount ?? 1) === 1
              ? "capture"
              : "captures"
          }`,
    };
  if (currentArtifact?.status === "failed")
    return {
      title: "No Gaussian building was produced",
      detail:
        "The photos did not form one connected camera path. Add adjacent views with 60–80% overlap and bridge every doorway and stair.",
    };
  if (currentArtifact)
    return {
      title: "Building Rescue View",
      detail: "LucidFrame is reconstructing and registering captures locally.",
    };
  const running = workspace.case.stages.find(
    (stage) => stage.status === "running",
  );
  if (running?.name === "address_resolution")
    return { title: "Finding the address", detail: running.message };
  if (running?.name === "building_records")
    return { title: "Loading map data", detail: running.message };
  if (workspace.case.status === "failed")
    return {
      title: "Address lookup failed",
      detail: "Check the address and try again.",
    };
  return {
    title: "Captures needed",
    detail: "Add one ordered, overlapping capture walk through the property.",
  };
}

function captureCountLabel(
  photoCount: number,
  artifact: CaseWorkspace["artifacts"][number] | undefined,
): string {
  if (artifact?.fallback) return `1/${photoCount} capture used`;
  if (artifact?.registration)
    return `${artifact.registration.connectedFrameCount}/${artifact.registration.frameCount} captures connected`;
  return `${photoCount} ${photoCount === 1 ? "capture" : "captures"}`;
}
