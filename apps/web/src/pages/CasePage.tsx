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
  ScanSearch,
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
  StageStatus,
} from "@structurefirst/contracts";
import { api } from "../lib/api";
import { StructureMap } from "../components/StructureMap";
import { SplatViewer } from "../components/SplatViewer";

type PropertyView = "rescue" | "map";

const PREPARATION_STAGES: PipelineStageName[] = [
  "address_resolution",
  "evidence_discovery",
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
  const [scanning, setScanning] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [selectedArtifactId, setSelectedArtifactId] = useState<string>();

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

  const readyArtifacts = useMemo(
    () =>
      workspace?.artifacts.filter((artifact) => artifact.status === "ready") ??
      [],
    [workspace?.artifacts],
  );
  const readyArtifact = useMemo(
    () =>
      readyArtifacts.find((artifact) => artifact.id === selectedArtifactId) ??
      readyArtifacts[0],
    [readyArtifacts, selectedArtifactId],
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
    const latest = readyArtifacts[0];
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
    }
  }, [readyArtifacts[0]?.id]);

  async function scanWeb() {
    setScanning(true);
    setMessage(undefined);
    setError(undefined);
    try {
      const result = await api.discover(caseId, {
        includeOpenverse: true,
        includeBrowser: true,
        includeBrave: true,
      });
      const detail = result.imported
        ? `${result.imported} reusable ${result.imported === 1 ? "photo" : "photos"} imported`
        : `${result.added} new source ${result.added === 1 ? "link" : "links"} found`;
      setMessage(
        result.warnings[0] ? `${detail}. ${result.warnings[0]}` : `${detail}.`,
      );
      await refresh();
    } catch (caught) {
      setError(
        caught instanceof Error ? caught.message : "Online scan failed.",
      );
    } finally {
      setScanning(false);
    }
  }

  async function uploadPhotos(event: ChangeEvent<HTMLInputElement>) {
    const files = [...(event.target.files ?? [])];
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
    for (const file of files) form.append("files", file, file.name);
    setUploading(true);
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
      (item.tags.includes("operator-upload") ||
        item.tags.includes("address-text-match") ||
        item.tags.includes("listing-address-match") ||
        item.visualAnalysis?.addressMatch === "supported"),
  );
  const listingLinks = workspace.evidence.filter(
    (item) =>
      item.tags.includes("listing-source") &&
      item.tags.includes("listing-address-match"),
  );
  const spaces = workspace.nodes.filter((node) => node.kind !== "exterior");
  const progress = preparationProgress(workspace);
  const status = preparationStatus(workspace, readyArtifact, currentArtifact);
  const evidenceStage = current.stages.find(
    (stage) => stage.name === "evidence_discovery",
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
            return (
              <li
                key={name}
                className={`pipeline-${stage?.status ?? "pending"}`}
              >
                <span className="pipeline-icon" aria-hidden="true">
                  {stageIcon(name)}
                </span>
                <span>
                  <strong>{stageLabel(name)}</strong>
                  <small>{stageStatusLabel(stage?.status)}</small>
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

      <div className="property-layout">
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
                aria-selected={view === "map"}
                onClick={() => setView("map")}
              >
                <MapIcon size={15} /> Map
              </button>
            </div>
            <span className="canvas-source-count">
              <ImageIcon size={14} />
              {captureCountLabel(photos.length, readyArtifact)}
            </span>
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
          {view === "rescue" ? (
            <div
              className="space-index"
              aria-label="Reconstructed floors and rooms"
            >
              <span className="space-index-label">
                <Layers3 size={14} /> Floors & rooms
              </span>
              {spaces.length ? (
                spaces.map((space) => {
                  const artifact = artifactForSpace(
                    space.sourceIds,
                    readyArtifacts,
                  );
                  const selected = artifact?.id === readyArtifact?.id;
                  return (
                    <button
                      type="button"
                      key={space.id}
                      aria-pressed={selected}
                      disabled={!artifact}
                      onClick={() =>
                        artifact && setSelectedArtifactId(artifact.id)
                      }
                    >
                      <strong>{space.floorLabel ?? "Unknown floor"}</strong>
                      <span>{space.label.split(" · ")[0]}</span>
                    </button>
                  );
                })
              ) : (
                <span className="space-index-empty">
                  Floor unknown · waiting for registered interior views
                </span>
              )}
            </div>
          ) : null}
          <div className="canvas-body">
            {view === "map" ? (
              <StructureMap cases={[current]} activeCaseId={current.id} />
            ) : readyArtifact ? (
              <SplatViewer artifact={readyArtifact} />
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
                    : photos.length
                      ? "Preparing Rescue View"
                      : "Rescue View needs overlapping photos"}
                </h2>
                <p>
                  {currentArtifact?.status === "failed"
                    ? "Add overlapping photos of the same space and StructureFirst will try again."
                    : photos.length
                      ? "LucidFrame is reconstructing and connecting the available captures."
                      : evidenceStage?.status === "running"
                        ? "Online collection is still looking for reusable property images."
                        : "No reusable online photos were found. Add adjacent responder photos with 60-80% overlap."}
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
        </section>

        <aside className="photo-panel">
          <header>
            <div>
              <span className="panel-kicker">Rescue View input</span>
              <h2>Capture set</h2>
            </div>
            <span className="panel-count">{photos.length}</span>
            <button
              type="button"
              className="small-button"
              disabled={scanning}
              onClick={() => void scanWeb()}
            >
              {scanning ? (
                <LoaderCircle className="spin" size={14} />
              ) : (
                <ScanSearch size={14} />
              )}
              Find images
            </button>
          </header>

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
                {uploading ? "Uploading photos…" : "Add overlapping photos"}
              </strong>
              <small>60-80% overlap · JPEG, PNG, or WebP · 1 GB total</small>
            </span>
          </button>
          <input
            ref={fileInput}
            className="sr-only"
            type="file"
            multiple
            accept="image/jpeg,image/png,image/webp"
            onChange={(event) => void uploadPhotos(event)}
          />

          <div className="photo-grid">
            {photos.length === 0 ? (
              <div className="photo-empty">
                <ImageIcon size={23} />
                <strong>No usable captures yet</strong>
                <span>Online collection continues in the background.</span>
              </div>
            ) : (
              photos.map((photo) => <PhotoTile key={photo.id} photo={photo} />)
            )}
          </div>

          {listingLinks.length ? (
            <section
              className="listing-checks"
              aria-label="Address source checks"
            >
              <header>
                <strong>Address source checks</strong>
                <span>Link only</span>
              </header>
              {listingLinks.slice(0, 6).map((source) => (
                <a
                  key={source.id}
                  href={source.originUrl}
                  target="_blank"
                  rel="noreferrer"
                >
                  <span>
                    <strong>{source.sourceProvider}</strong>
                    <small>
                      {source.tags.includes("listing-address-match")
                        ? "Address text matched"
                        : "Address candidate"}
                    </small>
                  </span>
                  <ExternalLink size={13} />
                </a>
              ))}
            </section>
          ) : null}

          <p className="photo-policy">
            Reusable photos are downloaded with attribution. Other search
            results stay linked to the original source.
          </p>
        </aside>
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
              {photo.tags.includes("reconstruction-excluded")
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

function artifactForSpace(
  sourceIds: string[],
  artifacts: CaseWorkspace["artifacts"],
) {
  const sources = new Set(sourceIds);
  return artifacts.find((artifact) =>
    (artifact.evidenceIds ?? [artifact.evidenceId]).some((id) =>
      sources.has(id),
    ),
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

function sameEvidenceSet(left?: string[], right?: string[]): boolean {
  if (!left || !right || left.length !== right.length) return false;
  const expected = new Set(left);
  return right.every((id) => expected.has(id));
}

function stageIcon(name: PipelineStageName) {
  if (name === "address_resolution") return <MapPin size={15} />;
  if (name === "evidence_discovery") return <ImageIcon size={15} />;
  return <Crosshair size={15} />;
}

function stageLabel(name: PipelineStageName): string {
  if (name === "address_resolution") return "Address";
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
): { title: string; detail: string } {
  if (readyArtifact)
    return {
      title: "Rescue View ready",
      detail: readyArtifact.fallback
        ? `${readyArtifact.gaussianCount?.toLocaleString() ?? "Gaussian"} scene from one exact source photo; the other captures did not register`
        : `${readyArtifact.gaussianCount?.toLocaleString() ?? "Gaussian"} scene from ${readyArtifact.registration?.connectedFrameCount ?? 1} connected ${readyArtifact.registration?.connectedFrameCount === 1 ? "capture" : "captures"}`,
    };
  if (currentArtifact?.status === "failed")
    return {
      title: "Rescue View could not be built",
      detail: "Add adjacent captures of the same space with 60-80% overlap.",
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
  if (running?.name === "evidence_discovery")
    return { title: "Searching for photos", detail: running.message };
  if (workspace.case.status === "failed")
    return {
      title: "Address lookup failed",
      detail: "Check the address and try again.",
    };
  return {
    title: "Captures needed",
    detail:
      "No reusable online photo was found yet. Add overlapping responder photos.",
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
