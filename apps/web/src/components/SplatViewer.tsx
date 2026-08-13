import {
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  Camera,
  Crosshair,
  ChevronsDown,
  ChevronsUp,
  Expand,
  Cpu,
  RefreshCcw,
  RotateCcw,
  TriangleAlert,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import type {
  ReconstructionArtifact,
  ReconstructionCameraPose,
} from "@structurefirst/contracts";
import { nextRenderScale } from "../lib/render-performance";
import { api, type LiveDetection } from "../lib/api";

type ThreeModule = typeof import("three");

type NavigationAction =
  "forward" | "backward" | "left" | "right" | "up" | "down";

type ViewerControl = {
  update: () => boolean;
  setMovement: (action: NavigationAction, active: boolean) => void;
  nudge: (action: NavigationAction) => void;
  dispose: () => void;
};

type Engine = {
  scene: import("three").Scene;
  camera: import("three").PerspectiveCamera;
  renderer: import("three").WebGLRenderer;
  spark: import("@sparkjsdev/spark").SparkRenderer;
  splat: import("@sparkjsdev/spark").SplatMesh;
  controls: ViewerControl;
  initialPosition: import("three").Vector3;
  initialQuaternion: import("three").Quaternion;
  THREE: ThreeModule;
};

export function SplatViewer({
  artifact,
  focusEvidenceId,
}: {
  artifact: ReconstructionArtifact;
  focusEvidenceId?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameRef = useRef<HTMLDivElement>(null);
  const engineRef = useRef<Engine | null>(null);
  const detectorEnabledRef = useRef(true);
  const detectionBusyRef = useRef(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string>();
  const [attempt, setAttempt] = useState(0);
  const [gpuRenderer, setGpuRenderer] = useState("");
  const [framesPerSecond, setFramesPerSecond] = useState(0);
  const [renderScale, setRenderScale] = useState(1);
  const [cameraPosition, setCameraPosition] = useState<
    [number, number, number]
  >([0, 0, 0]);
  const [selectedSourceIndex, setSelectedSourceIndex] = useState(0);
  const [showSourceViews, setShowSourceViews] = useState(false);
  const [detectorEnabled, setDetectorEnabled] = useState(true);
  const [detections, setDetections] = useState<LiveDetection[]>([]);
  const [detectorStatus, setDetectorStatus] = useState<{
    model: string;
    provider: string;
    inferenceMs: number;
  }>();
  const [detectorError, setDetectorError] = useState<string>();
  const sceneMode = artifact.mode === "panorama" ? "panorama" : "image";
  const sourceCameras = artifact.geometry?.cameraPoses ?? [];

  useEffect(() => {
    detectorEnabledRef.current = detectorEnabled;
    if (!detectorEnabled) {
      setDetections([]);
      setDetectorError(undefined);
    }
  }, [detectorEnabled]);

  useEffect(() => {
    const splatUrl = artifact.splatUrl;
    if (!splatUrl || !canvasRef.current) return;
    let cancelled = false;
    let pending: Partial<Engine> = {};
    setProgress(0);
    setError(undefined);
    setGpuRenderer("");
    setFramesPerSecond(0);
    setRenderScale(1);
    setCameraPosition([0, 0, 0]);
    setSelectedSourceIndex(0);

    void (async () => {
      try {
        const [THREE, sparkModule] = await Promise.all([
          import("three"),
          import("@sparkjsdev/spark"),
        ]);
        if (cancelled || !canvasRef.current) return;

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(
          60,
          1,
          0.01,
          sceneMode === "image" ? 1_000 : 250,
        );
        // LucidFrame SHARP exports OpenCV camera axes: +X right, +Y down,
        // +Z forward. These values preserve the exact source orientation.
        camera.up.set(0, -1, 0);
        camera.position.set(0, 0, 0);
        pointCamera(camera, THREE, 0, 0);

        const renderer = new THREE.WebGLRenderer({
          canvas: canvasRef.current,
          // Spark renders analytic Gaussian footprints; WebGL MSAA does not
          // improve them and costs substantial fill-rate at native DPR.
          antialias: false,
          alpha: false,
          premultipliedAlpha: true,
          powerPreference: "high-performance",
        });
        renderer.setClearColor(0x171b22, 1);
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        const gl = renderer.getContext();
        const debugRenderer = gl.getExtension("WEBGL_debug_renderer_info") as {
          UNMASKED_RENDERER_WEBGL: number;
          UNMASKED_VENDOR_WEBGL: number;
        } | null;
        const rendererName = String(
          gl.getParameter(
            debugRenderer?.UNMASKED_RENDERER_WEBGL ?? gl.RENDERER,
          ),
        );
        renderer.domElement.dataset.gpuRenderer = rendererName;
        renderer.domElement.dataset.gpuVendor = String(
          gl.getParameter(debugRenderer?.UNMASKED_VENDOR_WEBGL ?? gl.VENDOR),
        );
        setGpuRenderer(rendererName);
        pending = { scene, camera, renderer };

        const spark = new sparkModule.SparkRenderer({
          renderer,
          // SHARP exports are trained and previewed with depth-ordered alpha
          // compositing. Keep their native footprint: pre-blur enlarges every
          // Gaussian and makes independently reconstructed views bleed together.
          // Multi-view scenes are meant to be rotated and traversed. Spark's
          // radial metric is more stable under viewpoint rotation; a single
          // SHARP source retains its source-trained Z-depth sorting.
          sortRadial: artifact.mode === "multi_image",
          preBlurAmount: 0,
          blurAmount: artifact.mode === "multi_image" ? 0.12 : 0,
          maxStdDev:
            artifact.mode === "multi_image" ? Math.sqrt(9) : Math.sqrt(8),
          minAlpha:
            artifact.mode === "multi_image" ? 0.25 / 255 : 0.5 / 255,
          minSortIntervalMs:
            sceneMode === "panorama"
              ? 50
              : isIntegratedRenderer(rendererName)
                ? 33
                : 16,
          enableLod: false,
        });
        scene.add(spark);
        pending.spark = spark;

        const splat = new sparkModule.SplatMesh({
          url: splatUrl,
          lod: false,
          nonLod: true,
          enableLod: false,
          onProgress: (event: ProgressEvent) => {
            if (!cancelled && event.lengthComputable && event.total > 0) {
              setProgress(Math.min(0.98, event.loaded / event.total));
            }
          },
        });
        scene.add(splat);
        pending.splat = splat;
        await splat.initialized;
        if (cancelled) return;

        const engine: Engine = {
          scene,
          camera,
          renderer,
          spark,
          splat,
          controls: createNavigationControls(
            camera,
            renderer.domElement,
            THREE,
          ),
          initialPosition: camera.position.clone(),
          initialQuaternion: camera.quaternion.clone(),
          THREE,
        };
        engineRef.current = engine;
        const initialPose = sourceCameras.find(
          (pose) => pose.evidenceId === focusEvidenceId,
        );
        if (initialPose) {
          placeAtSourceCamera(engine, initialPose);
          setSelectedSourceIndex(initialPose.sourceIndex);
          setCameraPosition([...initialPose.position]);
        }
        pending = {};
        setProgress(1);
        canvasRef.current.dataset.splatReady = "true";
        canvasRef.current.dataset.gaussianCount = String(
          artifact.gaussianCount ?? "",
        );

        let renderedFrames = 0;
        let frameWindowStarted = performance.now();
        let activeRenderScale = 1;
        let lastMotionAt = frameWindowStarted;
        let lastDetectionAt = 0;
        const detectionCanvas = document.createElement("canvas");
        const render = () => {
          if (cancelled) return;
          resizeRenderer(renderer, camera, sceneMode, activeRenderScale);
          if (engine.controls.update()) lastMotionAt = performance.now();
          renderer.render(scene, camera);
          renderedFrames += 1;
          const now = performance.now();
          if (
            detectorEnabledRef.current &&
            now - lastDetectionAt >= 650 &&
            !detectionBusyRef.current
          ) {
            lastDetectionAt = now;
            detectionBusyRef.current = true;
            const sourceWidth = Math.max(1, renderer.domElement.width);
            const sourceHeight = Math.max(1, renderer.domElement.height);
            detectionCanvas.width = Math.min(640, sourceWidth);
            detectionCanvas.height = Math.max(
              1,
              Math.round(
                detectionCanvas.width * (sourceHeight / sourceWidth),
              ),
            );
            const context = detectionCanvas.getContext("2d", {
              alpha: false,
            });
            if (context) {
              context.drawImage(
                renderer.domElement,
                0,
                0,
                detectionCanvas.width,
                detectionCanvas.height,
              );
              const imageDataUrl = detectionCanvas.toDataURL("image/jpeg", 0.78);
              void api
                .detectFrame(imageDataUrl)
                .then((result) => {
                  if (cancelled || !detectorEnabledRef.current) return;
                  setDetections(result.detections);
                  setDetectorStatus({
                    model: result.model,
                    provider: result.provider,
                    inferenceMs: result.inferenceMs,
                  });
                  setDetectorError(undefined);
                })
                .catch((caught) => {
                  if (cancelled || !detectorEnabledRef.current) return;
                  setDetectorError(
                    caught instanceof Error
                      ? caught.message
                      : "Local detector unavailable.",
                  );
                })
                .finally(() => {
                  detectionBusyRef.current = false;
                });
            } else {
              detectionBusyRef.current = false;
            }
          }
          const frameWindowMs = now - frameWindowStarted;
          if (frameWindowMs >= 750) {
            const fps = (renderedFrames * 1_000) / frameWindowMs;
            const nextScale = nextRenderScale({
              currentScale: activeRenderScale,
              framesPerSecond: fps,
              moving: now - lastMotionAt < 260,
              integratedGpu: isIntegratedRenderer(rendererName),
            });
            if (nextScale !== activeRenderScale) {
              activeRenderScale = nextScale;
              setRenderScale(nextScale);
            }
            renderer.domElement.dataset.renderFps = fps.toFixed(1);
            setFramesPerSecond(fps);
            renderer.domElement.dataset.activeSplats = String(
              spark.activeSplats,
            );
            renderer.domElement.dataset.detailScale =
              activeRenderScale.toFixed(3);
            renderer.domElement.dataset.detailMode =
              activeRenderScale === 1 ? "native" : "motion-buffer";
            renderer.domElement.dataset.cameraPosition = [
              camera.position.x,
              camera.position.y,
              camera.position.z,
            ]
              .map((value) => value.toFixed(4))
              .join(",");
            setCameraPosition([
              camera.position.x,
              camera.position.y,
              camera.position.z,
            ]);
            renderedFrames = 0;
            frameWindowStarted = now;
          }
        };
        renderer.setAnimationLoop(render);
      } catch (caught) {
        if (!cancelled)
          setError(
            caught instanceof Error
              ? caught.message
              : "The Gaussian scene could not be opened.",
          );
      }
    })();

    return () => {
      cancelled = true;
      // Stop rendering synchronously before any Spark or Three resources are
      // disposed. This prevents a stale frame from using a torn-down renderer
      // while React mounts the next Gaussian artifact.
      engineRef.current?.renderer.setAnimationLoop(null);
      pending.renderer?.setAnimationLoop(null);
      if (canvasRef.current) {
        delete canvasRef.current.dataset.splatReady;
        delete canvasRef.current.dataset.gaussianCount;
        delete canvasRef.current.dataset.renderFps;
        delete canvasRef.current.dataset.cameraPosition;
        delete canvasRef.current.dataset.activeSplats;
        delete canvasRef.current.dataset.detailScale;
        delete canvasRef.current.dataset.detailMode;
        delete canvasRef.current.dataset.gpuRenderer;
        delete canvasRef.current.dataset.gpuVendor;
      }
      const engine = engineRef.current;
      engineRef.current = null;
      detectionBusyRef.current = false;
      if (engine) {
        void disposeEngine(engine);
      } else {
        try {
          pending.splat?.dispose();
          pending.spark?.dispose();
          pending.renderer?.dispose();
        } catch {
          // Spark can still be completing its first worker sort during unmount.
        }
      }
    };
  }, [artifact.id, artifact.splatUrl, attempt, sceneMode]);

  useEffect(() => {
    if (!focusEvidenceId) return;
    const engine = engineRef.current;
    const pose = sourceCameras.find(
      (candidate) => candidate.evidenceId === focusEvidenceId,
    );
    if (!engine || !pose) return;
    placeAtSourceCamera(engine, pose);
    setCameraPosition([...pose.position]);
    setSelectedSourceIndex(pose.sourceIndex);
  }, [artifact.id, focusEvidenceId]);

  function resetCamera() {
    const engine = engineRef.current;
    if (!engine) return;
    engine.controls.dispose();
    engine.camera.position.copy(engine.initialPosition);
    engine.camera.quaternion.copy(engine.initialQuaternion);
    engine.controls = createNavigationControls(
      engine.camera,
      engine.renderer.domElement,
      engine.THREE,
    );
    engine.renderer.domElement.focus({ preventScroll: true });
  }

  function setMovement(action: NavigationAction, active: boolean) {
    engineRef.current?.controls.setMovement(action, active);
  }

  function nudge(action: NavigationAction) {
    engineRef.current?.controls.nudge(action);
  }

  function focusSourceCamera(pose: ReconstructionCameraPose) {
    const engine = engineRef.current;
    if (!engine) return;
    placeAtSourceCamera(engine, pose);
    setCameraPosition([...pose.position]);
    setSelectedSourceIndex(pose.sourceIndex);
  }

  const captureCoverage = captureCoverageStatus(sourceCameras, cameraPosition);
  const crossViewSupportedRatio = artifact.quality?.crossViewSupportedRatio;

  return (
    <div className="splat-frame" ref={frameRef}>
      <canvas
        ref={canvasRef}
        tabIndex={0}
        aria-label="Interactive LucidFrame Gaussian reconstruction. Drag to look and use WASD to move."
      />
      {detectorEnabled ? (
        <div className="viewer-detection-layer" aria-live="polite">
          {detections.map((detection, index) => (
            <div
              className={`viewer-detection-box detection-${detection.priority}`}
              key={`${detection.label}-${index}`}
              style={{
                left: `${detection.x * 100}%`,
                top: `${detection.y * 100}%`,
                width: `${detection.width * 100}%`,
                height: `${detection.height * 100}%`,
              }}
            >
              <span>
                {detection.label} {Math.round(detection.score * 100)}%
              </span>
              {detection.priority !== "standard" ? (
                <strong>{detection.tacticalLabel}</strong>
              ) : null}
            </div>
          ))}
        </div>
      ) : null}
      {progress < 1 && !error ? (
        <div className="viewer-loading" role="status">
          <span>Loading Gaussian scene</span>
          <progress max={1} value={progress} />
          <strong>{Math.round(progress * 100)}%</strong>
        </div>
      ) : null}
      {error ? (
        <div className="viewer-error" role="alert">
          <TriangleAlert size={24} />
          <strong>Scene unavailable</strong>
          <span>{error}</span>
          <button
            type="button"
            onClick={() => setAttempt((value) => value + 1)}
          >
            <RefreshCcw size={14} /> Retry scene
          </button>
        </div>
      ) : null}
      <div className="viewer-tools">
        <button
          type="button"
          aria-pressed={detectorEnabled}
          onClick={() => setDetectorEnabled((enabled) => !enabled)}
          title="Toggle local YOLO26 view-space boxes"
        >
          <Crosshair size={16} /> {detectorEnabled ? "Boxes on" : "Boxes off"}
        </button>
        {sourceCameras.length > 1 ? (
          <button
            type="button"
            aria-pressed={showSourceViews}
            onClick={() => setShowSourceViews((visible) => !visible)}
            title="Show calibrated source-camera jump controls"
          >
            <Camera size={16} /> Camera views {sourceCameras.length}
          </button>
        ) : null}
        <button type="button" onClick={resetCamera} title="Reset camera">
          <RotateCcw size={16} /> Reset
        </button>
        <button
          type="button"
          onClick={() => void frameRef.current?.requestFullscreen()}
          title="Full screen"
        >
          <Expand size={16} /> Full screen
        </button>
      </div>
      <div className="viewer-mode">Rescue View · free flight</div>
      {gpuRenderer ? (
        <div
          className={`viewer-gpu ${isIntegratedRenderer(gpuRenderer) ? "viewer-gpu-warning" : ""}`}
          title={gpuRenderer}
        >
          {isIntegratedRenderer(gpuRenderer) ? (
            <TriangleAlert size={13} />
          ) : (
            <Cpu size={13} />
          )}
          <span>
            All splats · {shortGpuName(gpuRenderer)} ·{" "}
            {framesPerSecond > 0 ? `${Math.round(framesPerSecond)} FPS` : "…"}
          </span>
          <small>
            {renderScale === 1
              ? "Native pixels · no LOD"
              : `${Math.round(renderScale * 100)}% motion buffer · native when still`}
          </small>
          {isIntegratedRenderer(gpuRenderer) ? (
            <strong>Use the RTX 4080 for smooth navigation</strong>
          ) : null}
        </div>
      ) : null}
      {sourceCameras.length > 1 ? (
        <div className="viewer-camera-panel">
          <header>
            <span>Camera map</span>
            <strong>{sourceCameras.length} calibrated</strong>
          </header>
          <CameraMap
            cameras={sourceCameras}
            currentPosition={cameraPosition}
            selectedSourceIndex={selectedSourceIndex}
            onCameraSelect={focusSourceCamera}
          />
          <small>Arrow = current view · triangles = source cameras</small>
          {showSourceViews ? (
            <div className="viewer-camera-bookmarks">
              {sourceCameras.map((pose) => (
                <button
                  type="button"
                  key={pose.evidenceId}
                  aria-pressed={pose.sourceIndex === selectedSourceIndex}
                  title={`${Math.round(pose.confidenceScore * 100)}% camera calibration`}
                  onClick={() => focusSourceCamera(pose)}
                >
                  View {pose.sourceIndex + 1}
                </button>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}
      {captureCoverage?.outside ? (
        <div className="viewer-coverage-warning" role="status">
          <TriangleAlert size={14} />
          <span>
            Outside measured capture coverage. Unseen surfaces may open into
            holes.
          </span>
          <button type="button" onClick={resetCamera}>
            Return to source view
          </button>
        </div>
      ) : null}
      {detectorEnabled ? (
        <div
          className={`viewer-detector-status ${detectorError ? "viewer-detector-error" : ""}`}
        >
          <strong>
            {detectorError
              ? "Detector unavailable"
              : detectorStatus
                ? `${detectorStatus.model} · ${detections.length} view boxes`
                : "Starting local detector…"}
          </strong>
          <span>
            {detectorError ??
              (detectorStatus
                ? `${detectorStatus.provider} · ${Math.round(detectorStatus.inferenceMs)} ms · 2D observations, not verified hazards`
                : "Analyzing the rendered view locally")}
          </span>
        </div>
      ) : null}
      <div className="viewer-navigation" aria-label="Rescue View movement">
        <button
          type="button"
          className="nav-forward"
          aria-label="Move forward"
          {...movementButton("forward", setMovement, nudge)}
        >
          <ArrowUp size={16} />
        </button>
        <button
          type="button"
          className="nav-left"
          aria-label="Move left"
          {...movementButton("left", setMovement, nudge)}
        >
          <ArrowLeft size={16} />
        </button>
        <button
          type="button"
          className="nav-back"
          aria-label="Move backward"
          {...movementButton("backward", setMovement, nudge)}
        >
          <ArrowDown size={16} />
        </button>
        <button
          type="button"
          className="nav-right"
          aria-label="Move right"
          {...movementButton("right", setMovement, nudge)}
        >
          <ArrowRight size={16} />
        </button>
        <button
          type="button"
          className="nav-up"
          aria-label="Move up"
          title="Move up"
          {...movementButton("up", setMovement, nudge)}
        >
          <ChevronsUp size={15} />
        </button>
        <button
          type="button"
          className="nav-down"
          aria-label="Move down"
          title="Move down"
          {...movementButton("down", setMovement, nudge)}
        >
          <ChevronsDown size={15} />
        </button>
      </div>
      <div className="viewer-hint">
        Drag to look · scroll or WASD to move · Q/E height
      </div>
      <div className="viewer-provenance">
        <span>
          {artifact.fallback
            ? "Single exact source photo · overlap fallback"
            : artifact.mode === "multi_image" && artifact.registration
              ? `${artifact.registration.connectedFrameCount}/${artifact.registration.frameCount} captures · ${Math.round(artifact.registration.confidenceScore * 100)}% registration`
              : "Exact LucidFrame source view"}
        </span>
        <strong>
          {artifact.gaussianCount?.toLocaleString() ?? "—"} Gaussians
        </strong>
        {artifact.geometry?.backend === "vggt_sharp_joint" ? (
          <em>VGGT cameras + LucidFrame SHARP</em>
        ) : null}
        {crossViewSupportedRatio !== undefined ? (
          <em title="Gaussians whose depth agrees with at least one other measured source view">
            {Math.round(crossViewSupportedRatio * 100)}% cross-view supported
          </em>
        ) : null}
      </div>
    </div>
  );
}

function placeAtSourceCamera(engine: Engine, pose: ReconstructionCameraPose) {
  engine.controls.dispose();
  engine.camera.position.set(...pose.position);
  const [w, x, y, z] = pose.rotationWxyz;
  const sourceRotation = new engine.THREE.Quaternion(x, y, z, w).normalize();
  engine.camera.quaternion
    .copy(sourceRotation)
    .multiply(engine.initialQuaternion);
  engine.controls = createNavigationControls(
    engine.camera,
    engine.renderer.domElement,
    engine.THREE,
  );
  engine.renderer.domElement.focus({ preventScroll: true });
}

function CameraMap({
  cameras,
  currentPosition,
  selectedSourceIndex,
  onCameraSelect,
}: {
  cameras: ReconstructionCameraPose[];
  currentPosition: [number, number, number];
  selectedSourceIndex: number;
  onCameraSelect: (camera: ReconstructionCameraPose) => void;
}) {
  const points = [...cameras.map((camera) => camera.position), currentPosition];
  const minX = Math.min(...points.map((point) => point[0]));
  const maxX = Math.max(...points.map((point) => point[0]));
  const minZ = Math.min(...points.map((point) => point[2]));
  const maxZ = Math.max(...points.map((point) => point[2]));
  const spanX = Math.max(0.5, maxX - minX);
  const spanZ = Math.max(0.5, maxZ - minZ);
  const project = (position: [number, number, number]) =>
    [
      10 + ((position[0] - minX) / spanX) * 140,
      86 - ((position[2] - minZ) / spanZ) * 76,
    ] as const;
  const current = project(currentPosition);
  return (
    <svg
      className="viewer-camera-map"
      viewBox="0 0 160 96"
      role="img"
      aria-label="Top-down map of calibrated source cameras and current viewer position"
    >
      <path d="M10 86H150M10 48H150M10 10H150" />
      <path d="M10 10V86M80 10V86M150 10V86" />
      {cameras.map((camera) => {
        const [x, y] = project(camera.position);
        const [directionX, directionY] = cameraDirection2d(
          camera.rotationWxyz,
        );
        const tipX = x + directionX * 7;
        const tipY = y + directionY * 7;
        const backX = x - directionX * 4;
        const backY = y - directionY * 4;
        const sideX = -directionY * 4;
        const sideY = directionX * 4;
        return (
          <g
            key={camera.evidenceId}
            role="button"
            tabIndex={0}
            aria-label={`Jump to calibrated source view ${camera.sourceIndex + 1}`}
            onClick={() => onCameraSelect(camera)}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                onCameraSelect(camera);
              }
            }}
          >
            <polygon
              className={
                camera.sourceIndex === selectedSourceIndex
                  ? "viewer-selected-camera"
                  : undefined
              }
              points={`${tipX},${tipY} ${backX + sideX},${backY + sideY} ${backX - sideX},${backY - sideY}`}
            />
            <text x={x} y={y + 2}>
              {camera.sourceIndex + 1}
            </text>
          </g>
        );
      })}
      <path
        className="viewer-current-position"
        d={`M${current[0]},${current[1] - 5} L${current[0] + 4},${current[1] + 4} L${current[0]},${current[1] + 2} L${current[0] - 4},${current[1] + 4} Z`}
      />
    </svg>
  );
}

function cameraDirection2d(rotationWxyz: [number, number, number, number]) {
  const [w, x, y, z] = rotationWxyz;
  // Rotate the OpenCV camera's local +Z forward vector by the solved camera
  // quaternion, then project it into this top-down X/Z capture layout.
  const forwardX = 2 * (x * z + w * y);
  const forwardZ = 1 - 2 * (x * x + y * y);
  const length = Math.hypot(forwardX, forwardZ) || 1;
  return [forwardX / length, -forwardZ / length] as const;
}

function captureCoverageStatus(
  cameras: ReconstructionCameraPose[],
  currentPosition: [number, number, number],
) {
  if (cameras.length < 2) return undefined;
  const distances = cameras.map((camera) =>
    distance3(camera.position, currentPosition),
  );
  const nearestCameraSpacing = cameras.map((camera, index) =>
    Math.min(
      ...cameras
        .filter((_, candidateIndex) => candidateIndex !== index)
        .map((candidate) => distance3(camera.position, candidate.position)),
    ),
  );
  const sortedSpacing = nearestCameraSpacing
    .filter(Number.isFinite)
    .sort((left, right) => left - right);
  const medianSpacing =
    sortedSpacing[Math.floor(sortedSpacing.length / 2)] ?? 0;
  const measuredRadius = Math.max(2, medianSpacing * 3);
  const nearestDistance = Math.min(...distances);
  return {
    outside: nearestDistance > measuredRadius,
    nearestDistance,
    measuredRadius,
  };
}

function distance3(
  left: [number, number, number],
  right: [number, number, number],
) {
  return Math.hypot(left[0] - right[0], left[1] - right[1], left[2] - right[2]);
}

function isIntegratedRenderer(renderer: string) {
  return /Radeon\(TM\) 610M|Intel|UHD|Iris|SwiftShader|Microsoft Basic/i.test(
    renderer,
  );
}

function shortGpuName(renderer: string) {
  const nvidia = renderer.match(/NVIDIA (?:GeForce )?(?:RTX|GTX) [^,(]+/i);
  if (nvidia) return nvidia[0].replace(/^NVIDIA\s*/i, "").trim();
  const amd = renderer.match(/AMD Radeon(?:\(TM\))?\s+[^,(]+/i);
  if (amd) return amd[0].replace(/^AMD\s*/i, "").trim();
  const intel = renderer.match(/Intel[^,)]*/i);
  if (intel) return intel[0].trim();
  return (
    renderer
      .split(",")[0]
      ?.replace(/^ANGLE\s*\(/, "")
      .trim() || "GPU"
  );
}

function createNavigationControls(
  camera: import("three").PerspectiveCamera,
  canvas: HTMLCanvasElement,
  THREE: ThreeModule,
): ViewerControl {
  const direction = camera.getWorldDirection(new THREE.Vector3());
  let pitch = Math.asin(Math.max(-1, Math.min(1, direction.y)));
  let yaw = Math.atan2(direction.x, direction.z);
  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  let lastFrame = performance.now();
  const pressed = new Set<string>();
  const movement = new Set<NavigationAction>();
  const nudges = new Map<NavigationAction, number>();
  const velocity = new THREE.Vector3();
  const targetVelocity = new THREE.Vector3();

  const pointerDown = (event: PointerEvent) => {
    if (event.button !== 0) return;
    dragging = true;
    lastX = event.clientX;
    lastY = event.clientY;
    canvas.focus({ preventScroll: true });
    canvas.setPointerCapture(event.pointerId);
    canvas.style.cursor = "grabbing";
  };
  const pointerMove = (event: PointerEvent) => {
    if (!dragging) return;
    yaw -= (event.clientX - lastX) * 0.004;
    pitch -= (event.clientY - lastY) * 0.004;
    pitch = Math.max(-1.48, Math.min(1.48, pitch));
    lastX = event.clientX;
    lastY = event.clientY;
    pointCamera(camera, THREE, pitch, yaw);
  };
  const pointerUp = (event: PointerEvent) => {
    dragging = false;
    if (canvas.hasPointerCapture(event.pointerId))
      canvas.releasePointerCapture(event.pointerId);
    canvas.style.cursor = "grab";
  };
  const keyDown = (event: KeyboardEvent) => {
    if (document.activeElement !== canvas) return;
    if (
      [
        "KeyW",
        "KeyA",
        "KeyS",
        "KeyD",
        "KeyQ",
        "KeyE",
        "ShiftLeft",
        "ShiftRight",
      ].includes(event.code)
    ) {
      event.preventDefault();
      pressed.add(event.code);
    }
  };
  const keyUp = (event: KeyboardEvent) => pressed.delete(event.code);
  const wheel = (event: WheelEvent) => {
    event.preventDefault();
    const action = event.deltaY < 0 ? "forward" : "backward";
    nudges.set(action, (nudges.get(action) ?? 0) + 0.22);
  };
  const blur = () => {
    pressed.clear();
    movement.clear();
  };

  canvas.style.cursor = "grab";
  canvas.addEventListener("pointerdown", pointerDown);
  canvas.addEventListener("pointermove", pointerMove);
  canvas.addEventListener("pointerup", pointerUp);
  canvas.addEventListener("pointercancel", pointerUp);
  canvas.addEventListener("wheel", wheel, { passive: false });
  window.addEventListener("keydown", keyDown);
  window.addEventListener("keyup", keyUp);
  window.addEventListener("blur", blur);

  return {
    update: () => {
      const now = performance.now();
      const delta = Math.min((now - lastFrame) / 1_000, 0.05);
      lastFrame = now;
      const forward =
        Number(pressed.has("KeyW") || movement.has("forward")) -
        Number(pressed.has("KeyS") || movement.has("backward"));
      const right =
        Number(pressed.has("KeyD") || movement.has("right")) -
        Number(pressed.has("KeyA") || movement.has("left"));
      const vertical =
        Number(pressed.has("KeyE") || movement.has("up")) -
        Number(pressed.has("KeyQ") || movement.has("down"));
      const nudgeForward =
        (nudges.get("forward") ?? 0) - (nudges.get("backward") ?? 0);
      const nudgeRight = (nudges.get("right") ?? 0) - (nudges.get("left") ?? 0);
      const nudgeVertical = (nudges.get("up") ?? 0) - (nudges.get("down") ?? 0);
      const hasNudge =
        nudgeForward !== 0 || nudgeRight !== 0 || nudgeVertical !== 0;
      nudges.clear();
      const length = Math.hypot(forward, right, vertical) || 1;
      const speed =
        1.1 * (pressed.has("ShiftLeft") || pressed.has("ShiftRight") ? 2.4 : 1);
      targetVelocity.set(
        (forward / length) * speed * Math.sin(yaw) +
          (right / length) * speed * Math.cos(yaw),
        -(vertical / length) * speed,
        (forward / length) * speed * Math.cos(yaw) -
          (right / length) * speed * Math.sin(yaw),
      );
      const hasContinuousInput = forward !== 0 || right !== 0 || vertical !== 0;
      const smoothing = 1 - Math.exp(-delta * (hasContinuousInput ? 12 : 9));
      velocity.lerp(targetVelocity, smoothing);
      if (!hasContinuousInput && velocity.lengthSq() < 0.000001)
        velocity.set(0, 0, 0);
      camera.position.addScaledVector(velocity, delta);

      const forwardDistance = nudgeForward;
      const rightDistance = nudgeRight;
      const verticalDistance = nudgeVertical;
      camera.position.x +=
        forwardDistance * Math.sin(yaw) + rightDistance * Math.cos(yaw);
      camera.position.y -= verticalDistance;
      camera.position.z +=
        forwardDistance * Math.cos(yaw) - rightDistance * Math.sin(yaw);
      return (
        dragging ||
        hasContinuousInput ||
        hasNudge ||
        velocity.lengthSq() > 0.000001
      );
    },
    setMovement: (action, active) => {
      if (active) movement.add(action);
      else movement.delete(action);
    },
    nudge: (action) => nudges.set(action, (nudges.get(action) ?? 0) + 0.2),
    dispose: () => {
      canvas.removeEventListener("pointerdown", pointerDown);
      canvas.removeEventListener("pointermove", pointerMove);
      canvas.removeEventListener("pointerup", pointerUp);
      canvas.removeEventListener("pointercancel", pointerUp);
      canvas.removeEventListener("wheel", wheel);
      window.removeEventListener("keydown", keyDown);
      window.removeEventListener("keyup", keyUp);
      window.removeEventListener("blur", blur);
      canvas.style.cursor = "";
      pressed.clear();
      movement.clear();
      nudges.clear();
      velocity.set(0, 0, 0);
    },
  };
}

function movementButton(
  action: NavigationAction,
  setMovement: (action: NavigationAction, active: boolean) => void,
  nudge: (action: NavigationAction) => void,
) {
  return {
    onPointerDown: (event: ReactPointerEvent<HTMLButtonElement>) => {
      event.currentTarget.setPointerCapture(event.pointerId);
      setMovement(action, true);
    },
    onPointerUp: (event: ReactPointerEvent<HTMLButtonElement>) => {
      if (event.currentTarget.hasPointerCapture(event.pointerId))
        event.currentTarget.releasePointerCapture(event.pointerId);
      setMovement(action, false);
    },
    onPointerCancel: () => setMovement(action, false),
    onLostPointerCapture: () => setMovement(action, false),
    onClick: () => nudge(action),
  };
}

function pointCamera(
  camera: import("three").PerspectiveCamera,
  THREE: ThreeModule,
  pitch: number,
  yaw: number,
) {
  const cosPitch = Math.cos(pitch);
  camera.lookAt(
    new THREE.Vector3(
      camera.position.x + Math.sin(yaw) * cosPitch,
      camera.position.y + Math.sin(pitch),
      camera.position.z + Math.cos(yaw) * cosPitch,
    ),
  );
}

function resizeRenderer(
  renderer: import("three").WebGLRenderer,
  camera: import("three").PerspectiveCamera,
  sceneMode: "image" | "panorama",
  renderScale: number,
) {
  const canvas = renderer.domElement;
  // Keep every Gaussian resident and sorted. Only the pixel buffer adapts
  // during motion, then returns to native DPR when the camera settles.
  const pixelRatio = (window.devicePixelRatio || 1) * renderScale;
  const width = Math.max(1, Math.round(canvas.clientWidth));
  const height = Math.max(1, Math.round(canvas.clientHeight));
  if (
    renderer.getPixelRatio() !== pixelRatio ||
    canvas.width !== Math.round(width * pixelRatio) ||
    canvas.height !== Math.round(height * pixelRatio)
  ) {
    renderer.setPixelRatio(pixelRatio);
    renderer.setSize(width, height, false);
  }
  camera.aspect = width / height;
  const horizontalFov = sceneMode === "panorama" ? 78 : 62;
  camera.fov =
    (2 *
      Math.atan(
        Math.tan((horizontalFov * Math.PI) / 360) /
          Math.max(camera.aspect, 1e-6),
      ) *
      180) /
    Math.PI;
  camera.updateProjectionMatrix();
}

async function disposeEngine(engine: Engine): Promise<void> {
  engine.controls.dispose();
  engine.scene.remove(engine.splat);
  engine.scene.remove(engine.spark);
  const sortable = engine.spark as unknown as {
    sorting?: boolean;
    autoUpdate?: boolean;
    sortDirty?: boolean;
    lodDirty?: boolean;
  };
  sortable.autoUpdate = false;
  sortable.sortDirty = false;
  sortable.lodDirty = false;
  while (sortable.sorting) {
    await new Promise((resolve) => window.setTimeout(resolve, 16));
  }
  try {
    engine.splat.dispose();
    engine.spark.dispose();
  } finally {
    engine.renderer.dispose();
  }
}
