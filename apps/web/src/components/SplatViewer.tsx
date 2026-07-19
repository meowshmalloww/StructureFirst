import {
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
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
import type { ReconstructionArtifact } from "@structurefirst/contracts";

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
}: {
  artifact: ReconstructionArtifact;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameRef = useRef<HTMLDivElement>(null);
  const engineRef = useRef<Engine | null>(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string>();
  const [attempt, setAttempt] = useState(0);
  const [gpuRenderer, setGpuRenderer] = useState("");
  const sceneMode = artifact.mode === "panorama" ? "panorama" : "image";

  useEffect(() => {
    const splatUrl = artifact.splatUrl;
    if (!splatUrl || !canvasRef.current) return;
    let cancelled = false;
    let animationFrame = 0;
    let pending: Partial<Engine> = {};
    setProgress(0);
    setError(undefined);
    setGpuRenderer("");

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
          // Rescue View favors source fidelity. Z-depth sorting matches the
          // training convention, no pre-blur is added, and the full Gaussian
          // footprint is retained.
          sortRadial: false,
          preBlurAmount: 0,
          maxStdDev: Math.sqrt(8),
          minAlpha: 0.5 / 255,
          minSortIntervalMs: sceneMode === "panorama" ? 80 : 20,
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
        pending = {};
        setProgress(1);
        canvasRef.current.dataset.splatReady = "true";
        canvasRef.current.dataset.gaussianCount = String(
          artifact.gaussianCount ?? "",
        );

        let renderedFrames = 0;
        let frameWindowStarted = performance.now();
        const render = () => {
          if (cancelled) return;
          resizeRenderer(renderer, camera, sceneMode);
          engine.controls.update();
          renderer.render(scene, camera);
          renderedFrames += 1;
          const now = performance.now();
          const frameWindowMs = now - frameWindowStarted;
          if (frameWindowMs >= 750) {
            const fps = (renderedFrames * 1_000) / frameWindowMs;
            renderer.domElement.dataset.renderFps = fps.toFixed(1);
            renderer.domElement.dataset.activeSplats = String(
              spark.activeSplats,
            );
            renderer.domElement.dataset.detailScale = "1.000";
            renderer.domElement.dataset.detailMode = "full";
            renderer.domElement.dataset.cameraPosition = [
              camera.position.x,
              camera.position.y,
              camera.position.z,
            ]
              .map((value) => value.toFixed(4))
              .join(",");
            renderedFrames = 0;
            frameWindowStarted = now;
          }
          animationFrame = requestAnimationFrame(render);
        };
        render();
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
      cancelAnimationFrame(animationFrame);
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

  return (
    <div className="splat-frame" ref={frameRef}>
      <canvas
        ref={canvasRef}
        tabIndex={0}
        aria-label="Interactive LucidFrame Gaussian reconstruction. Drag to look and use WASD to move."
      />
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
          <span>Full detail · {shortGpuName(gpuRenderer)}</span>
          {isIntegratedRenderer(gpuRenderer) ? (
            <strong>Use the RTX 4080 for smooth navigation</strong>
          ) : null}
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
      </div>
    </div>
  );
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
      return hasContinuousInput || hasNudge || velocity.lengthSq() > 0.000001;
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
) {
  const canvas = renderer.domElement;
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 1.75);
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
    engine.spark.dispose();
    engine.splat.dispose();
  } finally {
    engine.renderer.dispose();
    engine.renderer.forceContextLoss();
  }
}
