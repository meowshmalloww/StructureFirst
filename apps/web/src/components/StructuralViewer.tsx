import {
  Camera,
  DoorOpen,
  Home,
  Layers3,
  LoaderCircle,
  Maximize2,
  RotateCcw,
  TriangleAlert,
} from "lucide-react";
import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
} from "react";
import type { ReconstructionArtifact } from "@structurefirst/contracts";
import {
  roomTypeFromPlanLabel,
  type PlanRoomCandidate,
  type PlanRoomResolution,
  type PlanRoomSelection,
} from "../lib/room-availability";

type Wall = {
  id: string;
  start: [number, number];
  end: [number, number];
  heightMeters: number;
  thicknessMeters: number;
  confidence: number;
};

export type StructuralRoomSelection = PlanRoomSelection;

type PlanRegion = {
  id: string;
  label: string;
  polygon?: Array<[number, number]>;
  confidence?: number;
  labelSource?: "plan_ocr" | "geometry_only";
};

type PlanLabel = {
  id: string;
  label: string;
  position: [number, number];
  confidence: number;
  source: "plan_ocr";
};

type Floor = {
  index: number;
  floorNumber: number;
  label: string;
  elevationMeters: number;
  sourceEvidenceId: string;
  sourceImageUrl: string;
  planOverlayUrl?: string;
  planImageUrl?: string;
  imageSize: { width: number; height: number };
  textureCrop: { left: number; top: number; right: number; bottom: number };
  widthMeters: number;
  depthMeters: number;
  wallHeightMeters: number;
  walls: Wall[];
  rooms: PlanRegion[];
  footprint?: Array<[number, number]>;
  features?: Array<PlanRegion & { kind: "stairs" }>;
  planLabels?: PlanLabel[];
  spaceGraph?: {
    nodes: Array<{
      id: string;
      label: string;
      position: [number, number];
    }>;
    edges: Array<{
      id: string;
      from: string;
      to: string;
      kind: "inferred_traversable_path";
      verified: false;
      confidence: number;
    }>;
    connected: boolean;
    verified: false;
    method: string;
    note?: string;
  };
  vectorization?: {
    status: "candidate" | "source_plan_only";
    sourcePlanAuthoritative: boolean;
    safeForExtrusion: boolean;
    wallGeometryVerified: boolean;
    roomTopologyVerified: boolean;
    labeledSpaces: number;
    labelsInsideEnclosedZones: number;
    labelEnclosureRatio: number;
    reason: string;
  };
  metrics: {
    wallSegments: number;
    enclosedZones: number;
    stairFeatures?: number;
    labeledSpaces?: number;
    candidateConnections?: number;
  };
};

type StructuralModel = {
  schemaVersion: number;
  floors: Floor[];
  referencePlans?: Array<{
    kind: "site_plan";
    label: string;
    structuralFloor: false;
    reason: string;
  }>;
  alignment?: {
    method: string;
    metricScaleSource: string;
    metricScaleVerified?: boolean;
    floorToFloorMeters: number;
    confidence: number;
    note: string;
  };
  roof: {
    type: "unobserved" | "flat_estimate";
    elevationMeters: number;
    confidence: number;
    note: string;
  };
  limitations: string[];
  metrics: { floorCount: number; wallSegments: number; enclosedZones: number };
};

type SceneEngine = {
  renderer: import("three").WebGLRenderer;
  floorGroups: import("three").Group[];
  roomAnchors: import("three").Object3D[];
  reset: () => void;
  focusFloor: (floorIndex: number) => void;
  setViewMode: (mode: ViewMode, floorIndex: number | "all") => void;
  dispose: () => void;
};

type ViewMode = "plan" | "candidate";

type RoomEntry = {
  key: string;
  room: PlanLabel;
  floor: Floor;
  selection: StructuralRoomSelection;
  resolution: PlanRoomResolution;
};

type PlanViewport = {
  floorIndex: number;
  x: number;
  y: number;
  width: number;
  height: number;
};

export function StructuralViewer({
  artifact,
  onRoomSelect,
  roomAvailability,
}: {
  artifact: ReconstructionArtifact;
  onRoomSelect?: (
    selection: StructuralRoomSelection,
    candidate?: PlanRoomCandidate,
  ) => void;
  roomAvailability?: (selection: StructuralRoomSelection) => PlanRoomResolution;
}) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<SceneEngine | null>(null);
  const [model, setModel] = useState<StructuralModel>();
  const [error, setError] = useState<string>();
  const [selectedFloor, setSelectedFloor] = useState<number | "all">("all");
  const [viewMode, setViewMode] = useState<ViewMode>("plan");
  const [gpu, setGpu] = useState("");
  const [hoveredRoomKey, setHoveredRoomKey] = useState<string>();
  const [pinnedRoomKey, setPinnedRoomKey] = useState<string>();
  const [showRoomIndex, setShowRoomIndex] = useState(true);
  const [planViewport, setPlanViewport] = useState<PlanViewport>();
  const planPanRef = useRef<{
    pointerId: number;
    clientX: number;
    clientY: number;
    viewport: PlanViewport;
  } | undefined>(undefined);

  const allRoomEntries = useMemo<RoomEntry[]>(
    () =>
      (model?.floors ?? []).flatMap((floor) => {
        const labels = floor.planLabels ?? [];
        return labels.map((room) => {
          const requestedType = roomTypeFromPlanLabel(room.label);
          const sameTypeLabelCount = requestedType
            ? labels.filter(
                (candidate) =>
                  roomTypeFromPlanLabel(candidate.label) === requestedType,
              ).length
            : 1;
          const selection: StructuralRoomSelection = {
            label: room.label,
            floorNumber: floor.floorNumber,
            sourceEvidenceId: floor.sourceEvidenceId,
            sameTypeLabelCount,
          };
          const resolution = roomAvailability?.(selection) ?? {
            available: Boolean(onRoomSelect),
            detail: onRoomSelect ? "3D available" : "No 3D view",
            candidates: [],
          };
          return {
            key: `${floor.index}-${room.id}`,
            room,
            floor,
            selection,
            resolution,
          };
        });
      }),
    [model, onRoomSelect, roomAvailability],
  );

  useEffect(() => {
    if (!artifact.structuralModelUrl) return;
    const abort = new AbortController();
    setModel(undefined);
    setError(undefined);
    void fetch(artifact.structuralModelUrl, {
      signal: abort.signal,
      cache: "no-store",
    })
      .then(async (response) => {
        if (!response.ok)
          throw new Error(`Structural model returned ${response.status}.`);
        return (await response.json()) as StructuralModel;
      })
      .then((loadedModel) => {
        setModel(loadedModel);
        setSelectedFloor(loadedModel.floors[0]?.index ?? "all");
        setViewMode("plan");
      })
      .catch((caught) => {
        if (!abort.signal.aborted)
          setError(
            caught instanceof Error
              ? caught.message
              : "The structural model could not be opened.",
          );
      });
    return () => abort.abort();
  }, [artifact.structuralModelUrl]);

  useEffect(() => {
    if (!model || model.floors.length === 0 || !canvasRef.current) return;
    let cancelled = false;
    let animationFrame = 0;
    void (async () => {
      try {
        const [THREE, { OrbitControls }] = await Promise.all([
          import("three"),
          import("three/addons/controls/OrbitControls.js"),
        ]);
        const canvas = canvasRef.current;
        if (cancelled || !canvas) return;
        const renderer = new THREE.WebGLRenderer({
          canvas,
          antialias: true,
          powerPreference: "high-performance",
        });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.setClearColor(0x15191f, 1);
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        const gl = renderer.getContext();
        const debug = gl.getExtension("WEBGL_debug_renderer_info") as {
          UNMASKED_RENDERER_WEBGL: number;
        } | null;
        setGpu(
          String(
            gl.getParameter(debug?.UNMASKED_RENDERER_WEBGL ?? gl.RENDERER),
          ),
        );

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(52, 1, 0.05, 500);
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.screenSpacePanning = true;
        controls.minDistance = 0.25;
        controls.maxDistance = 120;

        scene.add(new THREE.HemisphereLight(0xe6edf5, 0x26313b, 2.2));
        const keyLight = new THREE.DirectionalLight(0xffffff, 2.1);
        keyLight.position.set(12, 24, 8);
        scene.add(keyLight);
        const grid = new THREE.GridHelper(50, 50, 0x4a5664, 0x29323c);
        grid.position.y = -0.04;
        scene.add(grid);

        const textureLoader = new THREE.TextureLoader();
        const roomAnchors: import("three").Object3D[] = [];
        const candidateGroups: import("three").Group[] = [];
        const floorGroups = await Promise.all(
          model.floors.map(async (floor) => {
            const group = new THREE.Group();
            group.userData.floorIndex = floor.index;
            const candidateGroup = new THREE.Group();
            candidateGroup.userData.candidateGeometry = true;
            candidateGroups.push(candidateGroup);
            const texture = await textureLoader.loadAsync(
              floor.planOverlayUrl ?? floor.sourceImageUrl,
            );
            texture.colorSpace = THREE.SRGBColorSpace;
            texture.wrapS = THREE.ClampToEdgeWrapping;
            texture.wrapT = THREE.ClampToEdgeWrapping;
            const crop = floor.textureCrop;
            texture.repeat.set(
              crop.right - crop.left,
              -(crop.bottom - crop.top),
            );
            texture.offset.set(crop.left, crop.bottom);

            const floorMaterial = new THREE.MeshStandardMaterial({
              color: 0x43515e,
              roughness: 0.92,
              transparent: true,
              opacity: 0.58,
              side: THREE.DoubleSide,
            });
            const regions = floor.rooms.length
              ? floor.rooms
              : floor.footprint?.length
                ? [
                    {
                      id: "footprint",
                      label: "Plan footprint",
                      polygon: floor.footprint,
                    },
                  ]
                : [];
            for (const room of regions) {
              if (!room.polygon || room.polygon.length < 3) continue;
              const region = new THREE.Mesh(
                polygonGeometry(THREE, room.polygon),
                floorMaterial,
              );
              region.position.y = floor.elevationMeters + 0.012;
              candidateGroup.add(region);
            }

            for (const planLabel of floor.planLabels ?? []) {
              const label = textSprite(THREE, planLabel.label);
              label.position.set(
                planLabel.position[0],
                floor.elevationMeters + 0.17,
                planLabel.position[1],
              );
              candidateGroup.add(label);
            }

            for (const entry of allRoomEntries.filter(
              (candidate) => candidate.floor.index === floor.index,
            )) {
              const color = roomStatusColor(entry.resolution);
              const marker = new THREE.Group();
              marker.userData.roomKey = entry.key;
              marker.position.set(
                entry.room.position[0],
                floor.elevationMeters + 0.13,
                entry.room.position[1],
              );
              const halo = new THREE.Mesh(
                new THREE.RingGeometry(0.2, 0.34, 32),
                new THREE.MeshBasicMaterial({
                  color,
                  transparent: true,
                  opacity: entry.resolution.available
                    ? 0.92
                    : entry.resolution.candidates.length
                      ? 0.8
                      : 0.42,
                  depthTest: false,
                  side: THREE.DoubleSide,
                }),
              );
              halo.rotation.x = -Math.PI / 2;
              halo.renderOrder = 7;
              marker.add(halo);
              const beacon = new THREE.Mesh(
                new THREE.CylinderGeometry(0.07, 0.07, 0.3, 18),
                new THREE.MeshBasicMaterial({
                  color,
                  transparent: true,
                  opacity: 0.9,
                  depthTest: false,
                }),
              );
              beacon.position.y = 0.15;
              beacon.renderOrder = 7;
              marker.add(beacon);
              group.add(marker);
              roomAnchors.push(marker);
            }

            for (const feature of floor.features ?? []) {
              if (!feature.polygon || feature.polygon.length < 3) continue;
              const marker = new THREE.Mesh(
                polygonGeometry(THREE, feature.polygon),
                new THREE.MeshBasicMaterial({
                  color: 0xf1b96b,
                  transparent: true,
                  opacity: 0.38,
                  side: THREE.DoubleSide,
                }),
              );
              marker.position.y = floor.elevationMeters + 0.075;
              group.add(marker);
            }

            const evidenceOverlay = new THREE.Mesh(
              new THREE.PlaneGeometry(floor.widthMeters, floor.depthMeters),
              new THREE.MeshBasicMaterial({
                map: texture,
                transparent: true,
                opacity: 0.9,
                depthWrite: false,
                side: THREE.DoubleSide,
              }),
            );
            evidenceOverlay.rotation.x = -Math.PI / 2;
            evidenceOverlay.position.y = floor.elevationMeters + 0.045;
            evidenceOverlay.renderOrder = 2;
            group.add(evidenceOverlay);

            const wallMaterial = new THREE.MeshStandardMaterial({
              color: 0xd8dee5,
              roughness: 0.82,
              transparent: true,
              opacity: 0.74,
            });
            for (const wall of floor.walls) {
              const dx = wall.end[0] - wall.start[0];
              const dz = wall.end[1] - wall.start[1];
              const length = Math.hypot(dx, dz);
              const mesh = new THREE.Mesh(
                new THREE.BoxGeometry(
                  length,
                  wall.heightMeters,
                  wall.thicknessMeters,
                ),
                wallMaterial,
              );
              mesh.position.set(
                (wall.start[0] + wall.end[0]) / 2,
                floor.elevationMeters + wall.heightMeters / 2,
                (wall.start[1] + wall.end[1]) / 2,
              );
              mesh.rotation.y = -Math.atan2(dz, dx);
              candidateGroup.add(mesh);
            }
            group.add(candidateGroup);
            scene.add(group);
            return group;
          }),
        );

        const widest = Math.max(
          ...model.floors.map((floor) => floor.widthMeters),
        );
        const deepest = Math.max(
          ...model.floors.map((floor) => floor.depthMeters),
        );
        const structureHeight = Math.max(
          ...model.floors.map(
            (floor) => floor.elevationMeters + floor.wallHeightMeters,
          ),
        );
        const reset = () => {
          const radius = Math.max(widest, deepest, structureHeight) * 1.45;
          camera.position.set(radius, structureHeight + radius * 0.45, radius);
          controls.target.set(0, Math.max(1.3, structureHeight * 0.4), 0);
          controls.update();
        };
        const focusFloor = (floorIndex: number) => {
          const floor = model.floors.find(
            (candidate) => candidate.index === floorIndex,
          );
          if (!floor) return;
          const radius = Math.max(floor.widthMeters, floor.depthMeters) * 0.82;
          camera.position.set(
            radius * 0.7,
            floor.elevationMeters + radius * 0.95,
            radius * 0.72,
          );
          controls.target.set(0, floor.elevationMeters, 0);
          controls.update();
        };
        const focusPlan = (floorIndex: number | "all") => {
          const floor =
            floorIndex === "all"
              ? model.floors[0]
              : model.floors.find(
                  (candidate) => candidate.index === floorIndex,
                );
          if (!floor) return;
          const span = Math.max(floor.widthMeters, floor.depthMeters);
          camera.up.set(0, 0, -1);
          camera.position.set(
            0,
            floor.elevationMeters + span * 1.28,
            0.001,
          );
          controls.target.set(0, floor.elevationMeters, 0);
          controls.enableRotate = false;
          controls.enablePan = true;
          controls.update();
        };
        const setSceneViewMode = (
          mode: ViewMode,
          floorIndex: number | "all",
        ) => {
          for (const candidateGroup of candidateGroups)
            candidateGroup.visible = mode === "candidate";
          if (mode === "plan") focusPlan(floorIndex);
          else {
            camera.up.set(0, 1, 0);
            controls.enableRotate = true;
            if (floorIndex === "all") reset();
            else focusFloor(floorIndex);
          }
        };
        setSceneViewMode(viewMode, selectedFloor);

        const raycaster = new THREE.Raycaster();
        const pointer = new THREE.Vector2();
        const roomKeyAtPointer = (event: PointerEvent) => {
          const rect = canvas.getBoundingClientRect();
          pointer.set(
            ((event.clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1,
            -((event.clientY - rect.top) / Math.max(1, rect.height)) * 2 + 1,
          );
          raycaster.setFromCamera(pointer, camera);
          const hit = raycaster.intersectObjects(roomAnchors, true)[0]?.object;
          let target: import("three").Object3D | null = hit ?? null;
          while (target && !target.userData.roomKey) target = target.parent;
          return target?.userData.roomKey as string | undefined;
        };
        const onPointerMove = (event: PointerEvent) => {
          const key = roomKeyAtPointer(event);
          setHoveredRoomKey(key);
          canvas.style.cursor = key ? "pointer" : "";
        };
        const onPointerLeave = () => {
          setHoveredRoomKey(undefined);
          canvas.style.cursor = "";
        };
        const onCanvasClick = (event: PointerEvent) => {
          const key = roomKeyAtPointer(event);
          if (!key) return;
          const entry = allRoomEntries.find(
            (candidate) => candidate.key === key,
          );
          if (!entry) return;
          setPinnedRoomKey(key);
          if (entry.resolution.available) {
            onRoomSelect?.(entry.selection, entry.resolution.candidates[0]);
          }
        };
        canvas.addEventListener("pointermove", onPointerMove);
        canvas.addEventListener("pointerleave", onPointerLeave);
        canvas.addEventListener("click", onCanvasClick);

        const pressed = new Set<string>();
        const onKeyDown = (event: KeyboardEvent) => pressed.add(event.code);
        const onKeyUp = (event: KeyboardEvent) => pressed.delete(event.code);
        window.addEventListener("keydown", onKeyDown);
        window.addEventListener("keyup", onKeyUp);
        let lastTime = performance.now();
        const resize = () => {
          const width = canvas.clientWidth;
          const height = canvas.clientHeight;
          const pixelRatio = renderer.getPixelRatio();
          if (
            canvas.width !== Math.round(width * pixelRatio) ||
            canvas.height !== Math.round(height * pixelRatio)
          ) {
            renderer.setSize(width, height, false);
            camera.aspect = width / Math.max(1, height);
            camera.updateProjectionMatrix();
          }
        };
        const render = (time: number) => {
          if (cancelled) return;
          resize();
          const delta = Math.min(0.05, (time - lastTime) / 1000);
          lastTime = time;
          const forward = new THREE.Vector3();
          camera.getWorldDirection(forward);
          forward.y = 0;
          if (forward.lengthSq()) forward.normalize();
          const right = new THREE.Vector3(-forward.z, 0, forward.x);
          const movement = new THREE.Vector3();
          if (pressed.has("KeyW")) movement.add(forward);
          if (pressed.has("KeyS")) movement.sub(forward);
          if (pressed.has("KeyD")) movement.add(right);
          if (pressed.has("KeyA")) movement.sub(right);
          if (pressed.has("KeyE")) movement.y += 1;
          if (pressed.has("KeyQ")) movement.y -= 1;
          if (movement.lengthSq()) {
            movement
              .normalize()
              .multiplyScalar(delta * (pressed.has("ShiftLeft") ? 8 : 3.2));
            camera.position.add(movement);
            controls.target.add(movement);
          }
          controls.update();
          renderer.render(scene, camera);
          animationFrame = requestAnimationFrame(render);
        };
        const dispose = () => {
          canvas.removeEventListener("pointermove", onPointerMove);
          canvas.removeEventListener("pointerleave", onPointerLeave);
          canvas.removeEventListener("click", onCanvasClick);
          window.removeEventListener("keydown", onKeyDown);
          window.removeEventListener("keyup", onKeyUp);
          controls.dispose();
          scene.traverse((object) => {
            if (
              object instanceof THREE.Mesh ||
              object instanceof THREE.Sprite
            ) {
              object.geometry?.dispose();
              const materials = Array.isArray(object.material)
                ? object.material
                : [object.material];
              for (const material of materials) {
                if ("map" in material) material.map?.dispose();
                material.dispose();
              }
            }
          });
          renderer.dispose();
        };
        engineRef.current = {
          renderer,
          floorGroups,
          roomAnchors,
          reset,
          focusFloor,
          setViewMode: setSceneViewMode,
          dispose,
        };
        animationFrame = requestAnimationFrame(render);
      } catch (caught) {
        if (!cancelled)
          setError(
            caught instanceof Error
              ? caught.message
              : "The structural viewer could not start.",
          );
      }
    })();
    return () => {
      cancelled = true;
      cancelAnimationFrame(animationFrame);
      engineRef.current?.dispose();
      engineRef.current = null;
    };
  }, [allRoomEntries, model, onRoomSelect]);

  useEffect(() => {
    const engine = engineRef.current;
    if (!engine) return;
    for (const group of engine.floorGroups) {
      group.visible =
        selectedFloor === "all" || group.userData.floorIndex === selectedFloor;
    }
    engine.setViewMode(viewMode, selectedFloor);
    setHoveredRoomKey(undefined);
    setPinnedRoomKey(undefined);
  }, [selectedFloor, viewMode, model]);

  useEffect(() => {
    setPlanViewport(undefined);
    planPanRef.current = undefined;
  }, [selectedFloor]);

  const listedRooms = useMemo(
    () =>
      allRoomEntries.filter(
        (entry) =>
          selectedFloor === "all" || selectedFloor === entry.floor.index,
      ),
    [allRoomEntries, selectedFloor],
  );
  const activeRoom =
    allRoomEntries.find((entry) => entry.key === hoveredRoomKey) ??
    allRoomEntries.find((entry) => entry.key === pinnedRoomKey);
  const visibleFloors =
    selectedFloor === "all"
      ? (model?.floors ?? [])
      : (model?.floors ?? []).filter(
          (floor) => floor.index === selectedFloor,
        );
  const visibleLabeledSpaces = visibleFloors.reduce(
    (total, floor) =>
      total + (floor.metrics.labeledSpaces ?? floor.planLabels?.length ?? 0),
    0,
  );
  const visibleCandidateConnections = visibleFloors.reduce(
    (total, floor) =>
      total +
      (floor.metrics.candidateConnections ?? floor.spaceGraph?.edges.length ?? 0),
    0,
  );
  const selectedPlanFloor =
    selectedFloor === "all"
      ? undefined
      : model?.floors.find((floor) => floor.index === selectedFloor);
  const selectedPlanRooms = selectedPlanFloor
    ? allRoomEntries.filter(
        (entry) => entry.floor.index === selectedPlanFloor.index,
      )
    : [];
  const candidateStructureReady = Boolean(
    model?.floors.some((floor) => floor.vectorization?.safeForExtrusion),
  );
  const activePlanViewport = selectedPlanFloor
    ? planViewport?.floorIndex === selectedPlanFloor.index
      ? planViewport
      : defaultPlanViewport(selectedPlanFloor)
    : undefined;

  function zoomPlan(event: ReactWheelEvent<SVGSVGElement>) {
    if (!selectedPlanFloor || !activePlanViewport) return;
    event.preventDefault();
    const rect = event.currentTarget.getBoundingClientRect();
    const pointerX =
      activePlanViewport.x +
      ((event.clientX - rect.left) / Math.max(1, rect.width)) *
        activePlanViewport.width;
    const pointerY =
      activePlanViewport.y +
      ((event.clientY - rect.top) / Math.max(1, rect.height)) *
        activePlanViewport.height;
    const factor = Math.exp(Math.max(-320, Math.min(320, event.deltaY)) * 0.0012);
    const minimumWidth = selectedPlanFloor.imageSize.width * 0.18;
    const nextWidth = Math.max(
      minimumWidth,
      Math.min(selectedPlanFloor.imageSize.width, activePlanViewport.width * factor),
    );
    const nextHeight =
      nextWidth * (activePlanViewport.height / activePlanViewport.width);
    const relativeX =
      (pointerX - activePlanViewport.x) / activePlanViewport.width;
    const relativeY =
      (pointerY - activePlanViewport.y) / activePlanViewport.height;
    setPlanViewport(
      clampPlanViewport(selectedPlanFloor, {
        floorIndex: selectedPlanFloor.index,
        x: pointerX - relativeX * nextWidth,
        y: pointerY - relativeY * nextHeight,
        width: nextWidth,
        height: nextHeight,
      }),
    );
  }

  function startPlanPan(event: ReactPointerEvent<SVGSVGElement>) {
    if (!activePlanViewport) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    planPanRef.current = {
      pointerId: event.pointerId,
      clientX: event.clientX,
      clientY: event.clientY,
      viewport: activePlanViewport,
    };
  }

  function movePlanPan(event: ReactPointerEvent<SVGSVGElement>) {
    if (!selectedPlanFloor || planPanRef.current?.pointerId !== event.pointerId)
      return;
    const rect = event.currentTarget.getBoundingClientRect();
    const pan = planPanRef.current;
    setPlanViewport(
      clampPlanViewport(selectedPlanFloor, {
        ...pan.viewport,
        x:
          pan.viewport.x -
          ((event.clientX - pan.clientX) / Math.max(1, rect.width)) *
            pan.viewport.width,
        y:
          pan.viewport.y -
          ((event.clientY - pan.clientY) / Math.max(1, rect.height)) *
            pan.viewport.height,
      }),
    );
  }

  function endPlanPan(event: ReactPointerEvent<SVGSVGElement>) {
    if (planPanRef.current?.pointerId !== event.pointerId) return;
    planPanRef.current = undefined;
    event.currentTarget.releasePointerCapture(event.pointerId);
  }

  if (error)
    return (
      <div className="viewer-error">
        <TriangleAlert size={28} />
        <strong>Structural model unavailable</strong>
        <span>{error}</span>
      </div>
    );
  if (!model)
    return (
      <div className="viewer-loading">
        <LoaderCircle className="spin" size={24} /> Loading structural model
      </div>
    );
  if (model.floors.length === 0)
    return (
      <div className="viewer-error">
        <TriangleAlert size={28} />
        <strong>No interior floor drawing detected</strong>
        <span>
          {model.referencePlans?.length
            ? "The supplied sheet is a site reference, not an interior floor."
            : "No labeled floor panel could be extracted from the supplied plans."}
        </span>
      </div>
    );

  return (
    <div className="structural-viewer" ref={wrapperRef}>
      <canvas
        ref={canvasRef}
        className={viewMode === "plan" ? "candidate-canvas-hidden" : undefined}
        aria-label="Navigable floorplan structure"
      />
      {viewMode === "plan" && selectedPlanFloor ? (
        <svg
          className="source-plan-document"
          viewBox={`${activePlanViewport?.x ?? 0} ${activePlanViewport?.y ?? 0} ${activePlanViewport?.width ?? selectedPlanFloor.imageSize.width} ${activePlanViewport?.height ?? selectedPlanFloor.imageSize.height}`}
          preserveAspectRatio="xMidYMid meet"
          role="img"
          aria-label={`${selectedPlanFloor.label} exact supplied plan`}
          onWheel={zoomPlan}
          onPointerDown={startPlanPan}
          onPointerMove={movePlanPan}
          onPointerUp={endPlanPan}
          onPointerCancel={endPlanPan}
          onDoubleClick={() => setPlanViewport(undefined)}
        >
          <image
            href={
              selectedPlanFloor.planImageUrl ??
              selectedPlanFloor.planOverlayUrl ??
              selectedPlanFloor.sourceImageUrl
            }
            width={selectedPlanFloor.imageSize.width}
            height={selectedPlanFloor.imageSize.height}
            preserveAspectRatio="none"
          />
          {selectedPlanRooms.map((entry) => {
            const [x, y] = planImagePosition(
              selectedPlanFloor,
              entry.room.position,
            );
            const color = roomStatusCssColor(entry.resolution);
            const interactive =
              entry.resolution.available || entry.resolution.candidates.length;
            const markerScale = Math.max(
              12,
              Math.min(
                selectedPlanFloor.imageSize.width,
                selectedPlanFloor.imageSize.height,
              ) * 0.017,
            );
            const labelWidth = Math.max(
              markerScale * 3.1,
              Math.min(markerScale * 8.4, entry.room.label.length * markerScale * 0.52),
            );
            const labelHeight = markerScale * 1.45;
            return (
              <g
                className={interactive ? "source-plan-pin interactive" : "source-plan-pin"}
                key={entry.key}
                role={interactive ? "button" : undefined}
                tabIndex={interactive ? 0 : undefined}
                aria-label={`${entry.room.label}: ${entry.resolution.detail}`}
                onMouseEnter={() => setHoveredRoomKey(entry.key)}
                onMouseLeave={() => setHoveredRoomKey(undefined)}
                onFocus={() => setHoveredRoomKey(entry.key)}
                onBlur={() => setHoveredRoomKey(undefined)}
                onPointerDown={(event) => event.stopPropagation()}
                onClick={() => {
                  if (entry.resolution.available)
                    onRoomSelect?.(
                      entry.selection,
                      entry.resolution.candidates[0],
                    );
                  else if (entry.resolution.candidates.length)
                    setPinnedRoomKey(entry.key);
                }}
              >
                <rect
                  x={x - labelWidth / 2}
                  y={y - labelHeight / 2}
                  width={labelWidth}
                  height={labelHeight}
                  rx={labelHeight * 0.2}
                  fill={color}
                  fillOpacity={entry.resolution.available ? "0.92" : "0.82"}
                  stroke={color}
                  strokeWidth={entry.resolution.available ? "4" : "2"}
                  vectorEffect="non-scaling-stroke"
                />
                <text
                  x={x}
                  y={y}
                  fill="#ffffff"
                  fontSize={markerScale * 0.52}
                  fontWeight="700"
                  textAnchor="middle"
                  dominantBaseline="central"
                >
                  {entry.room.label}
                </text>
              </g>
            );
          })}
        </svg>
      ) : null}
      <div className="structural-toolbar">
        <span>
          <Layers3 size={14} /> Plan evidence
        </span>
        <button
          type="button"
          aria-pressed={viewMode === "plan"}
          onClick={() => {
            if (selectedFloor === "all")
              setSelectedFloor(model.floors[0]?.index ?? "all");
            setViewMode("plan");
          }}
        >
          Source plan
        </button>
        <button
          type="button"
          aria-pressed={viewMode === "candidate"}
          disabled={!candidateStructureReady}
          title={
            candidateStructureReady
              ? "Inspect the candidate wall extraction"
              : "Candidate walls failed the room-topology safety gate"
          }
          onClick={() => setViewMode("candidate")}
        >
          {candidateStructureReady ? "Candidate 3D" : "3D unavailable"}
        </button>
        <button
          type="button"
          aria-pressed={showRoomIndex}
          onClick={() => setShowRoomIndex((visible) => !visible)}
        >
          Rooms {listedRooms.length}
        </button>
        <button
          type="button"
          onClick={() => engineRef.current?.setViewMode(viewMode, selectedFloor)}
        >
          <RotateCcw size={14} /> Reset
        </button>
        <button
          type="button"
          onClick={() => void wrapperRef.current?.requestFullscreen()}
        >
          <Maximize2 size={14} /> Full screen
        </button>
      </div>
      <div className="floor-switcher" aria-label="Visible floor">
        {viewMode === "candidate" ? (
          <button
            type="button"
            aria-pressed={selectedFloor === "all"}
            onClick={() => setSelectedFloor("all")}
          >
            <Home size={14} /> All floors
          </button>
        ) : null}
        {model.floors.map((floor) => (
          <button
            type="button"
            key={floor.index}
            aria-pressed={selectedFloor === floor.index}
            onClick={() => setSelectedFloor(floor.index)}
          >
            {floor.label}
            <small>
              {viewMode === "plan"
                ? `${floor.metrics.labeledSpaces ?? floor.planLabels?.length ?? 0} labeled spaces`
                : `${floor.metrics.wallSegments} candidate walls`}
            </small>
          </button>
        ))}
      </div>
      {listedRooms.length && showRoomIndex ? (
        <div className="plan-room-index" aria-label="Rooms read from plan">
          <span className="plan-room-index-label">Room portals</span>
          <div className="plan-room-legend" aria-label="Room portal legend">
            <span data-status="linked">Linked 3D scene</span>
            <span data-status="candidate">Matching scene · position unverified</span>
            <span data-status="unavailable">No 3D</span>
          </div>
          {listedRooms.map((entry) => {
            const { room, selection, resolution } = entry;
            return resolution.available ? (
              <button
                type="button"
                className="plan-room-available"
                key={entry.key}
                title={resolution.detail}
                onMouseEnter={() => setHoveredRoomKey(entry.key)}
                onMouseLeave={() => setHoveredRoomKey(undefined)}
                onFocus={() => setHoveredRoomKey(entry.key)}
                onClick={() =>
                  onRoomSelect?.(selection, resolution.candidates[0])
                }
              >
                <DoorOpen size={13} />
                <span>{room.label}</span>
                <small>Linked 3D</small>
              </button>
            ) : resolution.candidates.length ? (
              <button
                type="button"
                className="plan-room-candidate"
                key={entry.key}
                title={resolution.detail}
                onMouseEnter={() => setHoveredRoomKey(entry.key)}
                onMouseLeave={() => setHoveredRoomKey(undefined)}
                onFocus={() => setHoveredRoomKey(entry.key)}
                onClick={() => setPinnedRoomKey(entry.key)}
              >
                <Camera size={13} />
                <span>{room.label}</span>
                <small>
                  {resolution.candidates.length} 3D candidate
                  {resolution.candidates.length === 1 ? "" : "s"}
                </small>
              </button>
            ) : (
              <span
                className="plan-room-unavailable"
                key={entry.key}
                title={resolution.detail}
                onMouseEnter={() => setHoveredRoomKey(entry.key)}
                onMouseLeave={() => setHoveredRoomKey(undefined)}
              >
                <DoorOpen size={13} />
                <span>{room.label}</span>
                <small>No 3D view</small>
              </span>
            );
          })}
        </div>
      ) : null}
      {activeRoom ? (
        <section className="plan-room-preview" aria-live="polite">
          <header>
            <div>
              <span>{activeRoom.floor.label}</span>
              <strong>{activeRoom.room.label}</strong>
            </div>
            <small>
              {activeRoom.resolution.available
                ? "Plan-linked"
                : activeRoom.resolution.candidates.length
                  ? "Position unverified"
                  : "Not reconstructed"}
            </small>
          </header>
          <p>{activeRoom.resolution.detail}</p>
          {activeRoom.resolution.candidates.length ? (
            <div className="plan-room-candidates">
              {activeRoom.resolution.candidates.map((candidate, index) => (
                <button
                  type="button"
                  key={candidate.artifact.id}
                  onClick={() =>
                    onRoomSelect?.(activeRoom.selection, candidate)
                  }
                >
                  {candidate.previewUrl ? (
                    <img
                      src={candidate.previewUrl}
                      alt={candidate.previewTitle ?? activeRoom.room.label}
                    />
                  ) : (
                    <Camera size={22} />
                  )}
                  <span>
                    <strong>Open 3D scene {index + 1}</strong>
                    <small>
                      {candidate.registeredViewCount} calibrated view
                      {candidate.registeredViewCount === 1 ? "" : "s"}
                    </small>
                  </span>
                </button>
              ))}
            </div>
          ) : null}
        </section>
      ) : null}
      <div className="structural-readout">
        <strong>
          {viewMode === "plan"
            ? `Exact supplied drawing · ${visibleLabeledSpaces} labeled spaces`
            : `${model.metrics.floorCount} floors · ${model.metrics.wallSegments} candidate walls`}
        </strong>
        <span>
          {visibleCandidateConnections} candidate room connections · not verified
        </span>
        {model.referencePlans?.length ? (
          <span>{model.referencePlans.length} site plan · reference only</span>
        ) : null}
        <span>
          {viewMode === "plan"
            ? "Drag to pan · wheel zoom · choose a floor"
            : "WASD move · Q/E down/up · drag orbit · wheel zoom"}
        </span>
        <small>GPU: {gpu || "detecting"}</small>
      </div>
      <div className="structural-caveat">
        {viewMode === "plan"
          ? "This is the supplied plan image. No guessed wall, room, stair, or roof geometry replaces it."
          : "Candidate 3D is an unverified extraction preview. It must not be used as a measured or collision-safe building model."}
      </div>
    </div>
  );
}

function polygonGeometry(
  THREE: typeof import("three"),
  polygon: Array<[number, number]>,
) {
  const shape = new THREE.Shape();
  polygon.forEach(([x, z], index) => {
    if (index === 0) shape.moveTo(x, -z);
    else shape.lineTo(x, -z);
  });
  shape.closePath();
  const geometry = new THREE.ShapeGeometry(shape);
  geometry.rotateX(-Math.PI / 2);
  return geometry;
}

function roomStatusColor(resolution: PlanRoomResolution): number {
  if (resolution.available) return 0x56c987;
  if (resolution.candidates.length) return 0xe0a548;
  return 0x56616a;
}

function roomStatusCssColor(resolution: PlanRoomResolution): string {
  if (resolution.available) return "#45bd78";
  if (resolution.candidates.length) return "#d99a3d";
  return "#6f7a82";
}

function clampPlanViewport(floor: Floor, viewport: PlanViewport): PlanViewport {
  const width = Math.min(floor.imageSize.width, Math.max(1, viewport.width));
  const height = Math.min(floor.imageSize.height, Math.max(1, viewport.height));
  return {
    ...viewport,
    width,
    height,
    x: Math.max(0, Math.min(floor.imageSize.width - width, viewport.x)),
    y: Math.max(0, Math.min(floor.imageSize.height - height, viewport.y)),
  };
}

function defaultPlanViewport(floor: Floor): PlanViewport {
  const crop = floor.textureCrop;
  const cropWidth = Math.max(0.05, crop.right - crop.left);
  const cropHeight = Math.max(0.05, crop.bottom - crop.top);
  const marginX = Math.min(0.035, cropWidth * 0.08);
  const marginY = Math.min(0.035, cropHeight * 0.08);
  return clampPlanViewport(floor, {
    floorIndex: floor.index,
    x: Math.max(0, crop.left - marginX) * floor.imageSize.width,
    y: Math.max(0, crop.top - marginY) * floor.imageSize.height,
    width:
      Math.min(1, crop.right + marginX) * floor.imageSize.width -
      Math.max(0, crop.left - marginX) * floor.imageSize.width,
    height:
      Math.min(1, crop.bottom + marginY) * floor.imageSize.height -
      Math.max(0, crop.top - marginY) * floor.imageSize.height,
  });
}

function planImagePosition(
  floor: Floor,
  position: [number, number],
): [number, number] {
  const cropWidth = floor.textureCrop.right - floor.textureCrop.left;
  const cropHeight = floor.textureCrop.bottom - floor.textureCrop.top;
  const xWithinCrop = position[0] / Math.max(floor.widthMeters, 1e-9) + 0.5;
  const yWithinCrop = position[1] / Math.max(floor.depthMeters, 1e-9) + 0.5;
  return [
    (floor.textureCrop.left + xWithinCrop * cropWidth) * floor.imageSize.width,
    (floor.textureCrop.top + yWithinCrop * cropHeight) * floor.imageSize.height,
  ];
}

function textSprite(THREE: typeof import("three"), text: string) {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 96;
  const context = canvas.getContext("2d")!;
  context.fillStyle = "rgba(16, 21, 27, .86)";
  context.roundRect(4, 4, 504, 88, 16);
  context.fill();
  context.fillStyle = "#f4f6f8";
  context.font = "600 30px system-ui";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(text.slice(0, 30), 256, 48);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthTest: false,
    }),
  );
  sprite.scale.set(2.8, 0.52, 1);
  sprite.renderOrder = 5;
  return sprite;
}
