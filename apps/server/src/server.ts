import { createHash, timingSafeEqual } from "node:crypto";
import {
  createWriteStream,
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
} from "node:fs";
import { pipeline as streamPipeline } from "node:stream/promises";
import { basename, extname, isAbsolute, relative, resolve } from "node:path";
import cookie from "@fastify/cookie";
import multipart from "@fastify/multipart";
import fastifyStatic from "@fastify/static";
import Fastify, { type FastifyInstance, type FastifyRequest } from "fastify";
import sharp from "sharp";
import { z } from "zod";
import {
  AddEvidenceLinkInputSchema,
  AiCaseAnalysisInputSchema,
  AiProviderIdSchema,
  CreateCaseInputSchema,
  CaptureUploadModeSchema,
  DiscoveryRunInputSchema,
  LoadAiProviderModelsInputSchema,
  MultiReconstructionRequestSchema,
  ReconstructionRequestSchema,
  ReviewInputSchema,
  SaveAiProviderInputSchema,
  SaveDiscoverySettingsInputSchema,
  TestAiProviderInputSchema,
  UpdateIncidentInputSchema,
  type EvidenceAsset,
  type CaptureInfo,
  type CaptureUploadMode,
  type IncidentContext,
  type PhotoUploadResult,
  type Role,
  type SystemHealth,
} from "@structurefirst/contracts";
import { type AppConfig, loadConfig } from "./config.js";
import { AiCaseAnalyzer } from "./ai.js";
import { EvidenceDiscoveryCoordinator } from "./discovery.js";
import { CaseEventHub } from "./events.js";
import { confidence } from "./lib/confidence.js";
import { createId, nowIso } from "./lib/ids.js";
import { classifySource } from "./lib/source-policy.js";
import { CasePipeline } from "./pipeline.js";
import { ReconstructionCoordinator } from "./reconstruction.js";
import { SceneIntelligenceService } from "./scene-intelligence.js";
import { SettingsService } from "./settings.js";
import { StructureStore } from "./store.js";

const LoginSchema = z.object({ accessKey: z.string().min(1).max(1000) });
const RoleUpdateSchema = z.object({
  role: z.enum(["fire", "law", "ems", "sar"]),
});
const ArchiveSchema = z.object({ archived: z.boolean() });
const IdParamsSchema = z.object({ id: z.string().min(8).max(128) });
const CaseChildParamsSchema = z.object({
  id: z.string().min(8).max(128),
  childId: z.string().min(8).max(128),
});
const ProviderParamsSchema = z.object({
  providerId: AiProviderIdSchema,
});

const SESSION_COOKIE = "sf_session";
const MAX_UPLOAD_BYTES = 1024 * 1024 * 1024;
const MAX_UPLOAD_FILES = 50;
const ClassifyImagesInputSchema = z.object({
  evidenceIds: z
    .array(z.string().min(8).max(128))
    .max(MAX_UPLOAD_FILES)
    .optional(),
});
const DetectionFrameInputSchema = z.object({
  imageDataUrl: z.string().min(32).max(2_000_000),
  scoreThreshold: z.number().min(0.1).max(0.9).optional(),
});
const MIME_EXTENSIONS: Record<string, string> = {
  "image/jpeg": ".jpg",
  "image/png": ".png",
  "image/webp": ".webp",
  "application/pdf": ".pdf",
};

export type AppServices = {
  config: AppConfig;
  store: StructureStore;
  eventHub: CaseEventHub;
  casePipeline: CasePipeline;
  reconstruction: ReconstructionCoordinator;
  settings: SettingsService;
  discovery: EvidenceDiscoveryCoordinator;
  aiAnalyzer: AiCaseAnalyzer;
  sceneIntelligence: SceneIntelligenceService;
};

export async function buildServer(
  overrides: Partial<AppConfig> = {},
): Promise<{ app: FastifyInstance; services: AppServices }> {
  const config = loadConfig(overrides);
  const store = new StructureStore(config.databasePath);
  const eventHub = new CaseEventHub();
  const settings = new SettingsService(store, config);
  const discovery = new EvidenceDiscoveryCoordinator(store, settings, config);
  const sceneIntelligence = new SceneIntelligenceService(
    store,
    settings,
    config,
  );
  const reconstruction = new ReconstructionCoordinator(
    store,
    eventHub,
    config,
    (artifact) => sceneIntelligence.rebuildSpatialGraph(artifact.caseId),
  );
  const casePipeline = new CasePipeline(
    store,
    eventHub,
    config,
    undefined,
    discovery,
    reconstruction,
    sceneIntelligence,
  );
  const aiAnalyzer = new AiCaseAnalyzer(store, settings, config);
  const app = Fastify({
    logger: process.env.NODE_ENV !== "test",
    bodyLimit: 2 * 1024 * 1024,
    requestTimeout: 30_000,
  });
  const backgroundTasks = new Set<Promise<void>>();

  const runBackground = (operation: () => Promise<void>): void => {
    const task = operation()
      .catch((error) => app.log.error(error))
      .finally(() => backgroundTasks.delete(task));
    backgroundTasks.add(task);
  };

  const organizeUploadedCapture = async (
    caseId: string,
    evidenceIds: string[],
    resolvedMode: Exclude<CaptureUploadMode, "auto">,
  ): Promise<void> => {
    reconstruction.reportWorkflow(
      caseId,
      "running",
      `Saved ${evidenceIds.length} files locally. Separating floorplans from photographic views.`,
    );
    // Classify the complete set before either reconstruction queue sees it.
    // A page-statistics heuristic can suggest a plan, but it cannot reliably
    // distinguish a plan, a bright room photo, or an unrelated image.
    await sceneIntelligence.analyzeCase(caseId, evidenceIds);
    const organized = evidenceIds
      .map((evidenceId) => store.getEvidence(evidenceId))
      .filter((item): item is EvidenceAsset => Boolean(item));
    const plans = organized.filter(isFloorPlanEvidence);
    if (plans.length) {
      await reconstruction.queueFloorPlans(
        caseId,
        plans.map((item) => item.id),
      );
    }

    const photoCandidates = organized.filter(
      (item) =>
        !isFloorPlanEvidence(item) &&
        !item.tags.includes("reconstruction-excluded"),
    );
    if (photoCandidates.length) {
      reconstruction.reportWorkflow(
        caseId,
        "running",
        plans.length
          ? `Building the plan reference while camera geometry checks ${photoCandidates.length} photographic views for a connected Gaussian scene.`
          : `Checking ${photoCandidates.length} photographic views for one connected camera path. No floorplan is required.`,
      );
      if (resolvedMode !== "perspective" && photoCandidates.length === 1) {
        await reconstruction.queue(caseId, {
          evidenceId: photoCandidates[0]!.id,
          mode: "panorama",
        });
      } else {
        await reconstruction.queueCaptureSet(
          caseId,
          photoCandidates.map((item) => item.id),
        );
      }
      return;
    }
    if (!photoCandidates.length) {
      if (!plans.length) {
        reconstruction.reportWorkflow(
          caseId,
          "limited",
          "No uploaded file was eligible for structural or Gaussian reconstruction.",
        );
      }
      return;
    }
  };

  await app.register(cookie, {
    secret: config.cookieSecret,
    hook: "onRequest",
  });
  await app.register(multipart, {
    limits: {
      fileSize: MAX_UPLOAD_BYTES,
      files: MAX_UPLOAD_FILES,
      fields: 2,
      parts: MAX_UPLOAD_FILES + 2,
    },
  });

  app.addHook("onRequest", async (request, reply) => {
    const pathname = request.url.split("?", 1)[0] ?? request.url;
    const protectedPath =
      pathname.startsWith("/api/") || pathname.startsWith("/assets/");
    const publicPath = ["/api/health", "/api/auth/session"].includes(pathname);
    if (!protectedPath || publicPath || !config.accessKey) return;
    if (!isAuthorized(request, config)) {
      return reply.code(401).send({ error: "Authentication required." });
    }
    if (
      ["POST", "PUT", "PATCH", "DELETE"].includes(request.method) &&
      request.headers.authorization === undefined &&
      request.headers["x-structurefirst-intent"] !== "operator-action"
    ) {
      return reply
        .code(403)
        .send({ error: "Operator action header required." });
    }
  });

  app.setErrorHandler((error, _request, reply) => {
    if (error instanceof z.ZodError) {
      return reply
        .code(400)
        .send({ error: "Invalid request.", details: error.issues });
    }
    if ((error as { code?: string }).code === "FST_REQ_FILE_TOO_LARGE") {
      return reply
        .code(413)
        .send({ error: "Upload exceeds the 1 GB photo limit." });
    }
    if (
      ["FST_FILES_LIMIT", "FST_PARTS_LIMIT"].includes(
        (error as { code?: string }).code ?? "",
      )
    ) {
      return reply
        .code(413)
        .send({ error: "Select no more than 50 photos at once." });
    }
    app.log.error(error);
    return reply.code(500).send({ error: errorMessage(error) });
  });

  app.get("/api/health", async (): Promise<SystemHealth> => {
    const worker = await reconstruction.health();
    const workerReady = worker.reachable && worker.details?.status === "ready";
    return {
      service: "structurefirst",
      status: workerReady || !config.lucidFrameRoot ? "ready" : "degraded",
      version: "0.1.0",
      database: "ready",
      braveSearch: settings.list().discovery.braveConfigured,
      browserResearch: settings.list().discovery.browserEnabled,
      reconstruction: {
        configured: Boolean(config.lucidFrameRoot),
        reachable: worker.reachable,
        ready: workerReady,
        ...(config.lucidFrameRoot
          ? { lucidFrameRoot: config.lucidFrameRoot }
          : {}),
        gpuExpected: true,
        ...(worker.details?.gpu_available === undefined
          ? {}
          : { gpuAvailable: worker.details.gpu_available }),
        ...(worker.details?.lucidframe_available === undefined
          ? {}
          : { lucidFrameAvailable: worker.details.lucidframe_available }),
        ...(worker.details?.sharp_checkpoint_verified === undefined
          ? {}
          : { modelVerified: worker.details.sharp_checkpoint_verified }),
        ...(worker.details?.runtime_error
          ? { error: worker.details.runtime_error }
          : {}),
      },
      sourcePolicy: "research_hard_blocklist",
    };
  });

  app.get("/api/auth/session", async (request) => ({
    required: Boolean(config.accessKey),
    authenticated: !config.accessKey || isAuthorized(request, config),
  }));

  app.post("/api/auth/session", async (request, reply) => {
    if (!config.accessKey) return { authenticated: true, required: false };
    const body = LoginSchema.parse(request.body);
    if (!safeEqual(body.accessKey, config.accessKey)) {
      return reply.code(401).send({ error: "The access key is not valid." });
    }
    reply.setCookie(SESSION_COOKIE, "authorized", {
      signed: true,
      httpOnly: true,
      sameSite: "strict",
      path: "/",
      maxAge: 12 * 60 * 60,
    });
    return { authenticated: true, required: true };
  });

  app.delete("/api/auth/session", async (_request, reply) => {
    reply.clearCookie(SESSION_COOKIE, { path: "/" });
    return { authenticated: false };
  });

  app.get("/api/settings", async () => settings.list());

  app.post("/api/detection/frame", async (request) => {
    const input = DetectionFrameInputSchema.parse(request.body);
    return reconstruction.detectFrame(input);
  });

  app.put("/api/settings/providers/:providerId", async (request) => {
    const { providerId } = ProviderParamsSchema.parse(request.params);
    const input = SaveAiProviderInputSchema.parse(request.body);
    return settings.saveProvider(providerId, input);
  });

  app.post("/api/settings/providers/:providerId/test", async (request) => {
    const { providerId } = ProviderParamsSchema.parse(request.params);
    const input = TestAiProviderInputSchema.parse(request.body ?? {});
    return settings.test(providerId, input);
  });

  app.post("/api/settings/providers/:providerId/models", async (request) => {
    const { providerId } = ProviderParamsSchema.parse(request.params);
    const input = LoadAiProviderModelsInputSchema.parse(request.body ?? {});
    return settings.loadModels(providerId, input.apiKey);
  });

  app.put("/api/settings/discovery", async (request) => {
    const input = SaveDiscoverySettingsInputSchema.parse(request.body);
    return settings.saveDiscovery(input);
  });

  app.get("/api/cases", async () => store.listCases());

  app.post("/api/cases", async (request, reply) => {
    const input = CreateCaseInputSchema.parse(request.body);
    const created = casePipeline.createCase(input);
    casePipeline.start(created.id);
    return reply.code(202).send(created);
  });

  app.get("/api/cases/:id", async (request, reply) => {
    const { id } = IdParamsSchema.parse(request.params);
    const workspace = store.getWorkspace(id);
    return workspace ?? reply.code(404).send({ error: "Case not found." });
  });

  app.delete("/api/cases/:id", async (request, reply) => {
    const { id } = IdParamsSchema.parse(request.params);
    if (!store.getCase(id))
      return reply.code(404).send({ error: "Property not found." });
    const directory = resolve(config.casesRoot, id);
    const traversal = relative(config.casesRoot, directory);
    if (!traversal || traversal.startsWith("..") || isAbsolute(traversal)) {
      return reply.code(400).send({ error: "Invalid property storage path." });
    }
    const deleted = store.deleteCase(id);
    if (deleted) rmSync(directory, { recursive: true, force: true });
    return { deleted };
  });

  app.post("/api/cases/:id/retry", async (request, reply) => {
    const { id } = IdParamsSchema.parse(request.params);
    const current = store.getCase(id);
    if (!current) return reply.code(404).send({ error: "Case not found." });
    if (casePipeline.isRunning(id))
      return reply
        .code(409)
        .send({ error: "Case pipeline is already running." });
    store.putCase({
      ...current,
      status: "collecting",
      updatedAt: nowIso(),
      stages: current.stages.map((stage) => ({
        name: stage.name,
        status: "pending",
        message: "Waiting",
      })),
    });
    casePipeline.start(id);
    return reply.code(202).send(store.getCase(id));
  });

  app.post("/api/cases/:id/discovery", async (request, reply) => {
    const { id } = IdParamsSchema.parse(request.params);
    if (!store.getCase(id))
      return reply.code(404).send({ error: "Case not found." });
    const input = DiscoveryRunInputSchema.parse(request.body ?? {});
    const result = await discovery.discoverCase(id, input);
    void sceneIntelligence
      .analyzeCase(id)
      .then(() => reconstruction.queueAvailable(id))
      .catch((error: unknown) => {
        request.log.warn(
          { error: errorMessage(error), caseId: id },
          "Post-discovery image analysis failed",
        );
      });
    return result;
  });

  app.post("/api/cases/:id/ai/analyze", async (request, reply) => {
    const { id } = IdParamsSchema.parse(request.params);
    if (!store.getCase(id))
      return reply.code(404).send({ error: "Case not found." });
    const input = AiCaseAnalysisInputSchema.parse(request.body ?? {});
    if (!settings.credential(input.provider)) {
      return reply.code(409).send({
        error:
          "Configure and enable an AI provider with an API key and model in Settings first.",
      });
    }
    return aiAnalyzer.analyze(id, input);
  });

  app.post("/api/cases/:id/ai/classify-images", async (request, reply) => {
    const { id } = IdParamsSchema.parse(request.params);
    if (!store.getCase(id))
      return reply.code(404).send({ error: "Case not found." });
    const input = ClassifyImagesInputSchema.parse(request.body ?? {});
    const result = await sceneIntelligence.analyzeCase(
      id,
      input.evidenceIds,
      true,
    );
    sceneIntelligence.rebuildSpatialGraph(id);
    return result;
  });

  app.patch("/api/cases/:id/incident", async (request, reply) => {
    const { id } = IdParamsSchema.parse(request.params);
    const patch = UpdateIncidentInputSchema.parse(request.body);
    const current = store.getCase(id);
    if (!current) return reply.code(404).send({ error: "Case not found." });
    const incident: IncidentContext = {
      ...current.incident,
      ...(patch.type !== undefined ? { type: patch.type } : {}),
      ...(patch.reportedOrigin !== undefined
        ? { reportedOrigin: patch.reportedOrigin }
        : {}),
      ...(patch.preferredEntry !== undefined
        ? { preferredEntry: patch.preferredEntry }
        : {}),
      ...(patch.victimReports !== undefined
        ? { victimReports: patch.victimReports }
        : {}),
      ...(patch.blockedAreas !== undefined
        ? { blockedAreas: patch.blockedAreas }
        : {}),
      ...(patch.notes !== undefined ? { notes: patch.notes } : {}),
      updatedAt: nowIso(),
    };
    return store.putCase({ ...current, incident, updatedAt: nowIso() });
  });

  app.patch("/api/cases/:id/role", async (request, reply) => {
    const { id } = IdParamsSchema.parse(request.params);
    const { role } = RoleUpdateSchema.parse(request.body) as { role: Role };
    const current = store.getCase(id);
    if (!current) return reply.code(404).send({ error: "Case not found." });
    return store.putCase({ ...current, activeRole: role, updatedAt: nowIso() });
  });

  app.patch("/api/cases/:id/archive", async (request, reply) => {
    const { id } = IdParamsSchema.parse(request.params);
    const { archived } = ArchiveSchema.parse(request.body);
    const current = store.getCase(id);
    if (!current) return reply.code(404).send({ error: "Case not found." });
    return store.putCase({
      ...current,
      status: archived ? "archived" : "limited_evidence",
      updatedAt: nowIso(),
    });
  });

  app.post("/api/cases/:id/evidence/link", async (request, reply) => {
    const { id } = IdParamsSchema.parse(request.params);
    if (!store.getCase(id))
      return reply.code(404).send({ error: "Case not found." });
    const input = AddEvidenceLinkInputSchema.parse(request.body);
    const policy = classifySource(input.url);
    const evidence: EvidenceAsset = {
      id: createId("evidence"),
      caseId: id,
      title: input.title,
      kind: input.kind,
      sourceProvider: policy.provider,
      originUrl: input.url,
      discoveredAt: nowIso(),
      rights: policy.rights,
      cachePolicy: policy.cachePolicy,
      redistributable: policy.redistributable,
      validation: "pending",
      tags: ["operator-linked"],
      notes: `${input.notes}${input.notes ? " " : ""}${policy.reason}`,
      confidence: confidence(
        0.35,
        "estimated",
        "observed",
        "The operator linked this source; its contents and address relevance still require review.",
        1,
      ),
    };
    return reply.code(201).send(store.putEvidence(evidence));
  });

  app.post("/api/cases/:id/evidence/upload", async (request, reply) => {
    const { id } = IdParamsSchema.parse(request.params);
    if (!store.getCase(id))
      return reply.code(404).send({ error: "Case not found." });
    const part = await request.file();
    if (!part) return reply.code(400).send({ error: "No file was supplied." });
    const extension = MIME_EXTENSIONS[part.mimetype];
    if (!extension) {
      part.file.resume();
      return reply
        .code(415)
        .send({ error: "Use JPEG, PNG, WebP, or PDF evidence." });
    }

    const uploadId = createId("upload");
    const directory = resolve(config.casesRoot, id, "uploads");
    mkdirSync(directory, { recursive: true });
    const outputName = `${uploadId}${extension}`;
    const outputPath = resolve(directory, outputName);
    const hash = createHash("sha256");
    let byteSize = 0;
    part.file.on("data", (chunk: Buffer) => {
      hash.update(chunk);
      byteSize += chunk.byteLength;
    });
    try {
      await streamPipeline(
        part.file,
        createWriteStream(outputPath, { flags: "wx" }),
      );
    } catch (error) {
      rmSync(outputPath, { force: true });
      throw error;
    }

    const scope = multipartField(part.fields, "scope") ?? "exterior";
    const safeScope = ["exterior", "interior", "blueprint"].includes(scope)
      ? scope
      : "exterior";
    const suppliedTitle = multipartField(part.fields, "title");
    const originalName = basename(part.filename || "evidence");
    const kind =
      safeScope === "blueprint"
        ? "blueprint"
        : part.mimetype === "application/pdf"
          ? "document"
          : "image";
    const evidence: EvidenceAsset = {
      id: createId("evidence"),
      caseId: id,
      title: suppliedTitle?.trim() || originalName,
      kind,
      sourceProvider: "Operator upload",
      localUrl: `/assets/${id}/uploads/${outputName}`,
      discoveredAt: nowIso(),
      rights: "operator_owned",
      cachePolicy: "local_allowed",
      redistributable: false,
      validation: "operator_uploaded",
      mimeType: part.mimetype,
      byteSize,
      sha256: hash.digest("hex"),
      tags: ["operator-upload", safeScope],
      notes:
        "Locally stored case evidence. Redistribution remains disabled until the operator confirms rights.",
      confidence: confidence(
        0.76,
        "verified",
        "observed",
        "File bytes and case assignment are operator supplied; visual contents have not been independently verified.",
        1,
      ),
    };
    return reply.code(201).send(store.putEvidence(evidence));
  });

  app.post("/api/cases/:id/photos", async (request, reply) => {
    const { id } = IdParamsSchema.parse(request.params);
    if (!store.getCase(id))
      return reply.code(404).send({ error: "Property not found." });

    const directory = resolve(config.casesRoot, id, "uploads");
    mkdirSync(directory, { recursive: true });
    const pending: Array<{
      asset: EvidenceAsset;
      outputPath: string;
      width?: number;
      height?: number;
    }> = [];
    const writtenPaths: string[] = [];
    const overlapSetId = createId("capture_set");
    let totalBytes = 0;
    let captureMode: CaptureUploadMode = "auto";

    try {
      const files = request.files({
        limits: {
          fileSize: MAX_UPLOAD_BYTES,
          files: MAX_UPLOAD_FILES,
          parts: MAX_UPLOAD_FILES + 2,
        },
      });
      for await (const part of files) {
        if (pending.length === 0) {
          const parsedMode = CaptureUploadModeSchema.safeParse(
            multipartField(part.fields, "captureMode") ?? "auto",
          );
          if (parsedMode.success) captureMode = parsedMode.data;
        }
        const extension = MIME_EXTENSIONS[part.mimetype];
        if (!extension || !part.mimetype.startsWith("image/")) {
          part.file.resume();
          throw new Error("PHOTO_TYPE_UNSUPPORTED");
        }

        const uploadId = createId("upload");
        const outputName = `${uploadId}${extension}`;
        const outputPath = resolve(directory, outputName);
        writtenPaths.push(outputPath);
        const hash = createHash("sha256");
        let byteSize = 0;
        part.file.on("data", (chunk: Buffer) => {
          byteSize += chunk.byteLength;
          totalBytes += chunk.byteLength;
          hash.update(chunk);
          if (totalBytes > MAX_UPLOAD_BYTES) {
            part.file.destroy(new Error("PHOTO_BATCH_TOO_LARGE"));
          }
        });
        await streamPipeline(
          part.file,
          createWriteStream(outputPath, { flags: "wx" }),
        );
        if (part.file.truncated) throw new Error("PHOTO_BATCH_TOO_LARGE");
        const metadata = await sharp(outputPath).metadata();
        const swapsAxes =
          metadata.orientation !== undefined &&
          metadata.orientation >= 5 &&
          metadata.orientation <= 8;
        const width = swapsAxes ? metadata.height : metadata.width;
        const height = swapsAxes ? metadata.width : metadata.height;

        pending.push({
          outputPath,
          ...(width !== undefined ? { width } : {}),
          ...(height !== undefined ? { height } : {}),
          asset: {
            id: createId("evidence"),
            caseId: id,
            title: basename(part.filename || "Property photo"),
            kind: "image",
            sourceProvider: "Responder upload",
            localUrl: `/assets/${id}/uploads/${outputName}`,
            discoveredAt: nowIso(),
            rights: "operator_owned",
            cachePolicy: "local_allowed",
            redistributable: false,
            validation: "operator_uploaded",
            mimeType: part.mimetype,
            byteSize,
            sha256: hash.digest("hex"),
            tags: [
              "operator-upload",
              "property-photo",
              `overlap-set:${overlapSetId}`,
            ],
            notes:
              "Locally stored responder photo. It is used only for this property unless the operator grants broader rights.",
            confidence: confidence(
              0.76,
              "verified",
              "observed",
              "The file bytes and property assignment were supplied by the operator.",
              1,
            ),
          },
        });
      }
    } catch (error) {
      for (const outputPath of writtenPaths)
        rmSync(outputPath, { force: true });
      if (errorMessage(error).includes("PHOTO_TYPE_UNSUPPORTED")) {
        return reply
          .code(415)
          .send({ error: "Use JPEG, PNG, or WebP photos." });
      }
      if (errorMessage(error).includes("PHOTO_BATCH_TOO_LARGE")) {
        return reply
          .code(413)
          .send({ error: "The selected photos exceed 1 GB in total." });
      }
      throw error;
    }

    if (pending.length === 0)
      return reply.code(400).send({ error: "Select at least one photo." });

    if (
      captureMode !== "auto" &&
      captureMode !== "perspective" &&
      pending.length !== 1
    ) {
      for (const outputPath of writtenPaths)
        rmSync(outputPath, { force: true });
      return reply.code(400).send({
        error: "Upload one stitched image for a 180° or 360° panorama.",
      });
    }
    const resolvedMode = resolveCaptureMode(captureMode, pending);
    const panoramaShapeError = validatePanoramaShape(resolvedMode, pending);
    if (panoramaShapeError) {
      for (const outputPath of writtenPaths)
        rmSync(outputPath, { force: true });
      return reply.code(400).send({ error: panoramaShapeError });
    }
    for (const [captureOrder, item] of pending.entries()) {
      item.asset.capture = captureInfo(
        resolvedMode,
        captureOrder,
        overlapSetId,
        item.width,
        item.height,
        captureMode === "auto"
          ? resolvedMode === "panorama_360"
            ? "aspect_ratio"
            : "default"
          : "operator",
      );
      item.asset.tags.push(`projection:${item.asset.capture.projection}`);
      item.asset.tags.push(`capture-order:${captureOrder}`);
    }

    await Promise.all(
      pending.map(async (item) => {
        if (
          resolvedMode === "perspective" &&
          (await isLikelyFloorPlan(item.outputPath))
        ) {
          item.asset.kind = "blueprint";
          item.asset.tags = item.asset.tags.filter(
            (tag) => tag !== "property-photo",
          );
          item.asset.tags.push("plan-candidate", "architectural-plan");
          item.asset.notes =
            "Locally stored operator upload. Image statistics indicate a likely floorplan; VLM classification and vector extraction run separately from photo splatting.";
        }
      }),
    );

    const assets = pending.map((item) => store.putEvidence(item.asset));
    const planCandidates = assets.filter((item) =>
      item.tags.includes("plan-candidate"),
    ).length;
    runBackground(() =>
      organizeUploadedCapture(
        id,
        assets.map((item) => item.id),
        resolvedMode,
      ),
    );
    const result: PhotoUploadResult = {
      assets,
      processing: {
        status: "organizing",
        totalFiles: assets.length,
        planCandidates,
      },
      note: `Saved ${assets.length} original ${assets.length === 1 ? "file" : "files"}. ${planCandidates ? `${planCandidates} likely floorplans will build the structural floors; ` : ""}photographs are being checked for real overlap in the background.`,
    };
    return reply.code(201).send(result);
  });

  app.post(
    "/api/cases/:id/evidence/:childId/import",
    async (request, reply) => {
      const { id, childId } = CaseChildParamsSchema.parse(request.params);
      if (!store.getCase(id))
        return reply.code(404).send({ error: "Case not found." });
      return discovery.importOpenAsset(id, childId);
    },
  );

  app.post("/api/cases/:id/reconstruction", async (request, reply) => {
    const { id } = IdParamsSchema.parse(request.params);
    if (!store.getCase(id))
      return reply.code(404).send({ error: "Case not found." });
    const input = ReconstructionRequestSchema.parse(request.body);
    const artifact = await reconstruction.queue(id, input);
    return reply.code(202).send(artifact);
  });

  app.post("/api/cases/:id/reconstruction/connect", async (request, reply) => {
    const { id } = IdParamsSchema.parse(request.params);
    if (!store.getCase(id))
      return reply.code(404).send({ error: "Case not found." });
    const input = MultiReconstructionRequestSchema.parse(request.body);
    const artifact = await reconstruction.queueMulti(id, input);
    return reply.code(202).send(artifact);
  });

  app.patch(
    "/api/cases/:id/hazards/:childId/review",
    async (request, reply) => {
      const { id, childId } = CaseChildParamsSchema.parse(request.params);
      const input = ReviewInputSchema.parse(request.body);
      const hazard = store.getHazard(childId);
      if (!hazard || hazard.caseId !== id)
        return reply.code(404).send({ error: "Hazard not found." });
      return store.putHazard({
        ...hazard,
        review: input.review,
        reviewNote: input.note,
        reviewedAt: nowIso(),
      });
    },
  );

  app.patch("/api/cases/:id/routes/:childId/review", async (request, reply) => {
    const { id, childId } = CaseChildParamsSchema.parse(request.params);
    const input = ReviewInputSchema.parse(request.body);
    const route = store.getRoute(childId);
    if (!route || route.caseId !== id)
      return reply.code(404).send({ error: "Route not found." });
    return store.putRoute({
      ...route,
      review: input.review,
      reviewNote: input.note,
      reviewedAt: nowIso(),
    });
  });

  app.get("/api/cases/:id/briefing", async (request, reply) => {
    const { id } = IdParamsSchema.parse(request.params);
    const workspace = store.getWorkspace(id);
    if (!workspace) return reply.code(404).send({ error: "Case not found." });
    return {
      generatedAt: nowIso(),
      final:
        workspace.routes.length > 0 &&
        workspace.routes.every((route) => route.review !== "pending"),
      case: workspace.case,
      confirmedHazards: workspace.hazards.filter(
        (hazard) => hazard.review === "confirmed",
      ),
      confirmedRoutes: workspace.routes.filter(
        (route) => route.review === "confirmed",
      ),
      unresolvedCandidates: [
        ...workspace.hazards
          .filter((hazard) => hazard.review === "pending")
          .map((hazard) => hazard.id),
        ...workspace.routes
          .filter((route) => route.review === "pending")
          .map((route) => route.id),
      ],
      evidenceManifest: workspace.evidence.map((item) => ({
        id: item.id,
        title: item.title,
        provider: item.sourceProvider,
        rights: item.rights,
        redistributable: item.redistributable,
        confidence: item.confidence,
      })),
      warning:
        "Candidate intelligence only. Verify conditions on arrival and follow agency SOPs and incident command.",
    };
  });

  app.get("/api/cases/:id/events", async (request, reply) => {
    const { id } = IdParamsSchema.parse(request.params);
    if (!store.getCase(id))
      return reply.code(404).send({ error: "Case not found." });
    reply.hijack();
    reply.raw.writeHead(200, {
      "content-type": "text/event-stream",
      "cache-control": "no-cache, no-transform",
      connection: "keep-alive",
      "x-accel-buffering": "no",
    });
    for (const event of store.listEvents(id)) writeSse(reply.raw, event);
    const unsubscribe = eventHub.subscribe(id, (event) =>
      writeSse(reply.raw, event),
    );
    const heartbeat = setInterval(
      () => reply.raw.write(": heartbeat\n\n"),
      15_000,
    );
    heartbeat.unref();
    request.raw.on("close", () => {
      clearInterval(heartbeat);
      unsubscribe();
    });
  });

  await app.register(fastifyStatic, {
    root: config.casesRoot,
    prefix: "/assets/",
    index: false,
    list: false,
    maxAge: 0,
    cacheControl: false,
    setHeaders: (reply, path) => {
      reply.header("x-content-type-options", "nosniff");
      if (extname(path).toLowerCase() === ".splat") {
        reply.type("application/octet-stream");
        reply.header("accept-ranges", "bytes");
      }
    },
  });

  if (existsSync(resolve(config.webDist, "index.html"))) {
    const indexHtml = readFileSync(
      resolve(config.webDist, "index.html"),
      "utf8",
    );
    await app.register(fastifyStatic, {
      root: config.webDist,
      prefix: "/",
      decorateReply: false,
      wildcard: false,
    });
    app.get("/*", async (_request, reply) =>
      reply.type("text/html").send(indexHtml),
    );
  }

  app.addHook("onClose", async () => store.close());
  for (const item of store.listCases()) {
    sceneIntelligence.rebuildSpatialGraph(item.id);
  }
  reconstruction.resumePending();

  return {
    app,
    services: {
      config,
      store,
      eventHub,
      casePipeline,
      reconstruction,
      settings,
      discovery,
      aiAnalyzer,
      sceneIntelligence,
    },
  };
}

function isAuthorized(request: FastifyRequest, config: AppConfig): boolean {
  if (!config.accessKey) return true;
  const authorization = request.headers.authorization;
  if (
    authorization?.startsWith("Bearer ") &&
    safeEqual(authorization.slice(7), config.accessKey)
  )
    return true;
  const signed = request.unsignCookie(request.cookies[SESSION_COOKIE] ?? "");
  return signed.valid && signed.value === "authorized";
}

function safeEqual(left: string, right: string): boolean {
  const leftHash = createHash("sha256").update(left).digest();
  const rightHash = createHash("sha256").update(right).digest();
  return timingSafeEqual(leftHash, rightHash);
}

function resolveCaptureMode(
  requested: CaptureUploadMode,
  photos: ReadonlyArray<{ width?: number; height?: number }>,
): CaptureInfo["projection"] {
  if (requested !== "auto") return requested;
  if (photos.length !== 1) return "perspective";
  const photo = photos[0];
  if (
    photo?.width !== undefined &&
    photo.height !== undefined &&
    photo.width / photo.height >= 1.85
  ) {
    return "panorama_360";
  }
  return "perspective";
}

function captureInfo(
  projection: CaptureInfo["projection"],
  captureOrder: number,
  overlapSetId: string,
  width: number | undefined,
  height: number | undefined,
  projectionSource: CaptureInfo["projectionSource"],
): CaptureInfo {
  return {
    projection,
    width: Math.max(1, width ?? 1),
    height: Math.max(1, height ?? 1),
    horizontalCoverageDegrees:
      projection === "panorama_360"
        ? 360
        : projection === "panorama_180"
          ? 180
          : 62,
    captureOrder,
    overlapSetId,
    projectionSource,
  };
}

function validatePanoramaShape(
  projection: CaptureInfo["projection"],
  photos: ReadonlyArray<{ width?: number; height?: number }>,
): string | undefined {
  if (projection === "perspective") return undefined;
  const photo = photos[0];
  if (!photo?.width || !photo.height) {
    return "The panorama dimensions could not be read.";
  }
  const ratio = photo.width / photo.height;
  if (projection === "panorama_180" && (ratio < 0.9 || ratio > 1.1)) {
    return "A 180° half-sphere panorama must use a 1:1 equirectangular layout.";
  }
  if (projection === "panorama_360" && (ratio < 1.9 || ratio > 2.1)) {
    return "A full 360° panorama must use a 2:1 equirectangular layout.";
  }
  return undefined;
}

function isFloorPlanEvidence(evidence: EvidenceAsset): boolean {
  if (evidence.tags.includes("reconstruction-excluded")) return false;
  if (evidence.visualAnalysis) {
    return (
      evidence.visualAnalysis.sceneType === "floor_plan" ||
      evidence.kind === "blueprint" ||
      evidence.tags.includes("plan-candidate")
    );
  }
  return (
    evidence.kind === "blueprint" || evidence.tags.includes("plan-candidate")
  );
}

async function isLikelyFloorPlan(path: string): Promise<boolean> {
  try {
    const stats = await sharp(path)
      .rotate()
      .resize({ width: 256, height: 256, fit: "inside" })
      .removeAlpha()
      .stats();
    const means = stats.channels.slice(0, 3).map((channel) => channel.mean);
    const mean = means.reduce((sum, value) => sum + value, 0) / means.length;
    const colorSpread = Math.max(...means) - Math.min(...means);
    const deviation =
      stats.channels
        .slice(0, 3)
        .reduce((sum, channel) => sum + channel.stdev, 0) / 3;
    return (
      mean >= 215 &&
      colorSpread <= 22 &&
      deviation >= 8 &&
      deviation <= 65 &&
      stats.entropy >= 0.5 &&
      stats.entropy <= 4.6
    );
  } catch {
    return false;
  }
}

function multipartField(
  fields: Record<string, unknown>,
  name: string,
): string | undefined {
  const field = fields[name];
  const first = Array.isArray(field) ? field[0] : field;
  if (first && typeof first === "object" && "value" in first) {
    const value = (first as { value?: unknown }).value;
    return typeof value === "string" ? value : undefined;
  }
  return undefined;
}

function writeSse(stream: NodeJS.WritableStream, event: unknown): void {
  stream.write(`data: ${JSON.stringify(event)}\n\n`);
}

function errorMessage(error: unknown): string {
  return error instanceof Error && error.message
    ? error.message
    : "Unexpected server error.";
}
