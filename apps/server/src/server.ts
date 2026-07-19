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
import { z } from "zod";
import {
  AddEvidenceLinkInputSchema,
  AiCaseAnalysisInputSchema,
  AiProviderIdSchema,
  CreateCaseInputSchema,
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
    const result = await sceneIntelligence.analyzeCase(id, undefined, true);
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
    const pending: Array<{ asset: EvidenceAsset; outputPath: string }> = [];
    const writtenPaths: string[] = [];
    const overlapSetId = createId("capture_set");
    let totalBytes = 0;

    try {
      const files = request.files({
        limits: {
          fileSize: MAX_UPLOAD_BYTES,
          files: MAX_UPLOAD_FILES,
          parts: MAX_UPLOAD_FILES,
        },
      });
      for await (const part of files) {
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

        pending.push({
          outputPath,
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

    const assets = pending.map((item) => store.putEvidence(item.asset));
    await sceneIntelligence.analyzeCase(
      id,
      assets.map((item) => item.id),
    );
    const analyzedAssets = assets.map(
      (item) => store.getEvidence(item.id) ?? item,
    );
    const selected = analyzedAssets
      .filter((item) => !item.tags.includes("reconstruction-excluded"))
      .slice(0, 12);
    if (selected.length === 0) {
      const result: PhotoUploadResult = {
        assets: analyzedAssets,
        note: "The vision check excluded every upload as non-property or contradictory imagery; no reconstruction was started.",
      };
      return reply.code(201).send(result);
    }
    const artifact =
      selected.length === 1
        ? await reconstruction.queue(id, {
            evidenceId: selected[0]!.id,
            mode: "single_image",
          })
        : await reconstruction.queueMulti(id, {
            evidenceIds: selected.map((item) => item.id),
          });
    const result: PhotoUploadResult = {
      assets: analyzedAssets,
      artifact,
      note:
        assets.length > selected.length
          ? `Uploaded ${assets.length} photos. LucidFrame is connecting the first ${selected.length}; all photos remain saved.`
          : `Uploaded ${assets.length} ${assets.length === 1 ? "photo" : "photos"} and started LucidFrame.`,
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
