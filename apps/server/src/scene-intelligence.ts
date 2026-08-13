import { readFileSync } from "node:fs";
import { relative, resolve } from "node:path";
import {
  EvidenceVisualAnalysisSchema,
  type AiProviderId,
  type EvidenceAsset,
  type EvidenceVisualAnalysis,
  type FloorHint,
  type ReconstructionArtifact,
  type ReconstructionCameraPose,
  type RoomType,
  type SpatialEdge,
  type SpatialNode,
} from "@structurefirst/contracts";
import sharp from "sharp";
import { z } from "zod";
import type { AppConfig } from "./config.js";
import { confidence } from "./lib/confidence.js";
import { createId, nowIso } from "./lib/ids.js";
import { SettingsService, type ProviderCredential } from "./settings.js";
import { StructureStore } from "./store.js";

const GeneratedVisualAnalysisSchema = z.object({
  sceneType: z.enum([
    "exterior",
    "interior",
    "floor_plan",
    "non_property",
    "unknown",
  ]),
  roomType: z.enum([
    "bedroom",
    "bathroom",
    "kitchen",
    "living_room",
    "dining_room",
    "office",
    "garage",
    "basement",
    "attic",
    "corridor",
    "closet",
    "stair",
    "utility",
    "exterior",
    "unknown",
  ]),
  floorHint: z.enum(["basement", "ground", "upper", "attic", "unknown"]),
  floorNumber: z
    .number()
    .int()
    .min(-5)
    .max(200)
    .nullable()
    .optional()
    .transform((value) => value ?? undefined),
  roomLabels: z.array(z.string().trim().min(1).max(120)).max(40).default([]),
  propertyRelevance: z.enum(["likely", "unlikely", "unknown"]),
  observedAddress: z.string().trim().max(300).default(""),
  connections: z
    .array(z.enum(["door", "corridor", "stair_up", "stair_down", "window"]))
    .max(12)
    .default([]),
  summary: z.string().trim().min(1).max(800),
  confidenceScore: z
    .number()
    .min(0)
    .max(1)
    .nullable()
    .transform((value) => value ?? 0.5),
});

type ChatResponse = {
  choices?: Array<{ message?: { content?: string | null } }>;
};

export type SceneAnalysisRun = {
  analyzed: number;
  rejected: number;
  provider?: AiProviderId;
  model?: string;
  warnings: string[];
};

export class SceneIntelligenceService {
  constructor(
    private readonly store: StructureStore,
    private readonly settings: SettingsService,
    private readonly config: AppConfig,
  ) {}

  async analyzeCase(
    caseId: string,
    evidenceIds?: string[],
    force = false,
  ): Promise<SceneAnalysisRun> {
    const workspace = this.store.getWorkspace(caseId);
    if (!workspace) throw new Error("Case not found.");
    const credential = this.settings.credential();
    if (!credential?.vision) {
      return {
        analyzed: 0,
        rejected: 0,
        warnings: [
          "No enabled vision-capable AI provider is configured; geometric verification remains active.",
        ],
      };
    }
    const requested = evidenceIds ? new Set(evidenceIds) : undefined;
    const candidates = workspace.evidence
      .filter(
        (item) =>
          (!requested || requested.has(item.id)) &&
          Boolean(item.localUrl) &&
          item.mimeType?.startsWith("image/") &&
          (force || !item.visualAnalysis),
      )
      .slice(0, 50);
    const warnings: string[] = [];
    let analyzed = 0;
    let rejected = 0;
    const queue = [...candidates];
    const workers = Array.from(
      { length: Math.min(4, queue.length) },
      async () => {
        while (queue.length) {
          const evidence = queue.shift();
          if (!evidence) return;
          try {
            const analysis = await this.analyzeImage(
              workspace.case.displayAddress,
              workspace.case.addressInput,
              evidence,
              credential,
            );
            this.store.putEvidence(withVisualAnalysis(evidence, analysis));
            analyzed += 1;
            if (
              analysis.sceneType === "non_property" ||
              analysis.propertyRelevance === "unlikely" ||
              analysis.addressMatch === "contradictory"
            ) {
              rejected += 1;
            }
          } catch (error) {
            warnings.push(`${evidence.title}: ${errorMessage(error)}`);
          }
        }
      },
    );
    await Promise.all(workers);
    return {
      analyzed,
      rejected,
      provider: credential.id,
      model: credential.model,
      warnings,
    };
  }

  rebuildSpatialGraph(caseId: string): void {
    const workspace = this.store.getWorkspace(caseId);
    if (!workspace) return;
    this.store.deleteSpatialGraph(caseId);
    if (workspace.case.profile) {
      this.store.putNode({
        id: createId("node"),
        caseId,
        label: "Known exterior location",
        kind: "exterior",
        floorLabel: "Ground floor",
        level: 0,
        sourceIds: [workspace.case.profile.source.id],
        confidence: workspace.case.profile.confidence,
      });
    }

    const seen = new Set<string>();
    const coveredEvidence = new Set<string>();
    const allArtifactNodes: ArtifactSpatialNode[] = [];
    const readyArtifacts = workspace.artifacts
      .filter((item) => item.status === "ready")
      .sort((left, right) => {
        const coverageDifference =
          connectedEvidenceIds(right, this.config.casesRoot).length -
          connectedEvidenceIds(left, this.config.casesRoot).length;
        return (
          coverageDifference || right.updatedAt.localeCompare(left.updatedAt)
        );
      });
    for (const artifact of readyArtifacts) {
      const evidenceIds = connectedEvidenceIds(artifact, this.config.casesRoot);
      const key = [...evidenceIds].sort().join("|");
      if (
        !key ||
        seen.has(key) ||
        evidenceIds.every((id) => coveredEvidence.has(id))
      ) {
        continue;
      }
      seen.add(key);
      for (const id of evidenceIds) coveredEvidence.add(id);
      const evidence = evidenceIds
        .map((id) => workspace.evidence.find((item) => item.id === id))
        .filter((item): item is EvidenceAsset => Boolean(item));
      const poseByEvidence = new Map(
        (artifact.geometry?.cameraPoses ?? []).map((pose) => [
          pose.evidenceId,
          pose.position,
        ]),
      );
      const observedLevelByEvidence = inferObservedLevels(
        artifact.geometry?.cameraPoses ?? [],
      );
      const report = artifact.registration?.reportUrl
        ? registrationReport(
            artifact.registration.reportUrl,
            this.config.casesRoot,
          )
        : undefined;
      const groups = spatialEvidenceGroups(
        evidence,
        artifact.evidenceIds ?? [artifact.evidenceId],
        report?.preflight?.acceptedPairs ?? [],
        poseByEvidence,
        observedLevelByEvidence,
      );

      const artifactNodes: ArtifactSpatialNode[] = [];
      for (const group of groups) {
        const floor = floorPresentation(group.floorHint, group.observedLevel);
        const displayedFloor =
          floor.floorLabel === "Unknown floor"
            ? "Floor not verified"
            : floor.floorLabel;
        const position = centroid(group.positions);
        const node: SpatialNode = {
          id: createId("node"),
          caseId,
          artifactId: artifact.id,
          coordinateFrameId: artifact.id,
          label: `${roomLabel(group.roomType)} · ${displayedFloor}`,
          kind: spatialKind(group.roomType, group.analyses),
          roomType: group.roomType,
          ...(floor.level === undefined ? {} : { level: floor.level }),
          floorLabel: floor.floorLabel,
          ...(position ? { position } : {}),
          sourceIds: group.evidence.map((item) => item.id),
          confidence: confidence(
            artifact.registration?.confidenceScore ?? artifact.confidence.score,
            "reconstructed",
            "derived",
            group.analyses.length
              ? `Room labels were inferred from ${group.analyses.length} registered source ${group.analyses.length === 1 ? "view" : "views"}; the local position is derived from calibrated reconstruction cameras. Named floors require visible or plan evidence.`
              : `Geometry includes ${group.evidence.length} registered source images; the room and floor remain unclassified.`,
            group.evidence.length,
          ),
        };
        this.store.putNode(node);
        artifactNodes.push({
          node,
          ...(position ? { position } : {}),
          connections: new Set(
            group.analyses.flatMap((analysis) => analysis.connections),
          ),
        });
      }
      for (const edge of localRoomEdges(caseId, artifact, artifactNodes)) {
        this.store.putEdge(edge);
      }
      allArtifactNodes.push(...artifactNodes);
    }
    for (const edge of crossArtifactBridgeEdges(caseId, allArtifactNodes)) {
      this.store.putEdge(edge);
    }
  }

  private async analyzeImage(
    displayAddress: string,
    submittedAddress: string,
    evidence: EvidenceAsset,
    credential: ProviderCredential,
  ): Promise<EvidenceVisualAnalysis> {
    const path = localAssetPath(
      this.config.casesRoot,
      evidence.caseId,
      evidence.localUrl,
    );
    if (!path) throw new Error("Local image path is unavailable.");
    const preview = await sharp(readFileSync(path), { failOn: "warning" })
      .rotate()
      .resize({
        width: 1024,
        height: 1024,
        fit: "inside",
        withoutEnlargement: true,
      })
      .jpeg({ quality: 84, mozjpeg: true })
      .toBuffer();
    const metadata = {
      evidenceId: evidence.id,
      title: evidence.title,
      provider: evidence.sourceProvider,
      tags: evidence.tags,
      notes: evidence.notes,
      caseAssignment: "Assigned to the active property by the source record.",
    };
    const system = [
      "You classify one property-evidence image for an emergency structure reconstruction system.",
      "Return JSON only with sceneType, roomType, floorHint, floorNumber, roomLabels, propertyRelevance, observedAddress, connections, summary, confidenceScore.",
      "Use only visible pixels and supplied metadata. Do not infer an exact address from visual similarity.",
      "sceneType is exterior|interior|floor_plan|non_property|unknown.",
      "roomType is bedroom|bathroom|kitchen|living_room|dining_room|office|garage|basement|attic|corridor|closet|stair|utility|exterior|unknown.",
      "Classify only the visible space, not the filename or listing sequence. A room dominated by a washer, dryer, laundry sink, or utility equipment is utility, not kitchen. A patio, yard, facade, or deck is exterior even if it is described as outdoor living space.",
      "floorHint is basement|ground|upper|attic|unknown. Use unknown unless a floor is visibly supported.",
      "For a floor plan, transcribe an explicitly printed floor number such as 1ST FLOOR PLAN as floorNumber=1 and list only visibly printed room names in roomLabels. Otherwise omit floorNumber and return an empty roomLabels array.",
      "propertyRelevance is unlikely for animals, objects, screenshots, unrelated scenes, or imagery that contradicts the metadata; otherwise likely or unknown.",
      "Only transcribe an address or house number that is actually visible. Otherwise observedAddress must be empty.",
      "connections may contain door, corridor, stair_up, stair_down, window.",
    ].join(" ");
    const response = await fetchWithRetry(
      `${credential.baseUrl}/chat/completions`,
      {
        method: "POST",
        headers: this.settings.headers(credential),
        body: JSON.stringify({
          model: credential.model,
          temperature: 0,
          max_tokens: 700,
          stream: false,
          ...(credential.id === "nvidia_nim"
            ? { response_format: NVIDIA_VISUAL_RESPONSE_FORMAT }
            : { response_format: { type: "json_object" } }),
          messages: [
            { role: "system", content: system },
            {
              role: "user",
              content: [
                {
                  type: "text",
                  text: `Classify this exact evidence record:\n${JSON.stringify(metadata)}`,
                },
                {
                  type: "image_url",
                  image_url: {
                    url: `data:image/jpeg;base64,${preview.toString("base64")}`,
                  },
                },
              ],
            },
          ],
        }),
        signal: AbortSignal.timeout(60_000),
      },
    );
    const payload = (await response.json()) as ChatResponse;
    const content = payload.choices?.[0]?.message?.content;
    if (!content)
      throw new Error("The vision model returned no analysis text.");
    let generated: z.infer<typeof GeneratedVisualAnalysisSchema>;
    try {
      generated = GeneratedVisualAnalysisSchema.parse(parseJson(content));
    } catch {
      try {
        generated = await this.normalizeAnalysis(content, credential);
      } catch {
        generated = parseLabeledVisualAnalysis(content);
      }
    }
    generated = normalizeGeneratedScene({
      ...generated,
      floorHint: verifiedFloorHint(generated.floorHint, generated.summary),
    });
    const observedAddress = normalizeObservedAddress(generated.observedAddress);
    const addressMatch = measuredAddressMatch(
      displayAddress,
      submittedAddress,
      observedAddress,
      evidence,
      generated.propertyRelevance,
    );
    return EvidenceVisualAnalysisSchema.parse({
      ...generated,
      addressMatch,
      observedAddress,
      provider: credential.id,
      model: credential.model,
      analyzedAt: nowIso(),
    });
  }

  private async normalizeAnalysis(
    sourceText: string,
    credential: ProviderCredential,
  ): Promise<z.infer<typeof GeneratedVisualAnalysisSchema>> {
    const response = await fetchWithRetry(
      `${credential.baseUrl}/chat/completions`,
      {
        method: "POST",
        headers: this.settings.headers(credential),
        body: JSON.stringify({
          model: credential.model,
          temperature: 0,
          max_tokens: 500,
          stream: false,
          ...(credential.id === "nvidia_nim"
            ? { response_format: NVIDIA_VISUAL_RESPONSE_FORMAT }
            : { response_format: { type: "json_object" } }),
          messages: [
            {
              role: "system",
              content:
                "Convert supplied classification text to the requested JSON object. Preserve only stated visual observations. Never invent or copy an address. Output JSON only.",
            },
            {
              role: "user",
              content: [
                "Return sceneType, roomType, floorHint, floorNumber, roomLabels, propertyRelevance, observedAddress, connections, summary, confidenceScore.",
                "Allowed values are the same as this example:",
                JSON.stringify({
                  sceneType: "interior",
                  roomType: "bedroom",
                  floorHint: "unknown",
                  floorNumber: null,
                  roomLabels: [],
                  propertyRelevance: "likely",
                  observedAddress: "",
                  connections: ["door", "window"],
                  summary: "Visible bedroom with a door and window.",
                  confidenceScore: 0.7,
                }),
                "Source classification:",
                sourceText.slice(0, 4_000),
              ].join("\n"),
            },
          ],
        }),
        signal: AbortSignal.timeout(60_000),
      },
    );
    const payload = (await response.json()) as ChatResponse;
    const content = payload.choices?.[0]?.message?.content;
    if (!content)
      throw new Error("The vision model returned no normalized analysis.");
    return GeneratedVisualAnalysisSchema.parse(parseJson(content));
  }
}

const NVIDIA_VISUAL_RESPONSE_FORMAT = {
  type: "json_schema",
  json_schema: {
    name: "structurefirst_visual_analysis",
    strict: true,
    schema: {
      type: "object",
      properties: {
        sceneType: {
          type: "string",
          enum: [
            "exterior",
            "interior",
            "floor_plan",
            "non_property",
            "unknown",
          ],
        },
        roomType: {
          type: "string",
          enum: [
            "bedroom",
            "bathroom",
            "kitchen",
            "living_room",
            "dining_room",
            "office",
            "garage",
            "basement",
            "attic",
            "corridor",
            "closet",
            "stair",
            "utility",
            "exterior",
            "unknown",
          ],
        },
        floorHint: {
          type: "string",
          enum: ["basement", "ground", "upper", "attic", "unknown"],
        },
        floorNumber: {
          anyOf: [
            { type: "integer", minimum: -5, maximum: 200 },
            { type: "null" },
          ],
        },
        roomLabels: {
          type: "array",
          items: { type: "string" },
          maxItems: 40,
        },
        propertyRelevance: {
          type: "string",
          enum: ["likely", "unlikely", "unknown"],
        },
        observedAddress: { type: "string" },
        connections: {
          type: "array",
          items: {
            type: "string",
            enum: ["door", "corridor", "stair_up", "stair_down", "window"],
          },
          maxItems: 12,
        },
        summary: { type: "string" },
        confidenceScore: { type: "number", minimum: 0, maximum: 1 },
      },
      required: [
        "sceneType",
        "roomType",
        "floorHint",
        "floorNumber",
        "roomLabels",
        "propertyRelevance",
        "observedAddress",
        "connections",
        "summary",
        "confidenceScore",
      ],
      additionalProperties: false,
    },
  },
} as const;

export function normalizeGeneratedScene(
  generated: z.infer<typeof GeneratedVisualAnalysisSchema>,
): z.infer<typeof GeneratedVisualAnalysisSchema> {
  if (
    generated.sceneType === "floor_plan" ||
    generated.sceneType === "non_property"
  ) {
    return generated;
  }
  const definitiveIndoorRooms = new Set<RoomType>([
    "bedroom",
    "bathroom",
    "kitchen",
    "living_room",
    "dining_room",
    "office",
    "garage",
    "basement",
    "attic",
    "corridor",
    "closet",
    "stair",
    "utility",
  ]);
  const summary = generated.summary.toLowerCase();
  const inferredRoom = inferRoomFromSummary(summary);
  const laundryVisible = /\b(?:washer|dryer|laundry|utility sink)\b/.test(
    summary,
  );
  const kitchenVisible = /\b(?:oven|stove|cooktop|range|kitchen island)\b/.test(
    summary,
  );
  const roomType: RoomType =
    generated.roomType === "kitchen" && laundryVisible && !kitchenVisible
      ? "utility"
      : generated.roomType === "unknown" && inferredRoom
        ? inferredRoom
        : generated.roomType;
  const sceneType =
    roomType === "exterior"
      ? "exterior"
      : generated.sceneType === "exterior" &&
          definitiveIndoorRooms.has(roomType)
        ? "interior"
        : definitiveIndoorRooms.has(roomType) &&
            generated.sceneType === "unknown"
          ? "interior"
          : generated.sceneType;
  const propertyRelevance =
    generated.sceneType === "unknown" &&
    generated.propertyRelevance === "unlikely" &&
    inferredRoom
      ? "likely"
      : generated.propertyRelevance;
  return { ...generated, roomType, sceneType, propertyRelevance };
}

function inferRoomFromSummary(summary: string): RoomType | undefined {
  const matchesAtLeast = (terms: RegExp[], required: number) =>
    terms.filter((term) => term.test(summary)).length >= required;
  if (
    /\bkitchen\b/.test(summary) &&
    matchesAtLeast(
      [
        /\bstove\b/,
        /\b(?:oven|cooktop|range)\b/,
        /\bsink\b/,
        /\brefrigerator\b/,
        /\bcabinet/,
        /\bisland\b/,
      ],
      2,
    )
  )
    return "kitchen";
  if (
    /\bbathroom\b/.test(summary) &&
    matchesAtLeast(
      [/\btoilet\b/, /\bsink\b/, /\b(?:shower|tub|bathtub)\b/, /\bvanity\b/],
      2,
    )
  )
    return "bathroom";
  if (
    /\b(?:laundry|utility room)\b/.test(summary) &&
    /\b(?:washer|dryer)\b/.test(summary)
  )
    return "utility";
  if (/\bbedroom\b/.test(summary) && /\bbed\b/.test(summary)) return "bedroom";
  if (
    /\bliving room\b/.test(summary) &&
    /\b(?:sofa|couch|coffee table|fireplace)\b/.test(summary)
  )
    return "living_room";
  if (/\bdining room\b/.test(summary) && /\b(?:table|chair)\b/.test(summary))
    return "dining_room";
  if (
    /\bgarage\b/.test(summary) &&
    /\b(?:garage door|vehicle|car)\b/.test(summary)
  )
    return "garage";
  if (/\b(?:hallway|corridor)\b/.test(summary) && /\bdoor/.test(summary))
    return "corridor";
  if (/\b(?:staircase|stairs|stairway)\b/.test(summary)) return "stair";
  return undefined;
}

function withVisualAnalysis(
  evidence: EvidenceAsset,
  analysis: EvidenceVisualAnalysis,
): EvidenceAsset {
  const locallyDetectedPlan = evidence.tags.includes("plan-candidate");
  const generatedTags = [
    "vlm-analyzed",
    `scene:${analysis.sceneType}`,
    `room:${analysis.roomType}`,
    `floor:${analysis.floorHint}`,
    ...(analysis.floorNumber === undefined
      ? []
      : [`floor-number:${analysis.floorNumber}`]),
    `property-relevance:${analysis.propertyRelevance}`,
    `address-match:${analysis.addressMatch}`,
    ...analysis.connections.map((item) => `connection:${item}`),
  ];
  if (
    analysis.sceneType === "non_property" ||
    analysis.propertyRelevance === "unlikely" ||
    analysis.addressMatch === "contradictory"
  ) {
    generatedTags.push("reconstruction-excluded");
  }
  return {
    ...evidence,
    kind:
      analysis.sceneType === "floor_plan" || locallyDetectedPlan
        ? "blueprint"
        : "image",
    visualAnalysis: analysis,
    tags: [
      ...new Set([
        ...evidence.tags.filter(
          (tag) =>
            !/^(?:vlm-analyzed|scene:|room:|floor:|floor-number:|property-relevance:|address-match:|connection:|reconstruction-excluded)/.test(
              tag,
            ),
        ),
        ...generatedTags,
      ]),
    ],
    notes: `${evidence.notes} VLM classification: ${analysis.summary}`,
  };
}

function measuredAddressMatch(
  displayAddress: string,
  submittedAddress: string,
  observedAddress: string,
  evidence: EvidenceAsset,
  relevance: "likely" | "unlikely" | "unknown",
): EvidenceVisualAnalysis["addressMatch"] {
  if (relevance === "unlikely") return "contradictory";
  // An operator explicitly assigned this file to the property. A VLM-only
  // address transcription is not independent OCR evidence and must not
  // silently override that assignment when the model hallucinates a number.
  if (evidence.tags.includes("operator-upload")) return "possible";
  const expectedNumber =
    houseNumber(displayAddress) ?? houseNumber(submittedAddress);
  const observedNumber = houseNumber(observedAddress);
  if (expectedNumber && observedNumber)
    return expectedNumber === observedNumber ? "supported" : "contradictory";
  if (
    evidence.tags.includes("operator-upload") ||
    evidence.tags.includes("listing-address-match") ||
    evidence.tags.includes("address-text-match")
  ) {
    return "possible";
  }
  return "unknown";
}

function houseNumber(value: string): string | undefined {
  return value.match(/(?:^|\s)(\d+[A-Za-z]?)(?:\s|$)/)?.[1]?.toLowerCase();
}

function normalizeObservedAddress(value: string): string {
  const trimmed = value.trim();
  return /^(?:possible|unknown|none|n\/?a|not visible|not provided)$/i.test(
    trimmed,
  )
    ? ""
    : trimmed;
}

function connectedEvidenceIds(
  artifact: ReconstructionArtifact,
  casesRoot: string,
): string[] {
  const all = artifact.evidenceIds ?? [artifact.evidenceId];
  if (artifact.fallback) return [artifact.fallback.sourceEvidenceId];
  if (!artifact.registration) return all.slice(0, 1);
  const report = artifact.registration.reportUrl
    ? registrationReport(artifact.registration.reportUrl, casesRoot)
    : undefined;
  const connected = Array.isArray(report?.connectedFrames)
    ? report.connectedFrames.filter((value): value is number =>
        Number.isInteger(value),
      )
    : undefined;
  return connected?.length
    ? connected.flatMap((index) => (all[index] ? [all[index]!] : []))
    : all.slice(0, artifact.registration.connectedFrameCount);
}

function registrationReport(
  reportUrl: string,
  casesRoot: string,
):
  | {
      connectedFrames?: unknown[];
      preflight?: {
        acceptedPairs?: Array<{
          frameA?: unknown;
          frameB?: unknown;
          confidence?: unknown;
        }>;
      };
    }
  | undefined {
  if (!reportUrl.startsWith("/assets/")) return undefined;
  try {
    const suffix = decodeURIComponent(reportUrl.slice("/assets/".length));
    const path = resolve(casesRoot, suffix);
    return JSON.parse(readFileSync(path, "utf8")) as {
      connectedFrames?: unknown[];
      preflight?: {
        acceptedPairs?: Array<{
          frameA?: unknown;
          frameB?: unknown;
          confidence?: unknown;
        }>;
      };
    };
  } catch {
    return undefined;
  }
}

function localAssetPath(
  casesRoot: string,
  caseId: string,
  localUrl?: string,
): string | undefined {
  if (!localUrl?.startsWith("/assets/")) return undefined;
  const suffix = decodeURIComponent(localUrl.slice("/assets/".length));
  if (suffix.split(/[\\/]/)[0] !== caseId) return undefined;
  const candidate = resolve(casesRoot, suffix);
  const traversal = relative(casesRoot, candidate);
  return traversal.startsWith("..") || resolve(traversal) === traversal
    ? undefined
    : candidate;
}

async function fetchWithRetry(
  url: string,
  init: RequestInit,
): Promise<Response> {
  for (let attempt = 0; attempt < 3; attempt += 1) {
    const response = await fetch(url, init);
    if (response.ok) return response;
    const body = await response.text();
    if (attempt === 2 || ![429, 500, 502, 503, 504].includes(response.status)) {
      throw new Error(
        `Vision provider returned ${response.status}: ${body.slice(0, 300)}`,
      );
    }
    await new Promise((resolveDelay) =>
      setTimeout(resolveDelay, 750 * (attempt + 1)),
    );
  }
  throw new Error("Vision provider retry loop ended unexpectedly.");
}

function parseJson(text: string): unknown {
  const trimmed = text.trim();
  const fenced = trimmed.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
  const candidate = fenced?.[1] ?? trimmed;
  try {
    return JSON.parse(candidate);
  } catch {
    const extracted = firstJsonObject(candidate);
    if (extracted) return JSON.parse(extracted);
    const preview = candidate.replace(/\s+/g, " ").slice(0, 240);
    throw new Error(
      `The vision provider did not return valid JSON${preview ? `: ${preview}` : "."}`,
    );
  }
}

function firstJsonObject(value: string): string | undefined {
  for (
    let start = value.indexOf("{");
    start >= 0;
    start = value.indexOf("{", start + 1)
  ) {
    let depth = 0;
    let quoted = false;
    let escaped = false;
    for (let index = start; index < value.length; index += 1) {
      const character = value[index]!;
      if (quoted) {
        if (escaped) escaped = false;
        else if (character === "\\") escaped = true;
        else if (character === '"') quoted = false;
        continue;
      }
      if (character === '"') quoted = true;
      else if (character === "{") depth += 1;
      else if (character === "}") {
        depth -= 1;
        if (depth === 0) return value.slice(start, index + 1);
      }
    }
  }
  return undefined;
}

function parseLabeledVisualAnalysis(
  value: string,
): z.infer<typeof GeneratedVisualAnalysisSchema> {
  const plain = value.replace(/[*`#]/g, "").replace(/\r/g, "").trim();
  const sceneText = labeledValue(plain, "scene(?:\\s+type|Type)") ?? plain;
  const roomText = labeledValue(plain, "room(?:\\s+type|Type)") ?? plain;
  const floorText = labeledValue(plain, "floor(?:\\s+hint|Hint)") ?? "unknown";
  const relevanceText =
    labeledValue(plain, "property(?:\\s+relevance|Relevance)") ?? "unknown";
  const addressText =
    labeledValue(plain, "observed(?:\\s+address|Address)") ?? "";
  const connectionText = labeledValue(plain, "connections?") ?? "";
  const summary =
    labeledValue(plain, "summary") ?? plain.replace(/\s+/g, " ").slice(0, 780);

  const roomType = roomToken(roomText);
  const sceneType = token(
    sceneText,
    ["floor_plan", "non_property", "exterior", "interior", "unknown"] as const,
    roomType !== "unknown" ? "interior" : "unknown",
  );
  const floorHint = token(
    floorText,
    ["basement", "ground", "upper", "attic", "unknown"] as const,
    "unknown",
  );
  const propertyRelevance = token(
    relevanceText,
    ["unlikely", "likely", "unknown"] as const,
    sceneType === "non_property"
      ? "unlikely"
      : sceneType === "interior" || sceneType === "exterior"
        ? "likely"
        : "unknown",
  );
  const connections = (
    ["stair_down", "stair_up", "corridor", "window", "door"] as const
  ).filter((connection) =>
    new RegExp(`\\b${connection.replace("_", "[ _-]")}\\b`, "i").test(
      connectionText,
    ),
  );
  const confidenceText = labeledValue(plain, "confidence(?:\\s+score|Score)?");
  const confidenceValue = confidenceText?.match(/\d+(?:\.\d+)?/)?.[0];
  const rawConfidence = confidenceValue ? Number(confidenceValue) : 0.5;
  const confidenceScore =
    rawConfidence > 1 ? rawConfidence / 100 : rawConfidence;
  return GeneratedVisualAnalysisSchema.parse({
    sceneType,
    roomType,
    floorHint,
    propertyRelevance,
    observedAddress: normalizeObservedAddress(addressText),
    connections,
    summary: summary || "The vision model returned no visual summary.",
    confidenceScore: Math.max(0, Math.min(1, confidenceScore)),
  });
}

function labeledValue(value: string, label: string): string | undefined {
  return value
    .match(new RegExp(`(?:^|\\n)\\s*${label}\\s*[:=-]\\s*([^\\n]+)`, "i"))?.[1]
    ?.trim();
}

function roomToken(value: string): RoomType {
  return token(
    value,
    [
      "living_room",
      "dining_room",
      "bathroom",
      "bedroom",
      "kitchen",
      "corridor",
      "basement",
      "utility",
      "exterior",
      "garage",
      "office",
      "closet",
      "attic",
      "stair",
      "unknown",
    ] as const,
    "unknown",
  );
}

function token<const T extends readonly string[]>(
  value: string,
  allowed: T,
  fallback: T[number],
): T[number] {
  const normalized = value.toLowerCase().replace(/[\s-]+/g, "_");
  return allowed.find((item) => normalized.includes(item)) ?? fallback;
}

type SpatialEvidenceGroup = {
  roomType: RoomType;
  floorHint: FloorHint;
  observedLevel?: number;
  evidence: EvidenceAsset[];
  analyses: EvidenceVisualAnalysis[];
  positions: Array<[number, number, number]>;
};

export function spatialEvidenceGroups(
  evidence: EvidenceAsset[],
  sourceOrder: string[],
  acceptedPairs: Array<{
    frameA?: unknown;
    frameB?: unknown;
    confidence?: unknown;
  }>,
  poseByEvidence: Map<string, [number, number, number]>,
  observedLevelByEvidence = new Map<string, number>(),
): SpatialEvidenceGroup[] {
  const byId = new Map(evidence.map((item) => [item.id, item]));
  const parent = new Map(evidence.map((item) => [item.id, item.id]));
  const root = (id: string): string => {
    const next = parent.get(id) ?? id;
    if (next === id) return id;
    const resolved = root(next);
    parent.set(id, resolved);
    return resolved;
  };
  const join = (leftId: string, rightId: string): void => {
    const left = byId.get(leftId);
    const right = byId.get(rightId);
    if (
      !left ||
      !right ||
      !sameObservedSpace(left, right, observedLevelByEvidence)
    )
      return;
    const leftRoot = root(leftId);
    const rightRoot = root(rightId);
    if (leftRoot !== rightRoot) parent.set(rightRoot, leftRoot);
  };

  let measuredEdgeCount = 0;
  for (const pair of acceptedPairs) {
    if (!Number.isInteger(pair.frameA) || !Number.isInteger(pair.frameB)) {
      continue;
    }
    const leftId = sourceOrder[Number(pair.frameA)];
    const rightId = sourceOrder[Number(pair.frameB)];
    if (!leftId || !rightId || !byId.has(leftId) || !byId.has(rightId)) {
      continue;
    }
    measuredEdgeCount += 1;
    join(leftId, rightId);
  }

  if (measuredEdgeCount === 0) {
    const ordered = [...evidence].sort(
      (left, right) =>
        (left.capture?.captureOrder ?? Number.MAX_SAFE_INTEGER) -
        (right.capture?.captureOrder ?? Number.MAX_SAFE_INTEGER),
    );
    for (let index = 1; index < ordered.length; index += 1) {
      const left = ordered[index - 1]!;
      const right = ordered[index]!;
      const leftPosition = poseByEvidence.get(left.id);
      const rightPosition = poseByEvidence.get(right.id);
      if (!leftPosition || !rightPosition) continue;
      const distance = Math.hypot(
        leftPosition[0] - rightPosition[0],
        leftPosition[1] - rightPosition[1],
        leftPosition[2] - rightPosition[2],
      );
      if (distance <= 3) join(left.id, right.id);
    }
  }

  const grouped = new Map<string, EvidenceAsset[]>();
  for (const item of evidence) {
    const groupRoot = root(item.id);
    grouped.set(groupRoot, [...(grouped.get(groupRoot) ?? []), item]);
  }
  return [...grouped.values()].map((items) => {
    const analyses = items
      .map((item) => item.visualAnalysis)
      .filter((item): item is EvidenceVisualAnalysis => Boolean(item));
    const roomType = majority(
      analyses
        .map((item) => item.roomType)
        .filter((item) => item !== "unknown"),
      "unknown" as RoomType,
    );
    const floorHint = majority(
      analyses
        .map((item) => item.floorHint)
        .filter((item) => item !== "unknown"),
      "unknown" as FloorHint,
    );
    const observedLevels = items.flatMap((item) => {
      const level = observedLevelByEvidence.get(item.id);
      return level === undefined ? [] : [level];
    });
    return {
      roomType,
      floorHint,
      ...(observedLevels.length
        ? { observedLevel: numericMode(observedLevels) }
        : {}),
      evidence: items,
      analyses,
      positions: items.flatMap((item) => {
        const position = poseByEvidence.get(item.id);
        return position ? [position] : [];
      }),
    };
  });
}

function sameObservedSpace(
  left: EvidenceAsset,
  right: EvidenceAsset,
  observedLevelByEvidence: Map<string, number>,
): boolean {
  const leftRoom = left.visualAnalysis?.roomType ?? "unknown";
  const rightRoom = right.visualAnalysis?.roomType ?? "unknown";
  const leftFloor = left.visualAnalysis?.floorHint ?? "unknown";
  const rightFloor = right.visualAnalysis?.floorHint ?? "unknown";
  const roomCompatible =
    leftRoom === "unknown" || rightRoom === "unknown" || leftRoom === rightRoom;
  const floorCompatible =
    leftFloor === "unknown" ||
    rightFloor === "unknown" ||
    leftFloor === rightFloor;
  const leftLevel = observedLevelByEvidence.get(left.id);
  const rightLevel = observedLevelByEvidence.get(right.id);
  const observedLevelCompatible =
    leftLevel === undefined ||
    rightLevel === undefined ||
    leftLevel === rightLevel;
  return roomCompatible && floorCompatible && observedLevelCompatible;
}

function numericMode(values: number[]): number {
  const counts = new Map<number, number>();
  for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1);
  return [...counts.entries()].sort(
    (left, right) => right[1] - left[1] || left[0] - right[0],
  )[0]![0];
}

export function inferObservedLevels(
  poses: ReconstructionCameraPose[],
): Map<string, number> {
  const result = new Map<string, number>();
  const finite = poses.filter(
    (pose) =>
      pose.position.every(Number.isFinite) &&
      pose.rotationWxyz.every(Number.isFinite),
  );
  if (!finite.length) return result;

  const referenceUp = rotateByQuaternion(finite[0]!.rotationWxyz, [0, -1, 0]);
  const accumulatedUp = finite.reduce(
    (sum, pose) => {
      let up = rotateByQuaternion(pose.rotationWxyz, [0, -1, 0]);
      if (dot(up, referenceUp) < 0) up = scaleVector(up, -1);
      return addVectors(sum, up);
    },
    [0, 0, 0] as [number, number, number],
  );
  const length = Math.hypot(...accumulatedUp);
  const up =
    length > 1e-6 ? scaleVector(accumulatedUp, 1 / length) : referenceUp;
  const elevations = finite.map((pose) => ({
    id: pose.evidenceId,
    elevation: dot(pose.position, up),
  }));
  const minimum = Math.min(...elevations.map((item) => item.elevation));
  const maximum = Math.max(...elevations.map((item) => item.elevation));
  const hasVerticalSeparation = maximum - minimum >= 2;
  for (const item of elevations) {
    result.set(
      item.id,
      hasVerticalSeparation
        ? Math.max(0, Math.round((item.elevation - minimum) / 2.8))
        : 0,
    );
  }
  return result;
}

function rotateByQuaternion(
  [w, x, y, z]: [number, number, number, number],
  vector: [number, number, number],
): [number, number, number] {
  const quaternionVector: [number, number, number] = [x, y, z];
  const twiceCross = scaleVector(cross(quaternionVector, vector), 2);
  return addVectors(
    vector,
    addVectors(scaleVector(twiceCross, w), cross(quaternionVector, twiceCross)),
  );
}

function cross(
  left: [number, number, number],
  right: [number, number, number],
): [number, number, number] {
  return [
    left[1] * right[2] - left[2] * right[1],
    left[2] * right[0] - left[0] * right[2],
    left[0] * right[1] - left[1] * right[0],
  ];
}

function dot(
  left: [number, number, number],
  right: [number, number, number],
): number {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

function addVectors(
  left: [number, number, number],
  right: [number, number, number],
): [number, number, number] {
  return [left[0] + right[0], left[1] + right[1], left[2] + right[2]];
}

function scaleVector(
  value: [number, number, number],
  scale: number,
): [number, number, number] {
  return [value[0] * scale, value[1] * scale, value[2] * scale];
}

function majority<T extends string>(values: T[], fallback: T): T {
  if (!values.length) return fallback;
  const counts = new Map<T, number>();
  for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1);
  return (
    [...counts.entries()].sort((left, right) => right[1] - left[1])[0]?.[0] ??
    fallback
  );
}

function centroid(
  positions: Array<[number, number, number]>,
): [number, number, number] | undefined {
  const finite = positions.filter((position) =>
    position.every((value) => Number.isFinite(value)),
  );
  if (!finite.length) return undefined;
  const sums = finite.reduce(
    (sum, position) => [
      sum[0] + position[0],
      sum[1] + position[1],
      sum[2] + position[2],
    ],
    [0, 0, 0] as [number, number, number],
  );
  return [
    sums[0] / finite.length,
    sums[1] / finite.length,
    sums[2] / finite.length,
  ];
}

type ArtifactSpatialNode = {
  node: SpatialNode;
  position?: [number, number, number];
  connections: Set<EvidenceVisualAnalysis["connections"][number]>;
};

function crossArtifactBridgeEdges(
  caseId: string,
  nodes: ArtifactSpatialNode[],
): SpatialEdge[] {
  const edges: SpatialEdge[] = [];
  const signatures = new Set<string>();
  for (let leftIndex = 0; leftIndex < nodes.length; leftIndex += 1) {
    const left = nodes[leftIndex]!;
    if (!left.node.artifactId) continue;
    const leftSources = new Set(left.node.sourceIds);
    for (
      let rightIndex = leftIndex + 1;
      rightIndex < nodes.length;
      rightIndex += 1
    ) {
      const right = nodes[rightIndex]!;
      if (
        !right.node.artifactId ||
        right.node.artifactId === left.node.artifactId
      ) {
        continue;
      }
      const shared = right.node.sourceIds.filter((id) => leftSources.has(id));
      if (!shared.length) continue;
      const signature = [left.node.id, right.node.id].sort().join("|");
      if (signatures.has(signature)) continue;
      signatures.add(signature);
      const connections = new Set([...left.connections, ...right.connections]);
      const traversal = connections.has("stair_up")
        ? "stair_up"
        : connections.has("stair_down")
          ? "stair_down"
          : connections.has("door")
            ? "door"
            : "walk";
      edges.push({
        id: createId("edge"),
        caseId,
        from: left.node.id,
        to: right.node.id,
        traversal,
        distanceMeters: 0.1,
        blocked: false,
        sourceIds: shared,
        confidence: confidence(
          0.82,
          "verified",
          "derived",
          "These reconstruction segments share exact source frames. The bridge is verified, but each splat remains in its own local coordinate frame until house-level similarity alignment is complete.",
          shared.length,
        ),
      });
    }
  }
  return edges;
}

function localRoomEdges(
  caseId: string,
  artifact: ReconstructionArtifact,
  nodes: Array<{
    node: SpatialNode;
    position?: [number, number, number];
    connections: Set<EvidenceVisualAnalysis["connections"][number]>;
  }>,
): SpatialEdge[] {
  const candidates: Array<{
    left: (typeof nodes)[number];
    right: (typeof nodes)[number];
    distance: number;
  }> = [];
  for (let leftIndex = 0; leftIndex < nodes.length; leftIndex += 1) {
    const left = nodes[leftIndex];
    if (!left?.position) continue;
    for (
      let rightIndex = leftIndex + 1;
      rightIndex < nodes.length;
      rightIndex += 1
    ) {
      const right = nodes[rightIndex];
      if (!right?.position) continue;
      const semanticConnections = new Set([
        ...left.connections,
        ...right.connections,
      ]);
      if (
        !["door", "corridor", "stair_up", "stair_down"].some((value) =>
          semanticConnections.has(
            value as EvidenceVisualAnalysis["connections"][number],
          ),
        )
      ) {
        continue;
      }
      const distance = Math.hypot(
        left.position[0] - right.position[0],
        left.position[1] - right.position[1],
        left.position[2] - right.position[2],
      );
      if (distance <= 12) candidates.push({ left, right, distance });
    }
  }
  candidates.sort((left, right) => left.distance - right.distance);
  const parent = new Map(nodes.map((item) => [item.node.id, item.node.id]));
  const root = (id: string): string => {
    const next = parent.get(id) ?? id;
    if (next === id) return id;
    const resolved = root(next);
    parent.set(id, resolved);
    return resolved;
  };
  const edges: SpatialEdge[] = [];
  for (const candidate of candidates) {
    const leftRoot = root(candidate.left.node.id);
    const rightRoot = root(candidate.right.node.id);
    if (leftRoot === rightRoot) continue;
    parent.set(leftRoot, rightRoot);
    const semanticConnections = new Set([
      ...candidate.left.connections,
      ...candidate.right.connections,
    ]);
    const traversal = semanticConnections.has("stair_up")
      ? "stair_up"
      : semanticConnections.has("stair_down")
        ? "stair_down"
        : semanticConnections.has("door")
          ? "door"
          : "walk";
    const sourceIds = [
      ...new Set([
        ...candidate.left.node.sourceIds,
        ...candidate.right.node.sourceIds,
      ]),
    ];
    edges.push({
      id: createId("edge"),
      caseId,
      from: candidate.left.node.id,
      to: candidate.right.node.id,
      traversal,
      distanceMeters: Math.max(0.1, candidate.distance),
      blocked: false,
      sourceIds,
      confidence: confidence(
        Math.min(
          0.72,
          artifact.registration?.confidenceScore ?? artifact.confidence.score,
        ),
        "estimated",
        "derived",
        "Candidate connection inferred inside one calibrated reconstruction frame from camera proximity plus a visible door, corridor, or stair. Verify before tactical use.",
        sourceIds.length,
      ),
    });
  }
  return edges;
}

function floorPresentation(
  floorHint: FloorHint,
  observedLevel?: number,
): {
  floorLabel: SpatialNode["floorLabel"];
  level?: number;
} {
  if (floorHint === "basement") return { floorLabel: "Basement", level: -1 };
  if (floorHint === "ground") return { floorLabel: "Ground floor", level: 0 };
  if (floorHint === "upper") return { floorLabel: "Upper floor" };
  if (floorHint === "attic") return { floorLabel: "Attic" };
  return {
    floorLabel: "Unknown floor",
    ...(observedLevel === undefined ? {} : { level: observedLevel }),
  };
}

function verifiedFloorHint(
  floorHint: FloorHint,
  visualSummary: string,
): FloorHint {
  if (floorHint === "unknown") return floorHint;
  const summary = visualSummary.toLowerCase();
  const evidence: Record<Exclude<FloorHint, "unknown">, RegExp> = {
    basement: /\b(?:basement|below grade|underground|foundation wall)\b/,
    ground: /\b(?:ground floor|street level|grade-level|at grade)\b/,
    upper: /\b(?:upper floor|second floor|third floor|upstairs)\b/,
    attic: /\b(?:attic|loft|eaves|roof rafters)\b/,
  };
  return evidence[floorHint].test(summary) ? floorHint : "unknown";
}

function roomLabel(roomType: RoomType): string {
  if (roomType === "unknown") return "Observed space";
  return roomType
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function spatialKind(
  roomType: RoomType,
  analyses: EvidenceVisualAnalysis[],
): SpatialNode["kind"] {
  if (roomType === "stair") return "stair";
  if (roomType === "corridor") return "corridor";
  if (roomType === "exterior") return "exterior";
  if (analyses.some((item) => item.sceneType === "exterior")) return "exterior";
  return "room";
}

function errorMessage(error: unknown): string {
  return error instanceof Error
    ? error.message
    : "Unknown scene-analysis error";
}
