import { readFileSync } from "node:fs";
import { relative, resolve } from "node:path";
import {
  EvidenceVisualAnalysisSchema,
  type AiProviderId,
  type EvidenceAsset,
  type EvidenceVisualAnalysis,
  type FloorHint,
  type ReconstructionArtifact,
  type RoomType,
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
      .slice(0, 12);
    const warnings: string[] = [];
    let analyzed = 0;
    let rejected = 0;
    for (const evidence of candidates) {
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
    for (const artifact of workspace.artifacts.filter(
      (item) => item.status === "ready",
    )) {
      const evidenceIds = connectedEvidenceIds(artifact, this.config.casesRoot);
      const key = [...evidenceIds].sort().join("|");
      if (!key || seen.has(key)) continue;
      seen.add(key);
      const evidence = evidenceIds
        .map((id) => workspace.evidence.find((item) => item.id === id))
        .filter((item): item is EvidenceAsset => Boolean(item));
      const analyses = evidence
        .map((item) => item.visualAnalysis)
        .filter((item): item is EvidenceVisualAnalysis => Boolean(item));
      const roomType = majority(
        analyses.map((item) => item.roomType),
        "unknown",
      );
      const floorHint = majority(
        analyses.map((item) => item.floorHint),
        "unknown",
      );
      const floor = floorPresentation(floorHint);
      const node: SpatialNode = {
        id: createId("node"),
        caseId,
        label: `${roomLabel(roomType)} · ${floor.floorLabel}`,
        kind: spatialKind(roomType, analyses),
        ...(floor.level === undefined ? {} : { level: floor.level }),
        floorLabel: floor.floorLabel,
        sourceIds: evidenceIds,
        confidence: confidence(
          artifact.registration?.confidenceScore ?? artifact.confidence.score,
          "reconstructed",
          "derived",
          analyses.length
            ? `Room and floor labels were inferred by a configured VLM; geometry includes only ${evidenceIds.length} registered source images.`
            : `Geometry includes ${evidenceIds.length} registered source images; the room and floor remain unclassified.`,
          evidenceIds.length,
        ),
      };
      this.store.putNode(node);
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
      "Return JSON only with sceneType, roomType, floorHint, propertyRelevance, observedAddress, connections, summary, confidenceScore.",
      "Use only visible pixels and supplied metadata. Do not infer an exact address from visual similarity.",
      "sceneType is exterior|interior|floor_plan|non_property|unknown.",
      "roomType is bedroom|bathroom|kitchen|living_room|dining_room|office|garage|basement|attic|corridor|closet|stair|utility|exterior|unknown.",
      "floorHint is basement|ground|upper|attic|unknown. Use unknown unless a floor is visibly supported.",
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
    generated = {
      ...generated,
      floorHint: verifiedFloorHint(generated.floorHint, generated.summary),
    };
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
                "Return sceneType, roomType, floorHint, propertyRelevance, observedAddress, connections, summary, confidenceScore.",
                "Allowed values are the same as this example:",
                JSON.stringify({
                  sceneType: "interior",
                  roomType: "bedroom",
                  floorHint: "unknown",
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

function withVisualAnalysis(
  evidence: EvidenceAsset,
  analysis: EvidenceVisualAnalysis,
): EvidenceAsset {
  const generatedTags = [
    "vlm-analyzed",
    `scene:${analysis.sceneType}`,
    `room:${analysis.roomType}`,
    `floor:${analysis.floorHint}`,
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
    visualAnalysis: analysis,
    tags: [
      ...new Set([
        ...evidence.tags.filter(
          (tag) =>
            !/^(?:vlm-analyzed|scene:|room:|floor:|property-relevance:|address-match:|connection:|reconstruction-excluded)/.test(
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
): { connectedFrames?: unknown[] } | undefined {
  if (!reportUrl.startsWith("/assets/")) return undefined;
  try {
    const suffix = decodeURIComponent(reportUrl.slice("/assets/".length));
    const path = resolve(casesRoot, suffix);
    return JSON.parse(readFileSync(path, "utf8")) as {
      connectedFrames?: unknown[];
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

function majority<T extends string>(values: T[], fallback: T): T {
  if (!values.length) return fallback;
  const counts = new Map<T, number>();
  for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1);
  return (
    [...counts.entries()].sort((left, right) => right[1] - left[1])[0]?.[0] ??
    fallback
  );
}

function floorPresentation(floorHint: FloorHint): {
  floorLabel: SpatialNode["floorLabel"];
  level?: number;
} {
  if (floorHint === "basement") return { floorLabel: "Basement", level: -1 };
  if (floorHint === "ground") return { floorLabel: "Ground floor", level: 0 };
  if (floorHint === "upper") return { floorLabel: "Upper floor" };
  if (floorHint === "attic") return { floorLabel: "Attic" };
  return { floorLabel: "Unknown floor" };
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
