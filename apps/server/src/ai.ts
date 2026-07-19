import { readFileSync, statSync } from "node:fs";
import { relative, resolve } from "node:path";
import { z } from "zod";
import {
  AiCaseAnalysisInputSchema,
  type AiCaseAnalysisInput,
  type AiCaseAnalysisResult,
  type HazardCandidate,
} from "@structurefirst/contracts";
import type { AppConfig } from "./config.js";
import { confidence } from "./lib/confidence.js";
import { createId, nowIso } from "./lib/ids.js";
import { SettingsService } from "./settings.js";
import { StructureStore } from "./store.js";

const GeneratedHazardSchema = z.object({
  label: z.string().trim().min(1).max(160),
  category: z.enum([
    "collapse",
    "fire_spread",
    "utility",
    "hazmat",
    "access",
    "visibility",
    "occupancy",
    "intelligence_gap",
    "other",
  ]),
  description: z.string().trim().min(1).max(1200),
  locationLabel: z.string().trim().max(240),
  severity: z.enum(["advisory", "caution", "critical"]),
  roles: z
    .array(z.enum(["fire", "law", "ems", "sar"]))
    .min(1)
    .max(4),
  sourceIds: z.array(z.string()).min(1).max(20),
});

const GeneratedAnalysisSchema = z.object({
  summary: z.string().trim().min(1).max(2000),
  hazards: z.array(GeneratedHazardSchema).max(10),
});

type ChatResponse = {
  choices?: Array<{ message?: { content?: string | null } }>;
};

export class AiCaseAnalyzer {
  constructor(
    private readonly store: StructureStore,
    private readonly settings: SettingsService,
    private readonly config: AppConfig,
  ) {}

  async analyze(
    caseId: string,
    rawInput: AiCaseAnalysisInput,
  ): Promise<AiCaseAnalysisResult> {
    const input = AiCaseAnalysisInputSchema.parse(rawInput);
    const workspace = this.store.getWorkspace(caseId);
    if (!workspace) throw new Error("Case not found.");
    const credential = this.settings.credential(input.provider);
    if (!credential) {
      throw new Error(
        "No enabled AI provider has both an API key and model. Configure one in Settings.",
      );
    }

    const validEvidenceIds = new Set(
      workspace.evidence.map((evidence) => evidence.id),
    );
    const image =
      input.includeImage && credential.vision
        ? this.firstEligibleImage(workspace.evidence)
        : undefined;
    const casePacket = {
      address: workspace.case.displayAddress,
      role: workspace.case.activeRole,
      incident: workspace.case.incident,
      building: workspace.case.profile
        ? {
            levels: workspace.case.profile.levels,
            buildingType: workspace.case.profile.buildingType,
            construction: workspace.case.profile.construction,
            yearBuilt: workspace.case.profile.yearBuilt,
            tags: workspace.case.profile.tags,
          }
        : null,
      evidence: workspace.evidence.map((evidence) => ({
        id: evidence.id,
        title: evidence.title,
        kind: evidence.kind,
        provider: evidence.sourceProvider,
        tags: evidence.tags,
        notes: evidence.notes,
        confidence: evidence.confidence,
      })),
    };
    const system = [
      "You assist an emergency-structure intelligence analyst.",
      "Return JSON only with shape {summary:string,hazards:array}.",
      "A hazard is a review candidate, never a verified fact.",
      "Use only supplied evidence. Never invent rooms, doors, routes, occupants, utilities, hazards, or hidden geometry.",
      "Every hazard must cite one or more exact evidence IDs from the packet.",
      "If evidence does not support a physical hazard, return only evidence-gap candidates or an empty list.",
      "Use category collapse|fire_spread|utility|hazmat|access|visibility|occupancy|intelligence_gap|other, severity advisory|caution|critical, and roles fire|law|ems|sar.",
    ].join(" ");
    const text = `Analyze this case packet:\n${JSON.stringify(casePacket)}`;
    const userContent = image
      ? [
          { type: "text", text },
          {
            type: "image_url",
            image_url: { url: image.dataUrl, detail: "low" },
          },
        ]
      : text;

    const response = await fetch(`${credential.baseUrl}/chat/completions`, {
      method: "POST",
      headers: this.settings.headers(credential),
      body: JSON.stringify({
        model: credential.model,
        temperature: 0.1,
        max_tokens: 1800,
        messages: [
          { role: "system", content: system },
          { role: "user", content: userContent },
        ],
      }),
      signal: AbortSignal.timeout(60_000),
    });
    if (!response.ok) {
      const body = await response.text();
      throw new Error(
        `${providerLabel(credential.id)} returned ${response.status}: ${body.slice(0, 300)}`,
      );
    }
    const payload = (await response.json()) as ChatResponse;
    const content = payload.choices?.[0]?.message?.content;
    if (!content) throw new Error("The AI provider returned no analysis text.");
    const generated = GeneratedAnalysisSchema.parse(parseJson(content));

    let hazardsAdded = 0;
    for (const candidate of generated.hazards) {
      const sourceIds = [...new Set(candidate.sourceIds)].filter((sourceId) =>
        validEvidenceIds.has(sourceId),
      );
      if (sourceIds.length === 0) continue;
      const hazard: HazardCandidate = {
        id: createId("hazard"),
        caseId,
        label: candidate.label,
        category: candidate.category,
        description: candidate.description,
        locationLabel: candidate.locationLabel,
        severity: candidate.severity,
        roles: [...new Set(candidate.roles)],
        sourceIds,
        confidence: confidence(
          image ? 0.42 : 0.28,
          "estimated",
          "inferred",
          `AI-generated review candidate from ${providerLabel(credential.id)} (${credential.model}); operator confirmation is required.`,
          sourceIds.length,
        ),
        review: "pending",
        reviewNote: "",
      };
      this.store.putHazard(hazard);
      hazardsAdded += 1;
    }
    if (hazardsAdded > 0) {
      this.store.putCase({
        ...workspace.case,
        status: "review_required",
        updatedAt: nowIso(),
      });
    }
    return {
      provider: credential.id,
      model: credential.model,
      summary: generated.summary,
      hazardsAdded,
      usedImage: Boolean(image),
    };
  }

  private firstEligibleImage(
    evidence: ReturnType<StructureStore["listEvidence"]>,
  ): { dataUrl: string } | undefined {
    for (const item of evidence) {
      if (!item.localUrl || !item.mimeType?.startsWith("image/")) continue;
      const path = localAssetPath(
        this.config.casesRoot,
        item.caseId,
        item.localUrl,
      );
      if (!path) continue;
      try {
        const size = statSync(path).size;
        if (size > 3 * 1024 * 1024) continue;
        return {
          dataUrl: `data:${item.mimeType};base64,${readFileSync(path).toString("base64")}`,
        };
      } catch {
        // A stale evidence record must not prevent metadata-only analysis.
      }
    }
    return undefined;
  }
}

function localAssetPath(
  casesRoot: string,
  caseId: string,
  localUrl: string,
): string | undefined {
  if (!localUrl.startsWith("/assets/")) return undefined;
  const suffix = decodeURIComponent(localUrl.slice("/assets/".length));
  if (suffix.split(/[\\/]/)[0] !== caseId) return undefined;
  const candidate = resolve(casesRoot, suffix);
  const traversal = relative(casesRoot, candidate);
  return traversal.startsWith("..") || resolve(traversal) === traversal
    ? undefined
    : candidate;
}

function parseJson(text: string): unknown {
  const trimmed = text.trim();
  const fenced = trimmed.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
  const candidate = fenced?.[1] ?? trimmed;
  try {
    return JSON.parse(candidate);
  } catch {
    const start = candidate.indexOf("{");
    const end = candidate.lastIndexOf("}");
    if (start >= 0 && end > start)
      return JSON.parse(candidate.slice(start, end + 1));
    throw new Error("The AI provider did not return valid JSON.");
  }
}

function providerLabel(provider: string): string {
  return provider === "nvidia_nim"
    ? "NVIDIA NIM"
    : provider.charAt(0).toUpperCase() + provider.slice(1);
}
