import { z } from "zod";

export const IdSchema = z.string().min(8).max(128);
export const IsoDateSchema = z.string().datetime({ offset: true });

export const RoleSchema = z.enum(["fire", "law", "ems", "sar"]);
export type Role = z.infer<typeof RoleSchema>;

export const CaseStatusSchema = z.enum([
  "collecting",
  "reconstructing",
  "review_required",
  "briefing_ready",
  "limited_evidence",
  "failed",
  "archived",
]);
export type CaseStatus = z.infer<typeof CaseStatusSchema>;

export const ConfidenceBandSchema = z.enum([
  "verified",
  "reconstructed",
  "estimated",
  "unknown",
]);
export type ConfidenceBand = z.infer<typeof ConfidenceBandSchema>;

export const EpistemicStateSchema = z.enum([
  "observed",
  "derived",
  "inferred",
  "unknown",
]);
export type EpistemicState = z.infer<typeof EpistemicStateSchema>;

export const ConfidenceSchema = z.object({
  score: z.number().min(0).max(1),
  band: ConfidenceBandSchema,
  state: EpistemicStateSchema,
  rationale: z.string().min(1),
  sourceCount: z.number().int().min(0),
  updatedAt: IsoDateSchema,
});
export type Confidence = z.infer<typeof ConfidenceSchema>;

export const GeoPointSchema = z.object({
  latitude: z.number().min(-90).max(90),
  longitude: z.number().min(-180).max(180),
});
export type GeoPoint = z.infer<typeof GeoPointSchema>;

export const PolygonSchema = z.object({
  type: z.literal("Polygon"),
  coordinates: z.array(z.array(z.tuple([z.number(), z.number()]))).min(1),
});
export type Polygon = z.infer<typeof PolygonSchema>;

export const SourceReferenceSchema = z.object({
  id: IdSchema,
  label: z.string().min(1),
  provider: z.string().min(1),
  url: z.url().optional(),
  retrievedAt: IsoDateSchema,
  observedAt: IsoDateSchema.optional(),
  license: z.string().optional(),
});
export type SourceReference = z.infer<typeof SourceReferenceSchema>;

export const BuildingProfileSchema = z.object({
  displayAddress: z.string().min(1),
  location: GeoPointSchema,
  footprint: PolygonSchema.optional(),
  levels: z.number().int().positive().optional(),
  buildingType: z.string().optional(),
  construction: z.string().optional(),
  yearBuilt: z.number().int().min(1600).max(2300).optional(),
  tags: z.record(z.string(), z.string()),
  source: SourceReferenceSchema,
  confidence: ConfidenceSchema,
});
export type BuildingProfile = z.infer<typeof BuildingProfileSchema>;

export const IncidentContextSchema = z.object({
  type: z.enum([
    "structure_fire",
    "law_enforcement",
    "medical",
    "search_rescue",
    "hazmat",
    "other",
  ]),
  reportedOrigin: z.string().max(240).optional(),
  preferredEntry: z.string().max(240).optional(),
  victimReports: z.string().max(500).optional(),
  blockedAreas: z.array(z.string().max(240)).max(50),
  notes: z.string().max(4000).optional(),
  updatedAt: IsoDateSchema,
});
export type IncidentContext = z.infer<typeof IncidentContextSchema>;

export const PipelineStageNameSchema = z.enum([
  "address_resolution",
  "building_records",
  "evidence_discovery",
  "evidence_validation",
  "reconstruction",
  "spatial_reasoning",
  "candidate_generation",
]);
export type PipelineStageName = z.infer<typeof PipelineStageNameSchema>;

export const StageStatusSchema = z.enum([
  "pending",
  "running",
  "complete",
  "limited",
  "skipped",
  "failed",
]);
export type StageStatus = z.infer<typeof StageStatusSchema>;

export const PipelineStageSchema = z.object({
  name: PipelineStageNameSchema,
  status: StageStatusSchema,
  message: z.string(),
  startedAt: IsoDateSchema.optional(),
  completedAt: IsoDateSchema.optional(),
  itemCount: z.number().int().min(0).optional(),
});
export type PipelineStage = z.infer<typeof PipelineStageSchema>;

export const EvidenceCoverageSchema = z.object({
  exterior: ConfidenceSchema,
  interior: ConfidenceSchema,
  records: ConfidenceSchema,
  lastAssessedAt: IsoDateSchema,
});
export type EvidenceCoverage = z.infer<typeof EvidenceCoverageSchema>;

export const CaseSchema = z.object({
  id: IdSchema,
  addressInput: z.string().min(1),
  displayAddress: z.string().min(1),
  status: CaseStatusSchema,
  activeRole: RoleSchema,
  createdAt: IsoDateSchema,
  updatedAt: IsoDateSchema,
  profile: BuildingProfileSchema.optional(),
  incident: IncidentContextSchema,
  coverage: EvidenceCoverageSchema,
  stages: z.array(PipelineStageSchema),
  operatorNotes: z.string(),
});
export type Case = z.infer<typeof CaseSchema>;

export const RightsStatusSchema = z.enum([
  "operator_owned",
  "open_license",
  "public_domain",
  "link_only",
  "research_unknown",
  "restricted",
]);
export type RightsStatus = z.infer<typeof RightsStatusSchema>;

export const CachePolicySchema = z.enum([
  "local_allowed",
  "metadata_only",
  "prohibited",
]);
export type CachePolicy = z.infer<typeof CachePolicySchema>;

export const EvidenceKindSchema = z.enum([
  "image",
  "video",
  "document",
  "map",
  "record",
  "web_page",
  "blueprint",
]);
export type EvidenceKind = z.infer<typeof EvidenceKindSchema>;

export const EvidenceValidationSchema = z.enum([
  "pending",
  "reachable",
  "operator_uploaded",
  "automated_imported",
  "unavailable",
  "rejected",
]);

export const EvidenceSceneTypeSchema = z.enum([
  "exterior",
  "interior",
  "floor_plan",
  "non_property",
  "unknown",
]);
export type EvidenceSceneType = z.infer<typeof EvidenceSceneTypeSchema>;

export const RoomTypeSchema = z.enum([
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
]);
export type RoomType = z.infer<typeof RoomTypeSchema>;

export const FloorHintSchema = z.enum([
  "basement",
  "ground",
  "upper",
  "attic",
  "unknown",
]);
export type FloorHint = z.infer<typeof FloorHintSchema>;

export const EvidenceVisualAnalysisSchema = z.object({
  sceneType: EvidenceSceneTypeSchema,
  roomType: RoomTypeSchema,
  floorHint: FloorHintSchema,
  propertyRelevance: z.enum(["likely", "unlikely", "unknown"]),
  addressMatch: z.enum(["supported", "possible", "contradictory", "unknown"]),
  observedAddress: z.string().max(300).optional(),
  connections: z
    .array(z.enum(["door", "corridor", "stair_up", "stair_down", "window"]))
    .max(12),
  summary: z.string().min(1).max(800),
  provider: z.enum(["groq", "cerebras", "openrouter", "nvidia_nim"]),
  model: z.string().min(1).max(300),
  confidenceScore: z.number().min(0).max(1),
  analyzedAt: IsoDateSchema,
});
export type EvidenceVisualAnalysis = z.infer<
  typeof EvidenceVisualAnalysisSchema
>;

export const EvidenceAssetSchema = z.object({
  id: IdSchema,
  caseId: IdSchema,
  title: z.string().min(1),
  kind: EvidenceKindSchema,
  sourceProvider: z.string().min(1),
  originUrl: z.url().optional(),
  downloadUrl: z.url().optional(),
  licenseUrl: z.url().optional(),
  creator: z.string().max(500).optional(),
  localUrl: z.string().optional(),
  thumbnailUrl: z.string().optional(),
  discoveredAt: IsoDateSchema,
  observedAt: IsoDateSchema.optional(),
  rights: RightsStatusSchema,
  cachePolicy: CachePolicySchema,
  redistributable: z.boolean(),
  validation: EvidenceValidationSchema,
  mimeType: z.string().optional(),
  byteSize: z.number().int().nonnegative().optional(),
  sha256: z.string().optional(),
  tags: z.array(z.string()),
  notes: z.string(),
  confidence: ConfidenceSchema,
  visualAnalysis: EvidenceVisualAnalysisSchema.optional(),
});
export type EvidenceAsset = z.infer<typeof EvidenceAssetSchema>;

export const AiProviderIdSchema = z.enum([
  "groq",
  "cerebras",
  "openrouter",
  "nvidia_nim",
]);
export type AiProviderId = z.infer<typeof AiProviderIdSchema>;

export const AiProviderSettingsSchema = z.object({
  id: AiProviderIdSchema,
  label: z.string().min(1),
  baseUrl: z.url(),
  model: z.string().max(300),
  enabled: z.boolean(),
  vision: z.boolean(),
  configured: z.boolean(),
  keyHint: z.string().max(32).optional(),
  updatedAt: IsoDateSchema.optional(),
});
export type AiProviderSettings = z.infer<typeof AiProviderSettingsSchema>;

export const SaveAiProviderInputSchema = z.object({
  baseUrl: z.url(),
  model: z.string().trim().max(300),
  enabled: z.boolean(),
  vision: z.boolean(),
  apiKey: z.string().trim().max(4000).optional(),
  clearKey: z.boolean().default(false),
});
export type SaveAiProviderInput = z.infer<typeof SaveAiProviderInputSchema>;

export const LoadAiProviderModelsInputSchema = z.object({
  apiKey: z.string().trim().max(4000).optional(),
});
export type LoadAiProviderModelsInput = z.infer<
  typeof LoadAiProviderModelsInputSchema
>;

export const AiProviderModelSchema = z.object({
  id: z.string().trim().min(1).max(300),
  name: z.string().trim().min(1).max(300),
  ownedBy: z.string().trim().min(1).max(160).optional(),
  contextWindow: z.number().int().positive().optional(),
  free: z.boolean().optional(),
  vision: z.boolean(),
});
export type AiProviderModel = z.infer<typeof AiProviderModelSchema>;

export const AiProviderCatalogKindSchema = z.enum([
  "free_plan",
  "free_models",
  "prototype",
]);
export type AiProviderCatalogKind = z.infer<typeof AiProviderCatalogKindSchema>;

export const AiProviderModelsResultSchema = z.object({
  provider: AiProviderIdSchema,
  models: z.array(AiProviderModelSchema),
  catalogKind: AiProviderCatalogKindSchema,
  notice: z.string().trim().min(1).max(500),
  latencyMs: z.number().int().nonnegative(),
});
export type AiProviderModelsResult = z.infer<
  typeof AiProviderModelsResultSchema
>;

export const TestAiProviderInputSchema = z.object({
  model: z.string().trim().min(1).max(300),
  apiKey: z.string().trim().max(4000).optional(),
});
export type TestAiProviderInput = z.infer<typeof TestAiProviderInputSchema>;

export const AiProviderTestResultSchema = z.object({
  ok: z.boolean(),
  provider: AiProviderIdSchema,
  model: z.string().trim().min(1).max(300),
  latencyMs: z.number().int().nonnegative(),
  message: z.string(),
});
export type AiProviderTestResult = z.infer<typeof AiProviderTestResultSchema>;

export const DiscoverySettingsSchema = z.object({
  openverseEnabled: z.boolean(),
  browserEnabled: z.boolean(),
  browserAgentEnabled: z.boolean(),
  browserAgentMaxSteps: z.number().int().min(4).max(60),
  browserExecutablePath: z.string().max(1000).optional(),
  braveConfigured: z.boolean(),
  braveKeyHint: z.string().max(32).optional(),
  updatedAt: IsoDateSchema.optional(),
});
export type DiscoverySettings = z.infer<typeof DiscoverySettingsSchema>;

export const SaveDiscoverySettingsInputSchema = z.object({
  openverseEnabled: z.boolean(),
  browserEnabled: z.boolean(),
  browserAgentEnabled: z.boolean().default(false),
  browserAgentMaxSteps: z.number().int().min(4).max(60).default(20),
  browserExecutablePath: z.string().trim().max(1000).optional(),
  clearBrowserExecutablePath: z.boolean().default(false),
  braveApiKey: z.string().trim().max(4000).optional(),
  clearBraveKey: z.boolean().default(false),
});
export type SaveDiscoverySettingsInput = z.infer<
  typeof SaveDiscoverySettingsInputSchema
>;

export const AppSettingsSchema = z.object({
  providers: z.array(AiProviderSettingsSchema),
  discovery: DiscoverySettingsSchema,
});
export type AppSettings = z.infer<typeof AppSettingsSchema>;

export const ReviewStatusSchema = z.enum(["pending", "confirmed", "rejected"]);
export type ReviewStatus = z.infer<typeof ReviewStatusSchema>;

export const SpatialNodeKindSchema = z.enum([
  "exterior",
  "entry",
  "room",
  "stair",
  "corridor",
  "window",
  "utility",
  "target",
]);

export const SpatialNodeSchema = z.object({
  id: IdSchema,
  caseId: IdSchema,
  label: z.string().min(1),
  kind: SpatialNodeKindSchema,
  level: z.number().int().optional(),
  floorLabel: z
    .enum(["Basement", "Ground floor", "Upper floor", "Attic", "Unknown floor"])
    .optional(),
  position: z.tuple([z.number(), z.number(), z.number()]).optional(),
  sourceIds: z.array(IdSchema),
  confidence: ConfidenceSchema,
});
export type SpatialNode = z.infer<typeof SpatialNodeSchema>;

export const SpatialEdgeSchema = z.object({
  id: IdSchema,
  caseId: IdSchema,
  from: IdSchema,
  to: IdSchema,
  traversal: z.enum([
    "walk",
    "door",
    "stair_up",
    "stair_down",
    "window",
    "unknown",
  ]),
  distanceMeters: z.number().positive(),
  blocked: z.boolean(),
  sourceIds: z.array(IdSchema),
  confidence: ConfidenceSchema,
});
export type SpatialEdge = z.infer<typeof SpatialEdgeSchema>;

export const HazardCategorySchema = z.enum([
  "collapse",
  "fire_spread",
  "utility",
  "hazmat",
  "access",
  "visibility",
  "occupancy",
  "intelligence_gap",
  "other",
]);

export const HazardCandidateSchema = z.object({
  id: IdSchema,
  caseId: IdSchema,
  label: z.string().min(1),
  category: HazardCategorySchema,
  description: z.string().min(1),
  locationLabel: z.string(),
  severity: z.enum(["advisory", "caution", "critical"]),
  roles: z.array(RoleSchema),
  sourceIds: z.array(IdSchema),
  confidence: ConfidenceSchema,
  review: ReviewStatusSchema,
  reviewNote: z.string(),
  reviewedAt: IsoDateSchema.optional(),
});
export type HazardCandidate = z.infer<typeof HazardCandidateSchema>;

export const RouteCandidateSchema = z.object({
  id: IdSchema,
  caseId: IdSchema,
  label: z.string().min(1),
  role: RoleSchema,
  nodeIds: z.array(IdSchema).min(2),
  distanceMeters: z.number().positive(),
  rationale: z.array(z.string()),
  warningIds: z.array(IdSchema),
  sourceIds: z.array(IdSchema),
  confidence: ConfidenceSchema,
  review: ReviewStatusSchema,
  reviewNote: z.string(),
  reviewedAt: IsoDateSchema.optional(),
});
export type RouteCandidate = z.infer<typeof RouteCandidateSchema>;

export const ReconstructionArtifactSchema = z.object({
  id: IdSchema,
  caseId: IdSchema,
  evidenceId: IdSchema,
  evidenceIds: z.array(IdSchema).min(1).optional(),
  status: z.enum(["queued", "running", "ready", "failed"]),
  mode: z.enum(["single_image", "panorama", "multi_image"]),
  splatUrl: z.string().optional(),
  manifestUrl: z.string().optional(),
  gaussianCount: z.number().int().positive().optional(),
  modelName: z.string(),
  modelLicense: z.string(),
  error: z.string().optional(),
  registration: z
    .object({
      status: z.enum(["connected", "partial", "failed"]),
      method: z.enum(["sift_metric_similarity", "sift_loftr_sharp_pose_graph"]),
      frameCount: z.number().int().positive(),
      connectedFrameCount: z.number().int().positive(),
      confidenceScore: z.number().min(0).max(1),
      reportUrl: z.string().optional(),
      note: z.string(),
    })
    .optional(),
  fallback: z
    .object({
      mode: z.literal("single_image"),
      sourceEvidenceId: IdSchema,
      reason: z.string().min(1),
    })
    .optional(),
  createdAt: IsoDateSchema,
  updatedAt: IsoDateSchema,
  confidence: ConfidenceSchema,
});
export type ReconstructionArtifact = z.infer<
  typeof ReconstructionArtifactSchema
>;

export const CaseWorkspaceSchema = z.object({
  case: CaseSchema,
  evidence: z.array(EvidenceAssetSchema),
  hazards: z.array(HazardCandidateSchema),
  routes: z.array(RouteCandidateSchema),
  nodes: z.array(SpatialNodeSchema),
  edges: z.array(SpatialEdgeSchema),
  artifacts: z.array(ReconstructionArtifactSchema),
});
export type CaseWorkspace = z.infer<typeof CaseWorkspaceSchema>;

export const CreateCaseInputSchema = z.object({
  address: z.string().trim().min(5).max(500),
  role: RoleSchema.default("fire"),
  incidentType: IncidentContextSchema.shape.type.default("other"),
});
export type CreateCaseInput = z.infer<typeof CreateCaseInputSchema>;

export const UpdateIncidentInputSchema = IncidentContextSchema.omit({
  updatedAt: true,
}).partial();
export type UpdateIncidentInput = z.infer<typeof UpdateIncidentInputSchema>;

export const AddEvidenceLinkInputSchema = z.object({
  url: z.url(),
  title: z.string().trim().min(1).max(500),
  kind: EvidenceKindSchema,
  notes: z.string().max(2000).default(""),
});
export type AddEvidenceLinkInput = z.infer<typeof AddEvidenceLinkInputSchema>;

export const ReviewInputSchema = z.object({
  review: z.enum(["confirmed", "rejected"]),
  note: z.string().trim().max(1000).default(""),
});
export type ReviewInput = z.infer<typeof ReviewInputSchema>;

export const ReconstructionRequestSchema = z.object({
  evidenceId: IdSchema,
  mode: z.enum(["single_image", "panorama"]).default("single_image"),
});
export type ReconstructionRequest = z.infer<typeof ReconstructionRequestSchema>;

export const MultiReconstructionRequestSchema = z.object({
  evidenceIds: z.array(IdSchema).min(2).max(12),
});
export type MultiReconstructionRequest = z.infer<
  typeof MultiReconstructionRequestSchema
>;

export const PhotoUploadResultSchema = z.object({
  assets: z.array(EvidenceAssetSchema).min(1),
  artifact: ReconstructionArtifactSchema.optional(),
  note: z.string(),
});
export type PhotoUploadResult = z.infer<typeof PhotoUploadResultSchema>;

export const DiscoveryRunInputSchema = z.object({
  includeOpenverse: z.boolean().default(true),
  includeBrowser: z.boolean().default(true),
  includeBrave: z.boolean().default(true),
  includeBrowserAgent: z.boolean().default(false),
});
export type DiscoveryRunInput = z.infer<typeof DiscoveryRunInputSchema>;

export const DiscoveryRunResultSchema = z.object({
  added: z.number().int().nonnegative(),
  imported: z.number().int().nonnegative(),
  providers: z.array(z.string()),
  warnings: z.array(z.string()),
});
export type DiscoveryRunResult = z.infer<typeof DiscoveryRunResultSchema>;

export const AiCaseAnalysisInputSchema = z.object({
  provider: AiProviderIdSchema.optional(),
  includeImage: z.boolean().default(true),
});
export type AiCaseAnalysisInput = z.infer<typeof AiCaseAnalysisInputSchema>;

export const AiCaseAnalysisResultSchema = z.object({
  provider: AiProviderIdSchema,
  model: z.string(),
  summary: z.string(),
  hazardsAdded: z.number().int().nonnegative(),
  usedImage: z.boolean(),
});
export type AiCaseAnalysisResult = z.infer<typeof AiCaseAnalysisResultSchema>;

export const PipelineEventSchema = z.object({
  id: IdSchema,
  caseId: IdSchema,
  stage: PipelineStageNameSchema,
  status: StageStatusSchema,
  message: z.string(),
  createdAt: IsoDateSchema,
});
export type PipelineEvent = z.infer<typeof PipelineEventSchema>;

export const SystemHealthSchema = z.object({
  service: z.literal("structurefirst"),
  status: z.enum(["ready", "degraded"]),
  version: z.string(),
  database: z.enum(["ready", "error"]),
  braveSearch: z.boolean(),
  browserResearch: z.boolean(),
  reconstruction: z.object({
    configured: z.boolean(),
    reachable: z.boolean(),
    ready: z.boolean().optional(),
    lucidFrameRoot: z.string().optional(),
    gpuExpected: z.boolean(),
    gpuAvailable: z.boolean().optional(),
    lucidFrameAvailable: z.boolean().optional(),
    modelVerified: z.boolean().optional(),
    error: z.string().optional(),
  }),
  sourcePolicy: z.literal("research_hard_blocklist"),
});
export type SystemHealth = z.infer<typeof SystemHealthSchema>;

export const ApiErrorSchema = z.object({
  error: z.string(),
  details: z.unknown().optional(),
});
export type ApiError = z.infer<typeof ApiErrorSchema>;

export const PIPELINE_STAGE_ORDER: PipelineStageName[] = [
  "address_resolution",
  "building_records",
  "evidence_discovery",
  "evidence_validation",
  "reconstruction",
  "spatial_reasoning",
  "candidate_generation",
];

export const ROLE_LABELS: Record<Role, string> = {
  fire: "Fire",
  law: "Law enforcement",
  ems: "EMS",
  sar: "Search & rescue",
};

export function createUnknownConfidence(
  rationale: string,
  updatedAt = new Date().toISOString(),
): Confidence {
  return {
    score: 0,
    band: "unknown",
    state: "unknown",
    rationale,
    sourceCount: 0,
    updatedAt,
  };
}
