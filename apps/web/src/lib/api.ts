import type {
  AiCaseAnalysisInput,
  AiCaseAnalysisResult,
  AiProviderId,
  AiProviderModelsResult,
  AiProviderSettings,
  AiProviderTestResult,
  AppSettings,
  Case,
  CaseWorkspace,
  CreateCaseInput,
  EvidenceAsset,
  HazardCandidate,
  IncidentContext,
  LoadAiProviderModelsInput,
  DiscoveryRunInput,
  DiscoveryRunResult,
  PhotoUploadResult,
  ReconstructionArtifact,
  ReconstructionRequest,
  ReviewInput,
  Role,
  SaveAiProviderInput,
  SaveDiscoverySettingsInput,
  TestAiProviderInput,
  RouteCandidate,
  SystemHealth,
} from "@structurefirst/contracts";

type Session = { required: boolean; authenticated: boolean };

export class ApiRequestError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly details?: unknown,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const isForm = init.body instanceof FormData;
  if (init.body !== undefined && !isForm)
    headers.set("content-type", "application/json");
  if (init.method && init.method !== "GET") {
    headers.set("x-structurefirst-intent", "operator-action");
  }
  const response = await fetch(path, {
    ...init,
    headers,
    credentials: "same-origin",
  });
  const body = (await response.json().catch(() => undefined)) as
    { error?: string; details?: unknown } | undefined;
  if (!response.ok) {
    throw new ApiRequestError(
      body?.error ?? `Request failed with ${response.status}.`,
      response.status,
      body?.details,
    );
  }
  return body as T;
}

export const api = {
  session: () => request<Session>("/api/auth/session"),
  login: (accessKey: string) =>
    request<Session>("/api/auth/session", {
      method: "POST",
      body: JSON.stringify({ accessKey }),
    }),
  logout: () => request<Session>("/api/auth/session", { method: "DELETE" }),
  health: () => request<SystemHealth>("/api/health"),
  settings: () => request<AppSettings>("/api/settings"),
  saveProvider: (provider: AiProviderId, input: SaveAiProviderInput) =>
    request<AiProviderSettings>(
      `/api/settings/providers/${encodeURIComponent(provider)}`,
      { method: "PUT", body: JSON.stringify(input) },
    ),
  testProvider: (provider: AiProviderId, input: TestAiProviderInput) =>
    request<AiProviderTestResult>(
      `/api/settings/providers/${encodeURIComponent(provider)}/test`,
      { method: "POST", body: JSON.stringify(input) },
    ),
  providerModels: (
    provider: AiProviderId,
    input: LoadAiProviderModelsInput = {},
  ) =>
    request<AiProviderModelsResult>(
      `/api/settings/providers/${encodeURIComponent(provider)}/models`,
      { method: "POST", body: JSON.stringify(input) },
    ),
  saveDiscovery: (input: SaveDiscoverySettingsInput) =>
    request<AppSettings["discovery"]>("/api/settings/discovery", {
      method: "PUT",
      body: JSON.stringify(input),
    }),
  cases: () => request<Case[]>("/api/cases"),
  createCase: (input: CreateCaseInput) =>
    request<Case>("/api/cases", {
      method: "POST",
      body: JSON.stringify(input),
    }),
  workspace: (id: string) =>
    request<CaseWorkspace>(`/api/cases/${encodeURIComponent(id)}`),
  deleteCase: (id: string) =>
    request<{ deleted: boolean }>(`/api/cases/${encodeURIComponent(id)}`, {
      method: "DELETE",
    }),
  retry: (id: string) =>
    request<Case>(`/api/cases/${encodeURIComponent(id)}/retry`, {
      method: "POST",
    }),
  discover: (id: string, input: DiscoveryRunInput) =>
    request<DiscoveryRunResult>(
      `/api/cases/${encodeURIComponent(id)}/discovery`,
      { method: "POST", body: JSON.stringify(input) },
    ),
  analyze: (id: string, input: AiCaseAnalysisInput) =>
    request<AiCaseAnalysisResult>(
      `/api/cases/${encodeURIComponent(id)}/ai/analyze`,
      { method: "POST", body: JSON.stringify(input) },
    ),
  setRole: (id: string, role: Role) =>
    request<Case>(`/api/cases/${encodeURIComponent(id)}/role`, {
      method: "PATCH",
      body: JSON.stringify({ role }),
    }),
  updateIncident: (
    id: string,
    incident: Partial<Omit<IncidentContext, "updatedAt">>,
  ) =>
    request<Case>(`/api/cases/${encodeURIComponent(id)}/incident`, {
      method: "PATCH",
      body: JSON.stringify(incident),
    }),
  archive: (id: string, archived: boolean) =>
    request<Case>(`/api/cases/${encodeURIComponent(id)}/archive`, {
      method: "PATCH",
      body: JSON.stringify({ archived }),
    }),
  addLink: (
    id: string,
    input: {
      url: string;
      title: string;
      kind: EvidenceAsset["kind"];
      notes: string;
    },
  ) =>
    request<EvidenceAsset>(
      `/api/cases/${encodeURIComponent(id)}/evidence/link`,
      {
        method: "POST",
        body: JSON.stringify(input),
      },
    ),
  upload: (id: string, form: FormData) =>
    request<EvidenceAsset>(
      `/api/cases/${encodeURIComponent(id)}/evidence/upload`,
      {
        method: "POST",
        body: form,
      },
    ),
  uploadPhotos: (id: string, form: FormData) =>
    request<PhotoUploadResult>(`/api/cases/${encodeURIComponent(id)}/photos`, {
      method: "POST",
      body: form,
    }),
  importEvidence: (caseId: string, evidenceId: string) =>
    request<EvidenceAsset>(
      `/api/cases/${encodeURIComponent(caseId)}/evidence/${encodeURIComponent(evidenceId)}/import`,
      { method: "POST" },
    ),
  reconstruct: (id: string, input: ReconstructionRequest) =>
    request<ReconstructionArtifact>(
      `/api/cases/${encodeURIComponent(id)}/reconstruction`,
      {
        method: "POST",
        body: JSON.stringify(input),
      },
    ),
  connectReconstruction: (id: string, evidenceIds: string[]) =>
    request<ReconstructionArtifact>(
      `/api/cases/${encodeURIComponent(id)}/reconstruction/connect`,
      { method: "POST", body: JSON.stringify({ evidenceIds }) },
    ),
  reviewHazard: (caseId: string, hazardId: string, input: ReviewInput) =>
    request<HazardCandidate>(
      `/api/cases/${encodeURIComponent(caseId)}/hazards/${encodeURIComponent(hazardId)}/review`,
      { method: "PATCH", body: JSON.stringify(input) },
    ),
  reviewRoute: (caseId: string, routeId: string, input: ReviewInput) =>
    request<RouteCandidate>(
      `/api/cases/${encodeURIComponent(caseId)}/routes/${encodeURIComponent(routeId)}/review`,
      { method: "PATCH", body: JSON.stringify(input) },
    ),
};
