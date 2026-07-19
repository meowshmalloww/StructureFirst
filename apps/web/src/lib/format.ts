import type {
  CaseStatus,
  ConfidenceBand,
  RightsStatus,
  StageStatus,
} from "@structurefirst/contracts";

export function formatRelative(iso: string): string {
  const elapsed = Date.now() - new Date(iso).getTime();
  const minutes = Math.round(elapsed / 60_000);
  if (Math.abs(minutes) < 1) return "just now";
  if (Math.abs(minutes) < 60)
    return `${Math.abs(minutes)} min ${minutes < 0 ? "from now" : "ago"}`;
  const hours = Math.round(minutes / 60);
  if (Math.abs(hours) < 24)
    return `${Math.abs(hours)} hr ${hours < 0 ? "from now" : "ago"}`;
  const days = Math.round(hours / 24);
  return `${Math.abs(days)} d ${days < 0 ? "from now" : "ago"}`;
}

export function sentence(value: string): string {
  return value
    .replaceAll("_", " ")
    .replace(/^./, (letter) => letter.toUpperCase());
}

export const CASE_STATUS_LABELS: Record<CaseStatus, string> = {
  collecting: "Collecting",
  reconstructing: "Reconstructing",
  review_required: "AI findings",
  briefing_ready: "Ready",
  limited_evidence: "Limited evidence",
  failed: "Attention needed",
  archived: "Archived",
};

export const STAGE_STATUS_LABELS: Record<StageStatus, string> = {
  pending: "Pending",
  running: "In progress",
  complete: "Complete",
  limited: "Limited",
  skipped: "Not configured",
  failed: "Failed",
};

export const CONFIDENCE_LABELS: Record<ConfidenceBand, string> = {
  verified: "Verified",
  reconstructed: "Reconstructed",
  estimated: "Estimated",
  unknown: "Unknown",
};

export const RIGHTS_LABELS: Record<RightsStatus, string> = {
  operator_owned: "Operator supplied",
  open_license: "Open license",
  public_domain: "Public domain",
  link_only: "Link only",
  research_unknown: "Rights unknown",
  restricted: "Restricted source",
};
