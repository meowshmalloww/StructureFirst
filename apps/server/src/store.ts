import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { DatabaseSync } from "node:sqlite";
import {
  CaseSchema,
  CaseWorkspaceSchema,
  EvidenceAssetSchema,
  HazardCandidateSchema,
  PipelineEventSchema,
  ReconstructionArtifactSchema,
  RouteCandidateSchema,
  SpatialEdgeSchema,
  SpatialNodeSchema,
  type Case,
  type CaseWorkspace,
  type EvidenceAsset,
  type HazardCandidate,
  type PipelineEvent,
  type ReconstructionArtifact,
  type RouteCandidate,
  type SpatialEdge,
  type SpatialNode,
} from "@structurefirst/contracts";

type JsonEntity =
  | EvidenceAsset
  | HazardCandidate
  | RouteCandidate
  | SpatialNode
  | SpatialEdge
  | ReconstructionArtifact;

type EntityTable =
  "evidence" | "hazards" | "routes" | "nodes" | "edges" | "artifacts";

type CaseRow = {
  id: string;
  payload: string;
};

type EntityRow = {
  payload: string;
};

type SettingRow = {
  payload: string;
};

export class StructureStore {
  readonly database: DatabaseSync;

  constructor(databasePath: string) {
    if (databasePath !== ":memory:") {
      mkdirSync(dirname(databasePath), { recursive: true });
    }
    this.database = new DatabaseSync(databasePath, { timeout: 5_000 });
    this.database.exec("PRAGMA foreign_keys = ON;");
    if (databasePath !== ":memory:") {
      this.database.exec(
        "PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;",
      );
    }
    this.migrate();
    this.reconcilePreparedStatuses();
  }

  private migrate(): void {
    this.database.exec(`
      CREATE TABLE IF NOT EXISTS cases (
        id TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        payload TEXT NOT NULL
      ) STRICT;

      CREATE INDEX IF NOT EXISTS cases_updated_idx ON cases(updated_at DESC);

      CREATE TABLE IF NOT EXISTS evidence (
        id TEXT PRIMARY KEY,
        case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
        created_at TEXT NOT NULL,
        payload TEXT NOT NULL
      ) STRICT;

      CREATE TABLE IF NOT EXISTS hazards (
        id TEXT PRIMARY KEY,
        case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
        created_at TEXT NOT NULL,
        payload TEXT NOT NULL
      ) STRICT;

      CREATE TABLE IF NOT EXISTS routes (
        id TEXT PRIMARY KEY,
        case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
        created_at TEXT NOT NULL,
        payload TEXT NOT NULL
      ) STRICT;

      CREATE TABLE IF NOT EXISTS nodes (
        id TEXT PRIMARY KEY,
        case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
        created_at TEXT NOT NULL,
        payload TEXT NOT NULL
      ) STRICT;

      CREATE TABLE IF NOT EXISTS edges (
        id TEXT PRIMARY KEY,
        case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
        created_at TEXT NOT NULL,
        payload TEXT NOT NULL
      ) STRICT;

      CREATE TABLE IF NOT EXISTS artifacts (
        id TEXT PRIMARY KEY,
        case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
        created_at TEXT NOT NULL,
        payload TEXT NOT NULL
      ) STRICT;

      CREATE TABLE IF NOT EXISTS pipeline_events (
        id TEXT PRIMARY KEY,
        case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
        created_at TEXT NOT NULL,
        payload TEXT NOT NULL
      ) STRICT;

      CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        updated_at TEXT NOT NULL,
        payload TEXT NOT NULL
      ) STRICT;

      CREATE INDEX IF NOT EXISTS evidence_case_idx ON evidence(case_id, created_at DESC);
      CREATE INDEX IF NOT EXISTS hazards_case_idx ON hazards(case_id, created_at DESC);
      CREATE INDEX IF NOT EXISTS routes_case_idx ON routes(case_id, created_at DESC);
      CREATE INDEX IF NOT EXISTS nodes_case_idx ON nodes(case_id, created_at ASC);
      CREATE INDEX IF NOT EXISTS edges_case_idx ON edges(case_id, created_at ASC);
      CREATE INDEX IF NOT EXISTS artifacts_case_idx ON artifacts(case_id, created_at DESC);
      CREATE INDEX IF NOT EXISTS events_case_idx ON pipeline_events(case_id, created_at ASC);
    `);
  }

  private reconcilePreparedStatuses(): void {
    for (const current of this.listCases()) {
      if (current.status !== "review_required") continue;
      const hasReadyScene = this.listArtifacts(current.id).some(
        (artifact) => artifact.status === "ready",
      );
      if (!hasReadyScene) continue;
      const findings = this.listHazards(current.id);
      const onlyAutomaticGap = findings.every(
        (finding) =>
          finding.category === "intelligence_gap" &&
          finding.label === "Interior layout unknown",
      );
      if (onlyAutomaticGap)
        this.putCase({ ...current, status: "briefing_ready" });
    }
  }

  close(): void {
    if (this.database.isOpen) {
      this.database.close();
    }
  }

  putCase(value: Case): Case {
    const parsed = CaseSchema.parse(value);
    this.database
      .prepare(
        `INSERT INTO cases (id, status, updated_at, payload)
         VALUES (?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET
           status = excluded.status,
           updated_at = excluded.updated_at,
           payload = excluded.payload`,
      )
      .run(parsed.id, parsed.status, parsed.updatedAt, JSON.stringify(parsed));
    return parsed;
  }

  getCase(id: string): Case | undefined {
    const row = this.database
      .prepare("SELECT id, payload FROM cases WHERE id = ?")
      .get(id) as CaseRow | undefined;
    return row ? CaseSchema.parse(JSON.parse(row.payload)) : undefined;
  }

  listCases(): Case[] {
    const rows = this.database
      .prepare("SELECT id, payload FROM cases ORDER BY updated_at DESC")
      .all() as unknown as CaseRow[];
    return rows.map((row) => CaseSchema.parse(JSON.parse(row.payload)));
  }

  deleteCase(id: string): boolean {
    return (
      Number(
        this.database.prepare("DELETE FROM cases WHERE id = ?").run(id).changes,
      ) > 0
    );
  }

  putEvidence(value: EvidenceAsset): EvidenceAsset {
    return this.putEntity("evidence", EvidenceAssetSchema.parse(value));
  }

  putHazard(value: HazardCandidate): HazardCandidate {
    return this.putEntity("hazards", HazardCandidateSchema.parse(value));
  }

  putRoute(value: RouteCandidate): RouteCandidate {
    return this.putEntity("routes", RouteCandidateSchema.parse(value));
  }

  putNode(value: SpatialNode): SpatialNode {
    return this.putEntity("nodes", SpatialNodeSchema.parse(value));
  }

  putEdge(value: SpatialEdge): SpatialEdge {
    return this.putEntity("edges", SpatialEdgeSchema.parse(value));
  }

  putArtifact(value: ReconstructionArtifact): ReconstructionArtifact {
    return this.putEntity(
      "artifacts",
      ReconstructionArtifactSchema.parse(value),
    );
  }

  getEvidence(id: string): EvidenceAsset | undefined {
    return this.getEntity("evidence", id, EvidenceAssetSchema);
  }

  getHazard(id: string): HazardCandidate | undefined {
    return this.getEntity("hazards", id, HazardCandidateSchema);
  }

  getRoute(id: string): RouteCandidate | undefined {
    return this.getEntity("routes", id, RouteCandidateSchema);
  }

  getArtifact(id: string): ReconstructionArtifact | undefined {
    return this.getEntity("artifacts", id, ReconstructionArtifactSchema);
  }

  listEvidence(caseId: string): EvidenceAsset[] {
    return this.listEntities("evidence", caseId, EvidenceAssetSchema);
  }

  listHazards(caseId: string): HazardCandidate[] {
    return this.listEntities("hazards", caseId, HazardCandidateSchema);
  }

  listRoutes(caseId: string): RouteCandidate[] {
    return this.listEntities("routes", caseId, RouteCandidateSchema);
  }

  listNodes(caseId: string): SpatialNode[] {
    return this.listEntities("nodes", caseId, SpatialNodeSchema, "ASC");
  }

  listEdges(caseId: string): SpatialEdge[] {
    return this.listEntities("edges", caseId, SpatialEdgeSchema, "ASC");
  }

  deleteSpatialGraph(caseId: string): void {
    this.database.prepare("DELETE FROM edges WHERE case_id = ?").run(caseId);
    this.database.prepare("DELETE FROM nodes WHERE case_id = ?").run(caseId);
  }

  listArtifacts(caseId: string): ReconstructionArtifact[] {
    return this.listEntities("artifacts", caseId, ReconstructionArtifactSchema);
  }

  putEvent(value: PipelineEvent): PipelineEvent {
    const parsed = PipelineEventSchema.parse(value);
    this.database
      .prepare(
        "INSERT INTO pipeline_events (id, case_id, created_at, payload) VALUES (?, ?, ?, ?)",
      )
      .run(parsed.id, parsed.caseId, parsed.createdAt, JSON.stringify(parsed));
    return parsed;
  }

  listEvents(caseId: string): PipelineEvent[] {
    const rows = this.database
      .prepare(
        "SELECT payload FROM pipeline_events WHERE case_id = ? ORDER BY created_at ASC",
      )
      .all(caseId) as unknown as EntityRow[];
    return rows.map((row) =>
      PipelineEventSchema.parse(JSON.parse(row.payload)),
    );
  }

  putSetting<T>(key: string, value: T): T {
    this.database
      .prepare(
        `INSERT INTO settings (key, updated_at, payload)
         VALUES (?, ?, ?)
         ON CONFLICT(key) DO UPDATE SET
           updated_at = excluded.updated_at,
           payload = excluded.payload`,
      )
      .run(key, new Date().toISOString(), JSON.stringify(value));
    return value;
  }

  getSetting<T>(key: string): T | undefined {
    const row = this.database
      .prepare("SELECT payload FROM settings WHERE key = ?")
      .get(key) as SettingRow | undefined;
    return row ? (JSON.parse(row.payload) as T) : undefined;
  }

  getWorkspace(caseId: string): CaseWorkspace | undefined {
    const caseValue = this.getCase(caseId);
    if (!caseValue) return undefined;
    return CaseWorkspaceSchema.parse({
      case: caseValue,
      evidence: this.listEvidence(caseId),
      hazards: this.listHazards(caseId),
      routes: this.listRoutes(caseId),
      nodes: this.listNodes(caseId),
      edges: this.listEdges(caseId),
      artifacts: this.listArtifacts(caseId),
    });
  }

  private putEntity<T extends JsonEntity>(table: EntityTable, value: T): T {
    const createdAt =
      "createdAt" in value ? String(value.createdAt) : new Date().toISOString();
    this.database
      .prepare(
        `INSERT INTO ${table} (id, case_id, created_at, payload)
         VALUES (?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET payload = excluded.payload`,
      )
      .run(value.id, value.caseId, createdAt, JSON.stringify(value));
    return value;
  }

  private getEntity<T>(
    table: EntityTable,
    id: string,
    schema: { parse(input: unknown): T },
  ): T | undefined {
    const row = this.database
      .prepare(`SELECT payload FROM ${table} WHERE id = ?`)
      .get(id) as EntityRow | undefined;
    return row ? schema.parse(JSON.parse(row.payload)) : undefined;
  }

  private listEntities<T>(
    table: EntityTable,
    caseId: string,
    schema: { parse(input: unknown): T },
    order: "ASC" | "DESC" = "DESC",
  ): T[] {
    const rows = this.database
      .prepare(
        `SELECT payload FROM ${table} WHERE case_id = ? ORDER BY created_at ${order}`,
      )
      .all(caseId) as unknown as EntityRow[];
    return rows.map((row) => schema.parse(JSON.parse(row.payload)));
  }
}
