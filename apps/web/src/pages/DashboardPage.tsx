import {
  ArrowRight,
  Box,
  Building2,
  CircleAlert,
  Clock3,
  Image as ImageIcon,
  LoaderCircle,
  MapPin,
  Search,
  Trash2,
} from "lucide-react";
import { useEffect, useMemo, useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import type { Case, CreateCaseInput } from "@structurefirst/contracts";
import { StructureMap } from "../components/StructureMap";
import { api } from "../lib/api";
import { formatRelative } from "../lib/format";

export function DashboardPage() {
  const navigate = useNavigate();
  const [cases, setCases] = useState<Case[]>([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [deleting, setDeleting] = useState<string>();
  const [error, setError] = useState<string>();

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const next = await api.cases();
        if (!cancelled)
          setCases(next.filter((item) => item.status !== "archived"));
      } catch (caught) {
        if (!cancelled)
          setError(
            caught instanceof Error
              ? caught.message
              : "Properties unavailable.",
          );
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    void load();
    const timer = window.setInterval(() => void load(), 20_000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  const summary = useMemo(
    () => ({
      active: cases.filter((item) =>
        ["collecting", "reconstructing"].includes(item.status),
      ).length,
      ready: cases.filter((item) => item.status === "briefing_ready").length,
      mapped: cases.filter((item) => item.profile).length,
    }),
    [cases],
  );

  async function createProperty(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setError(undefined);
    const address = String(
      new FormData(event.currentTarget).get("address") ?? "",
    );
    const input: CreateCaseInput = {
      address,
      role: "fire",
      incidentType: "other",
    };
    try {
      const created = await api.createCase(input);
      navigate(`/cases/${created.id}`);
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "The property could not be added.",
      );
      setSubmitting(false);
    }
  }

  async function deleteProperty(item: Case) {
    if (!window.confirm(`Delete ${item.displayAddress} and its saved photos?`))
      return;
    setDeleting(item.id);
    setError(undefined);
    try {
      await api.deleteCase(item.id);
      setCases((current) => current.filter((entry) => entry.id !== item.id));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Delete failed.");
    } finally {
      setDeleting(undefined);
    }
  }

  return (
    <div className="page-frame operations-page">
      <header className="page-heading operations-heading">
        <div>
          <span className="eyebrow">Pre-arrival operations</span>
          <h1>Property intelligence</h1>
          <p>
            Prepare a mapped, sourced building view before the team arrives.
          </p>
        </div>
        <dl className="operations-summary" aria-label="Property summary">
          <div>
            <dt>Properties</dt>
            <dd>{cases.length}</dd>
          </div>
          <div>
            <dt>In progress</dt>
            <dd>{summary.active}</dd>
          </div>
          <div>
            <dt>Prepared</dt>
            <dd>{summary.ready}</dd>
          </div>
        </dl>
      </header>

      <section className="intake-panel" aria-labelledby="new-property-title">
        <div className="intake-copy">
          <span className="intake-icon" aria-hidden="true">
            <Search size={18} />
          </span>
          <div>
            <h2 id="new-property-title">Prepare a property</h2>
            <p>Enter an address to begin mapping and image collection.</p>
          </div>
        </div>
        <form
          className="address-command"
          onSubmit={(event) => void createProperty(event)}
        >
          <label>
            <MapPin size={18} aria-hidden="true" />
            <span className="sr-only">Property address</span>
            <input
              name="address"
              autoComplete="street-address"
              minLength={5}
              maxLength={500}
              required
              autoFocus
              placeholder="Street address, city, state"
            />
          </label>
          <button className="primary-button" disabled={submitting}>
            {submitting ? (
              <LoaderCircle className="spin" size={17} />
            ) : (
              <Search size={17} />
            )}
            {submitting ? "Starting" : "Prepare"}
          </button>
        </form>
        <small className="intake-note">
          Public imagery works without an AI key. Analysis providers are
          optional.
        </small>
        {error ? (
          <p className="inline-error" role="alert">
            <CircleAlert size={15} /> {error}
          </p>
        ) : null}
      </section>

      <section className="operations-workspace" aria-label="Saved properties">
        <article className="operations-panel map-panel">
          <header className="panel-heading">
            <div>
              <span className="panel-kicker">Common view</span>
              <h2>Property map</h2>
            </div>
            <span className="panel-count">{summary.mapped} mapped</span>
          </header>
          <div className="operations-map">
            <StructureMap
              cases={cases}
              {...(cases.find((item) => item.profile)?.id
                ? { activeCaseId: cases.find((item) => item.profile)!.id }
                : {})}
            />
            {!loading && summary.mapped === 0 ? (
              <div className="map-empty-note">
                <MapPin size={17} /> Mapped properties will appear here.
              </div>
            ) : null}
          </div>
        </article>

        <section className="operations-panel recent-panel">
          <header className="panel-heading">
            <div>
              <span className="panel-kicker">Local workspace</span>
              <h2>Recent properties</h2>
            </div>
            <span className="panel-count">{cases.length}</span>
          </header>

          {loading ? (
            <div className="library-state" role="status">
              <LoaderCircle className="spin" size={17} /> Loading properties
            </div>
          ) : null}

          {!loading && cases.length === 0 ? (
            <div className="library-state library-empty">
              <Building2 size={24} />
              <strong>No prepared properties</strong>
              <span>Enter an address to create the first workspace.</span>
            </div>
          ) : null}

          <div className="property-table">
            {cases.map((item) => {
              const state = propertyStatus(item);
              return (
                <article className="property-record" key={item.id}>
                  <Link to={`/cases/${item.id}`}>
                    <span className="record-icon" aria-hidden="true">
                      {item.status === "briefing_ready" ? (
                        <Box size={17} />
                      ) : item.status === "collecting" ? (
                        <ImageIcon size={17} />
                      ) : (
                        <Building2 size={17} />
                      )}
                    </span>
                    <span className="record-copy">
                      <strong>{item.displayAddress}</strong>
                      <small>
                        <Clock3 size={12} /> Updated{" "}
                        {formatRelative(item.updatedAt)}
                      </small>
                    </span>
                    <span className={`record-status status-${state.tone}`}>
                      {state.label}
                    </span>
                    <ArrowRight size={16} aria-hidden="true" />
                  </Link>
                  <button
                    type="button"
                    className="row-delete"
                    aria-label={`Delete ${item.displayAddress}`}
                    disabled={deleting === item.id}
                    onClick={() => void deleteProperty(item)}
                  >
                    {deleting === item.id ? (
                      <LoaderCircle className="spin" size={15} />
                    ) : (
                      <Trash2 size={15} />
                    )}
                  </button>
                </article>
              );
            })}
          </div>
        </section>
      </section>
    </div>
  );
}

function propertyStatus(item: Case): {
  label: string;
  tone: "neutral" | "active" | "warning" | "danger";
} {
  if (item.status === "collecting")
    return { label: "Collecting", tone: "active" };
  if (item.status === "reconstructing")
    return { label: "Building 3D", tone: "active" };
  if (item.status === "failed")
    return { label: "Needs attention", tone: "danger" };
  if (item.status === "limited_evidence")
    return { label: "Images needed", tone: "warning" };
  if (item.status === "review_required")
    return { label: "AI findings", tone: "warning" };
  return { label: "Ready", tone: "neutral" };
}
