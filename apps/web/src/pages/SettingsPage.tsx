import {
  Activity,
  Bot,
  Check,
  ExternalLink,
  KeyRound,
  LoaderCircle,
  RefreshCw,
  Save,
  Search,
  ShieldCheck,
  TriangleAlert,
} from "lucide-react";
import {
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type FormEvent,
} from "react";
import type {
  AiProviderCatalogKind,
  AiProviderId,
  AiProviderModel,
  AiProviderModelsResult,
  AiProviderSettings,
  AppSettings,
  SaveAiProviderInput,
} from "@structurefirst/contracts";
import { api } from "../lib/api";

const PROVIDER_COPY: Record<
  AiProviderId,
  {
    access: string;
    detail: string;
    model: string;
    keyUrl: string;
    limitsUrl: string;
  }
> = {
  groq: {
    access: "Free plan available",
    detail: "Only current general-chat Free Plan models are shown.",
    model: "llama-3.3-70b-versatile",
    keyUrl: "https://console.groq.com/keys",
    limitsUrl: "https://console.groq.com/docs/rate-limits",
  },
  cerebras: {
    access: "Free tier available",
    detail: "The key's current public chat catalog is shown.",
    model: "gpt-oss-120b",
    keyUrl: "https://cloud.cerebras.ai/",
    limitsUrl: "https://inference-docs.cerebras.ai/support/pricing",
  },
  openrouter: {
    access: "Zero-price models",
    detail: "Only text-output models priced at $0 are shown.",
    model: "openrouter/free",
    keyUrl: "https://openrouter.ai/settings/keys",
    limitsUrl: "https://openrouter.ai/docs/api/reference/limits",
  },
  nvidia_nim: {
    access: "Developer prototype access",
    detail:
      "Only NVIDIA-documented chat and image-understanding endpoints are shown.",
    model: "meta/llama-3.2-11b-vision-instruct",
    keyUrl: "https://build.nvidia.com/",
    limitsUrl: "https://docs.api.nvidia.com/nim/docs/run-anywhere",
  },
};

type Feedback = {
  tone: "info" | "success" | "error";
  text: string;
};

export function SettingsPage() {
  const [settings, setSettings] = useState<AppSettings>();
  const [selectedId, setSelectedId] = useState<AiProviderId>("openrouter");
  const [error, setError] = useState<string>();

  useEffect(() => {
    api
      .settings()
      .then((next) => {
        setSettings(next);
        setSelectedId(
          next.providers.find((provider) => provider.enabled)?.id ??
            next.providers.find((provider) => provider.configured)?.id ??
            "openrouter",
        );
      })
      .catch((caught: unknown) =>
        setError(
          caught instanceof Error ? caught.message : "Settings unavailable.",
        ),
      );
  }, []);

  const selected = useMemo(
    () => settings?.providers.find((provider) => provider.id === selectedId),
    [selectedId, settings?.providers],
  );

  function mergeProviders(next: AiProviderSettings[]) {
    setSettings((current) =>
      current ? { ...current, providers: next } : current,
    );
  }

  return (
    <div className="simple-settings-page">
      <header className="settings-title">
        <div>
          <span className="eyebrow">Operations configuration</span>
          <h1>Settings</h1>
        </div>
        <p>
          Configure one analysis connection and the automatic property-image
          collector.
        </p>
      </header>

      {error ? (
        <div className="settings-alert" role="alert">
          <TriangleAlert size={16} /> {error}
        </div>
      ) : null}

      {!settings || !selected ? (
        <div className="settings-loading">
          <LoaderCircle className="spin" size={17} /> Loading settings
        </div>
      ) : (
        <div className="settings-dashboard">
          <section className="connection-card ai-connection-card">
            <header>
              <span className="settings-icon">
                <Bot size={18} />
              </span>
              <div>
                <span className="card-kicker">Analysis</span>
                <h2>AI connection</h2>
                <p>
                  Load a live catalog, choose a model, then verify it with one
                  small request.
                </p>
              </div>
            </header>
            <ProviderForm
              key={selected.id}
              provider={selected}
              providers={settings.providers}
              selectedId={selectedId}
              onSelect={setSelectedId}
              onSaved={mergeProviders}
            />
            <footer className="credential-note">
              <ShieldCheck size={16} />
              <span>
                Saved keys are encrypted in the local data folder. The browser
                receives only a masked hint.
              </span>
            </footer>
          </section>

          <section className="connection-card source-connection-card">
            <header>
              <span className="settings-icon">
                <Search size={18} />
              </span>
              <div>
                <span className="card-kicker">Collection</span>
                <h2>Online imagery</h2>
                <p>Runs automatically when a new address is submitted.</p>
              </div>
            </header>
            <SearchSettings
              settings={settings}
              onSaved={(discovery) =>
                setSettings((current) =>
                  current ? { ...current, discovery } : current,
                )
              }
            />
          </section>
        </div>
      )}
    </div>
  );
}

function ProviderForm({
  provider,
  providers,
  selectedId,
  onSelect,
  onSaved,
}: {
  provider: AiProviderSettings;
  providers: AiProviderSettings[];
  selectedId: AiProviderId;
  onSelect: (id: AiProviderId) => void;
  onSaved: (value: AiProviderSettings[]) => void;
}) {
  const copy = PROVIDER_COPY[provider.id];
  const providerInputId = useId();
  const keyInputId = useId();
  const modelInputId = useId();
  const keyInput = useRef<HTMLInputElement>(null);
  const lastAttemptedKey = useRef<string | undefined>(undefined);
  const [busy, setBusy] = useState<"models" | "save" | "remove">();
  const [models, setModels] = useState<AiProviderModel[]>([]);
  const [model, setModel] = useState("");
  const [enabled, setEnabled] = useState(provider.enabled);
  const [catalogKind, setCatalogKind] = useState<AiProviderCatalogKind>();
  const [feedback, setFeedback] = useState<Feedback>();

  const selectedModel = useMemo(
    () => models.find((item) => item.id === model),
    [model, models],
  );
  useEffect(() => {
    if (provider.configured) void loadModels();
  }, []);

  function formInput(form: HTMLFormElement): SaveAiProviderInput {
    const values = new FormData(form);
    const key = String(values.get("apiKey") ?? "").trim();
    return {
      baseUrl: provider.baseUrl,
      model,
      enabled: values.get("enabled") === "on",
      vision: selectedModel?.vision ?? false,
      ...(key ? { apiKey: key } : {}),
      clearKey: false,
    };
  }

  async function discoverModels(apiKey?: string) {
    return api.providerModels(provider.id, apiKey ? { apiKey } : {});
  }

  function applyCatalog(result: AiProviderModelsResult): string {
    setModels(result.models);
    setCatalogKind(result.catalogKind);
    const preferred =
      result.models.find((item) => item.id === model) ??
      result.models.find((item) => item.id === provider.model) ??
      result.models.find((item) => item.id === copy.model) ??
      result.models.find((item) => item.free === true) ??
      result.models[0];
    if (!preferred) throw new Error("The provider returned no models.");
    setModel(preferred.id);
    setFeedback({
      tone: "success",
      text: `${result.notice} Loaded in ${result.latencyMs} ms.`,
    });
    return preferred.id;
  }

  async function loadModels(apiKey?: string) {
    setBusy("models");
    setFeedback({ tone: "info", text: "Loading the live model catalog…" });
    try {
      applyCatalog(await discoverModels(apiKey));
    } catch (caught) {
      setModels([]);
      setModel("");
      setCatalogKind(undefined);
      setFeedback({
        tone: "error",
        text:
          caught instanceof Error ? caught.message : "Models could not load.",
      });
    } finally {
      setBusy(undefined);
    }
  }

  async function saveAll(input: SaveAiProviderInput) {
    const next: AiProviderSettings[] = [];
    for (const item of providers) {
      if (item.id === provider.id) {
        next.push(await api.saveProvider(item.id, input));
      } else if (item.enabled) {
        next.push(
          await api.saveProvider(item.id, {
            baseUrl: item.baseUrl,
            model: item.model,
            enabled: false,
            vision: item.vision,
            clearKey: false,
          }),
        );
      } else {
        next.push(item);
      }
    }
    onSaved(next);
    return next.find((item) => item.id === provider.id)!;
  }

  async function save(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy("save");
    const input = formInput(event.currentTarget);
    try {
      if (!selectedModel) {
        throw new Error("Load the catalog and select a model before saving.");
      }
      setFeedback({
        tone: "info",
        text: `Verifying ${selectedModel.id} with one compatibility request…`,
      });
      const key = keyInput.current?.value.trim();
      const test = await api.testProvider(provider.id, {
        model: selectedModel.id,
        ...(key ? { apiKey: key } : {}),
      });
      if (!test.ok) throw new Error(test.message);
      const saved = await saveAll(input);
      setFeedback({
        tone: "success",
        text: saved.enabled
          ? `${saved.label} is verified and active with ${saved.model} (${test.latencyMs} ms).`
          : `${saved.label} passed verification and was saved disabled (${test.latencyMs} ms).`,
      });
      if (keyInput.current) keyInput.current.value = "";
    } catch (caught) {
      setFeedback({
        tone: "error",
        text: caught instanceof Error ? caught.message : "Save failed.",
      });
    } finally {
      setBusy(undefined);
    }
  }

  async function removeKey() {
    setBusy("remove");
    setFeedback(undefined);
    try {
      const saved = await api.saveProvider(provider.id, {
        baseUrl: provider.baseUrl,
        model: provider.model,
        enabled: false,
        vision: provider.vision,
        clearKey: true,
      });
      setModels([]);
      setModel("");
      setEnabled(false);
      setCatalogKind(undefined);
      onSaved(providers.map((item) => (item.id === saved.id ? saved : item)));
      setFeedback({
        tone: "info",
        text: saved.configured
          ? "Saved key removed. An environment key is still configured."
          : "Saved key removed.",
      });
    } catch (caught) {
      setFeedback({
        tone: "error",
        text: caught instanceof Error ? caught.message : "Key removal failed.",
      });
    } finally {
      setBusy(undefined);
    }
  }

  function checkEnteredKey(value: string) {
    const key = value.trim();
    if (key.length < 8 || key === lastAttemptedKey.current) return;
    lastAttemptedKey.current = key;
    void loadModels(key);
  }

  function apiKeyChanged() {
    lastAttemptedKey.current = undefined;
    if (models.length === 0) return;
    setModels([]);
    setModel("");
    setCatalogKind(undefined);
    setFeedback({
      tone: "info",
      text: "API key changed. Load its model catalog again.",
    });
  }

  return (
    <form
      className="single-provider-form"
      onSubmit={(event) => void save(event)}
    >
      <div className="provider-selection-row">
        <label className="control-label" htmlFor={providerInputId}>
          Provider
          <select
            id={providerInputId}
            value={selectedId}
            disabled={Boolean(busy)}
            onChange={(event) =>
              onSelect(event.currentTarget.value as AiProviderId)
            }
          >
            {providers.map((item) => (
              <option key={item.id} value={item.id}>
                {item.label}
                {item.configured ? " — key saved" : ""}
              </option>
            ))}
          </select>
        </label>
        <div className="provider-access-note">
          <span>{copy.access}</span>
          <p>{copy.detail}</p>
          <a href={copy.limitsUrl} target="_blank" rel="noreferrer">
            Access details <ExternalLink size={11} />
          </a>
        </div>
      </div>

      <div className="provider-fields">
        <div className="control-label">
          <span className="control-label-row">
            <label htmlFor={keyInputId}>API key</label>
            <a href={copy.keyUrl} target="_blank" rel="noreferrer">
              Get key <ExternalLink size={12} />
            </a>
          </span>
          <span className="key-input-row">
            <span className="input-with-icon">
              <KeyRound size={15} />
              <input
                id={keyInputId}
                ref={keyInput}
                name="apiKey"
                type="password"
                autoComplete="off"
                placeholder={
                  provider.configured
                    ? `Saved ${provider.keyHint ?? "key"} — leave blank to keep`
                    : "Paste an API key"
                }
                onChange={apiKeyChanged}
                onBlur={(event) => checkEnteredKey(event.currentTarget.value)}
              />
            </span>
            <button
              type="button"
              className="secondary-button load-models-button"
              disabled={Boolean(busy)}
              onClick={() =>
                void loadModels(keyInput.current?.value.trim() || undefined)
              }
            >
              {busy === "models" ? (
                <LoaderCircle className="spin" size={15} />
              ) : (
                <RefreshCw size={15} />
              )}
              {models.length ? "Reload" : "Load models"}
            </button>
          </span>
        </div>

        <div className="control-label model-control">
          <span className="control-label-row">
            <label htmlFor={modelInputId}>Model</label>
            <span className="model-count">
              {models.length
                ? `${models.length} ${catalogCountLabel(catalogKind)}`
                : "Catalog not loaded"}
            </span>
          </span>
          <select
            id={modelInputId}
            name="model"
            value={model}
            onChange={(event) => {
              setModel(event.currentTarget.value);
              setFeedback(undefined);
            }}
            required
            disabled={models.length === 0 || busy === "models"}
          >
            <option value="" disabled>
              {busy === "models" ? "Loading models…" : "Load models first"}
            </option>
            {models.length ? (
              <optgroup label={catalogGroupLabel(catalogKind, models.length)}>
                {models.map((item) => (
                  <option key={item.id} value={item.id}>
                    {modelOptionLabel(item)}
                  </option>
                ))}
              </optgroup>
            ) : null}
          </select>
          <small className="model-meta">
            {selectedModel
              ? modelSummary(selectedModel)
              : "Load the key-scoped, no-cost catalog; manual IDs are not accepted."}
          </small>
        </div>
      </div>

      <div className="connection-actions provider-actions">
        <label className="plain-check">
          <input
            name="enabled"
            type="checkbox"
            checked={enabled}
            onChange={(event) => setEnabled(event.currentTarget.checked)}
          />
          Use for structure analysis
        </label>
        <span />
        {provider.configured ? (
          <button
            type="button"
            className="text-button danger-text"
            disabled={Boolean(busy)}
            onClick={() => void removeKey()}
          >
            Remove key
          </button>
        ) : null}
        <button
          className="primary-button"
          disabled={Boolean(busy) || !selectedModel}
        >
          {busy === "save" ? (
            <LoaderCircle className="spin" size={15} />
          ) : (
            <Save size={15} />
          )}
          Verify & save
        </button>
      </div>

      <p className="provider-test-note">
        <Activity size={13} /> Verify & save sends one short JSON-format request
        and activates the connection only when the selected model passes.
      </p>

      {feedback ? (
        <p
          className={`connection-result result-${feedback.tone}`}
          role={feedback.tone === "error" ? "alert" : "status"}
        >
          {feedback.tone === "error" ? (
            <TriangleAlert size={14} />
          ) : feedback.tone === "success" ? (
            <Check size={14} />
          ) : (
            <RefreshCw size={14} />
          )}
          {feedback.text}
        </p>
      ) : null}
    </form>
  );
}

function SearchSettings({
  settings,
  onSaved,
}: {
  settings: AppSettings;
  onSaved: (value: AppSettings["discovery"]) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState<Feedback>();

  async function save(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setFeedback(undefined);
    const form = event.currentTarget;
    const values = new FormData(form);
    const enabled = values.get("automatic") === "on";
    const key = String(values.get("braveApiKey") ?? "").trim();
    try {
      const saved = await api.saveDiscovery({
        openverseEnabled: enabled,
        browserEnabled: enabled,
        ...(key ? { braveApiKey: key } : {}),
        clearBraveKey: values.get("clearBraveKey") === "on",
      });
      onSaved(saved);
      setFeedback({ tone: "success", text: "Collection settings saved." });
      const input = form.elements.namedItem("braveApiKey");
      if (input instanceof HTMLInputElement) input.value = "";
    } catch (caught) {
      setFeedback({
        tone: "error",
        text: caught instanceof Error ? caught.message : "Save failed.",
      });
    } finally {
      setBusy(false);
    }
  }

  return (
    <form
      className="search-settings-form"
      onSubmit={(event) => void save(event)}
    >
      <label className="search-toggle-row">
        <input
          name="automatic"
          type="checkbox"
          defaultChecked={
            settings.discovery.openverseEnabled &&
            settings.discovery.browserEnabled
          }
        />
        <span>
          <strong>Automatic image collection</strong>
          <small>
            Uses KartaView, Wikimedia Commons, Openverse, and a local Chrome or
            Edge browser. No search key is required.
          </small>
        </span>
      </label>

      <details className="optional-search-key">
        <summary>Optional Brave Search connection</summary>
        <p>
          Add Brave only if you want its structured image index in addition to
          the key-free collectors.
        </p>
        <label>
          Brave Search key
          <input
            name="braveApiKey"
            type="password"
            autoComplete="off"
            placeholder={
              settings.discovery.braveConfigured
                ? `Saved ${settings.discovery.braveKeyHint ?? "key"}`
                : "Paste a Brave Search key"
            }
          />
        </label>
        {settings.discovery.braveConfigured ? (
          <label className="plain-check">
            <input name="clearBraveKey" type="checkbox" /> Remove saved Brave
            key
          </label>
        ) : null}
      </details>

      <div className="connection-actions search-actions">
        <span />
        <button className="primary-button" disabled={busy}>
          {busy ? (
            <LoaderCircle className="spin" size={15} />
          ) : (
            <Save size={15} />
          )}
          Save collection
        </button>
      </div>
      {feedback ? (
        <p
          className={`connection-result result-${feedback.tone}`}
          role={feedback.tone === "error" ? "alert" : "status"}
        >
          {feedback.tone === "error" ? (
            <TriangleAlert size={14} />
          ) : (
            <Check size={14} />
          )}
          {feedback.text}
        </p>
      ) : null}
    </form>
  );
}

function modelOptionLabel(model: AiProviderModel): string {
  const details = [
    model.name !== model.id ? model.name : undefined,
    model.ownedBy,
  ]
    .filter(Boolean)
    .join(" — ");
  return details ? `${details} — ${model.id}` : model.id;
}

function modelSummary(model: AiProviderModel): string {
  const details = [
    model.ownedBy,
    model.contextWindow
      ? `${formatTokens(model.contextWindow)} context`
      : undefined,
    model.free === true ? "free" : undefined,
    model.vision ? "image input" : undefined,
  ].filter(Boolean);
  return details.length ? details.join(" · ") : "Available from provider";
}

function catalogCountLabel(kind?: AiProviderCatalogKind): string {
  return kind === "prototype" ? "prototype choices" : "no-cost choices";
}

function catalogGroupLabel(
  kind: AiProviderCatalogKind | undefined,
  count: number,
): string {
  if (kind === "prototype") return `Prototype chat models (${count})`;
  if (kind === "free_models") return `Zero-price text models (${count})`;
  return `Free-plan chat models (${count})`;
}

function formatTokens(value: number): string {
  return value >= 1000 && value % 1000 === 0
    ? `${value / 1000}K`
    : value.toLocaleString();
}
