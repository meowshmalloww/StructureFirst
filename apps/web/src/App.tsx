import { LockKeyhole } from "lucide-react";
import { useEffect, useState, type FormEvent } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { api } from "./lib/api";
import { AppShell } from "./components/AppShell";
import { DashboardPage } from "./pages/DashboardPage";
import { CasePage } from "./pages/CasePage";
import { SettingsPage } from "./pages/SettingsPage";

type Session = { required: boolean; authenticated: boolean };

export default function App() {
  const [session, setSession] = useState<Session>();
  const [sessionError, setSessionError] = useState<string>();

  useEffect(() => {
    api
      .session()
      .then(setSession)
      .catch((error: unknown) =>
        setSessionError(
          error instanceof Error ? error.message : "Server unavailable.",
        ),
      );
  }, []);

  if (sessionError)
    return (
      <div className="boot-state">
        <strong>StructureFirst service unavailable</strong>
        <span>{sessionError}</span>
        <button onClick={() => window.location.reload()}>Retry</button>
      </div>
    );
  if (!session)
    return (
      <div className="boot-state">
        <span className="loader" />
        <strong>Starting StructureFirst</strong>
        <span>Checking the local case service…</span>
      </div>
    );
  if (session.required && !session.authenticated)
    return <LoginScreen onAuthenticated={setSession} />;

  return (
    <BrowserRouter>
      <Routes>
        <Route
          element={
            <AppShell
              onLogout={() => setSession({ ...session, authenticated: false })}
            />
          }
        >
          <Route index element={<DashboardPage />} />
          <Route path="cases/:caseId" element={<CasePage />} />
          <Route path="settings" element={<SettingsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

function LoginScreen({
  onAuthenticated,
}: {
  onAuthenticated: (session: Session) => void;
}) {
  const [error, setError] = useState<string>();
  const [busy, setBusy] = useState(false);
  async function login(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError(undefined);
    const value = String(
      new FormData(event.currentTarget).get("accessKey") ?? "",
    );
    try {
      onAuthenticated(await api.login(value));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Sign-in failed.");
      setBusy(false);
    }
  }
  return (
    <main className="login-page">
      <section className="login-panel">
        <div className="login-mark">
          <LockKeyhole size={23} />
        </div>
        <span className="eyebrow">Trusted workstation access</span>
        <h1>StructureFirst</h1>
        <p>
          Enter the access key configured for this local or LAN installation.
        </p>
        <form onSubmit={(event) => void login(event)}>
          <label>
            Access key
            <input
              name="accessKey"
              type="password"
              autoFocus
              required
              autoComplete="current-password"
            />
          </label>
          <button className="primary-button" disabled={busy}>
            {busy ? "Verifying…" : "Open operations console"}
          </button>
          {error ? (
            <span className="dialog-error" role="alert">
              {error}
            </span>
          ) : null}
        </form>
      </section>
      <aside>
        <strong>Local-first property intelligence</strong>
        <span>
          Property photos and 3D files stay on the configured workstation.
          Access is intended only for authorized users on a trusted network.
        </span>
      </aside>
    </main>
  );
}
