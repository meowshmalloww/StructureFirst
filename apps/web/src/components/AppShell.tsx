import {
  Building2,
  LayoutDashboard,
  LogOut,
  Moon,
  Settings,
  Sun,
} from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";
import { useEffect, useState } from "react";
import { api } from "../lib/api";

type Props = { onLogout: () => void };

export function AppShell({ onLogout }: Props) {
  const [theme, setTheme] = useState<"light" | "dark">(() =>
    localStorage.getItem("sf-theme") === "dark" ? "dark" : "light",
  );

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("sf-theme", theme);
  }, [theme]);

  async function logout() {
    await api.logout().catch(() => undefined);
    onLogout();
  }

  return (
    <div className="app-shell">
      <a href="#main-content" className="skip-link">
        Skip to main content
      </a>
      <aside className="primary-nav" aria-label="Primary navigation">
        <div className="brand-lockup">
          <span className="brand-mark" aria-hidden="true">
            <Building2 size={19} strokeWidth={1.8} />
          </span>
          <span>
            <strong>StructureFirst</strong>
            <small>Emergency structure intelligence</small>
          </span>
        </div>
        <span className="nav-section-label">Workspace</span>
        <nav>
          <NavLink to="/" end>
            <LayoutDashboard size={18} />
            Operations
          </NavLink>
          <NavLink to="/settings">
            <Settings size={18} />
            Settings
          </NavLink>
        </nav>
        <div className="nav-actions">
          <span className="nav-action-label">Appearance</span>
          <div className="theme-switch" aria-label="Color theme">
            <button
              type="button"
              aria-pressed={theme === "light"}
              onClick={() => setTheme("light")}
            >
              <Sun size={15} /> Light
            </button>
            <button
              type="button"
              aria-pressed={theme === "dark"}
              onClick={() => setTheme("dark")}
            >
              <Moon size={15} /> Dark
            </button>
          </div>
          <button
            type="button"
            className="sign-out-button"
            onClick={() => void logout()}
          >
            <LogOut size={16} /> Sign out
          </button>
        </div>
      </aside>
      <main id="main-content" className="main-content">
        <Outlet />
      </main>
    </div>
  );
}
