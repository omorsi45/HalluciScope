import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import { AnalysisPage } from './pages/AnalysisPage';
import { DashboardPage } from './pages/DashboardPage';

function ScopeLogo() {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none" className="shrink-0">
      <circle cx="14" cy="14" r="12" stroke="url(#logo-grad)" strokeWidth="1.5" opacity="0.6" />
      <circle cx="14" cy="14" r="7" stroke="#00e5ff" strokeWidth="1.5" />
      <circle cx="14" cy="14" r="2.5" fill="#00e5ff" />
      <line x1="14" y1="0" x2="14" y2="5" stroke="#00e5ff" strokeWidth="1" opacity="0.4" />
      <line x1="14" y1="23" x2="14" y2="28" stroke="#00e5ff" strokeWidth="1" opacity="0.4" />
      <line x1="0" y1="14" x2="5" y2="14" stroke="#00e5ff" strokeWidth="1" opacity="0.4" />
      <line x1="23" y1="14" x2="28" y2="14" stroke="#00e5ff" strokeWidth="1" opacity="0.4" />
      <defs>
        <linearGradient id="logo-grad" x1="0" y1="0" x2="28" y2="28">
          <stop stopColor="#00e5ff" />
          <stop offset="1" stopColor="#0891b2" />
        </linearGradient>
      </defs>
    </svg>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen flex flex-col">
        {/* ── Navigation ── */}
        <nav className="sticky top-0 z-50 border-b border-border-dim backdrop-blur-xl bg-obsidian/70">
          <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
            <NavLink to="/" className="flex items-center gap-3 group">
              <ScopeLogo />
              <span className="font-display text-xl font-700 tracking-tight text-text-primary">
                Halluci<span className="text-cyan-glow group-hover:drop-shadow-[0_0_8px_rgba(0,229,255,0.5)] transition-all duration-300">Scope</span>
              </span>
            </NavLink>

            <div className="flex items-center gap-1 rounded-full border border-border-dim bg-surface-0/50 px-1 py-1">
              <NavLink
                to="/"
                end
                className={({ isActive }) =>
                  `rounded-full px-5 py-1.5 text-sm font-body font-500 transition-all duration-300 ${
                    isActive
                      ? 'bg-surface-2 text-cyan-glow shadow-[0_0_12px_rgba(0,229,255,0.15)]'
                      : 'text-text-secondary hover:text-text-primary'
                  }`
                }
              >
                Analyze
              </NavLink>
              <NavLink
                to="/dashboard"
                className={({ isActive }) =>
                  `rounded-full px-5 py-1.5 text-sm font-body font-500 transition-all duration-300 ${
                    isActive
                      ? 'bg-surface-2 text-cyan-glow shadow-[0_0_12px_rgba(0,229,255,0.15)]'
                      : 'text-text-secondary hover:text-text-primary'
                  }`
                }
              >
                History
              </NavLink>
            </div>
          </div>
        </nav>

        {/* ── Main content ── */}
        <main className="flex-1 mx-auto w-full max-w-7xl px-6 py-8">
          <Routes>
            <Route path="/" element={<AnalysisPage />} />
            <Route path="/dashboard" element={<DashboardPage />} />
          </Routes>
        </main>

        {/* ── Footer ── */}
        <footer className="border-t border-border-dim px-6 py-4">
          <p className="text-center font-mono text-xs text-text-muted tracking-wider">
            HALLUCISCOPE v1.0 &middot; MULTI-SIGNAL HALLUCINATION DETECTION
          </p>
        </footer>
      </div>
    </BrowserRouter>
  );
}
