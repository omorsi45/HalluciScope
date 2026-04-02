import { useEffect, useState } from 'react';
import { motion } from 'motion/react';
import { Link } from 'react-router-dom';
import { listAnalyses, type AnalysisListItem } from '../api/client';

export function DashboardPage() {
  const [analyses, setAnalyses] = useState<AnalysisListItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listAnalyses()
      .then(setAnalyses)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const tierInfo = (score: number) => {
    if (score < 0.2) return { color: 'var(--color-verdict-safe)', label: 'LOW' };
    if (score > 0.6) return { color: 'var(--color-verdict-danger)', label: 'HIGH' };
    return { color: 'var(--color-verdict-warn)', label: 'MED' };
  };

  return (
    <div className="space-y-8">
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-end justify-between"
      >
        <div>
          <h1 className="font-display text-3xl font-800 tracking-tight text-text-primary">
            Analysis History
          </h1>
          <p className="font-body text-sm text-text-secondary mt-1">
            {analyses.length} analyses recorded
          </p>
        </div>
        <Link
          to="/"
          className="font-mono text-[10px] tracking-wider text-cyan-dim border border-cyan-dim/20 rounded-full px-4 py-1.5 hover:bg-cyan-glow/5 hover:border-cyan-dim/40 transition-all duration-200"
        >
          NEW ANALYSIS
        </Link>
      </motion.div>

      {loading ? (
        <div className="glass-panel p-12 flex flex-col items-center gap-3">
          <svg className="animate-spin h-6 w-6 text-cyan-dim" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" opacity="0.3" />
            <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
          <span className="font-mono text-xs text-text-muted">Loading analyses...</span>
        </div>
      ) : analyses.length === 0 ? (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="glass-panel flex flex-col items-center justify-center py-20"
        >
          <svg width="48" height="48" viewBox="0 0 48 48" fill="none" className="mb-4 opacity-20">
            <rect x="6" y="10" width="36" height="28" rx="3" stroke="var(--color-cyan-glow)" strokeWidth="1" />
            <line x1="12" y1="20" x2="36" y2="20" stroke="var(--color-cyan-glow)" strokeWidth="0.5" opacity="0.5" />
            <line x1="12" y1="26" x2="30" y2="26" stroke="var(--color-cyan-glow)" strokeWidth="0.5" opacity="0.5" />
            <line x1="12" y1="32" x2="24" y2="32" stroke="var(--color-cyan-glow)" strokeWidth="0.5" opacity="0.5" />
          </svg>
          <p className="font-display text-sm font-600 text-text-muted mb-1">No analyses yet</p>
          <p className="font-body text-xs text-text-muted/60">
            Run your first analysis to see results here
          </p>
        </motion.div>
      ) : (
        <div className="glass-panel overflow-hidden">
          {/* Table header */}
          <div className="grid grid-cols-[60px_1fr_100px_80px_140px] gap-4 px-5 py-3 border-b border-border-dim">
            <span className="font-mono text-[10px] text-text-muted tracking-[0.15em] uppercase">ID</span>
            <span className="font-mono text-[10px] text-text-muted tracking-[0.15em] uppercase">Question</span>
            <span className="font-mono text-[10px] text-text-muted tracking-[0.15em] uppercase">Score</span>
            <span className="font-mono text-[10px] text-text-muted tracking-[0.15em] uppercase">Risk</span>
            <span className="font-mono text-[10px] text-text-muted tracking-[0.15em] uppercase">Date</span>
          </div>

          {/* Table rows */}
          {analyses.map((a, i) => {
            const tier = tierInfo(a.overall_score);
            return (
              <motion.div
                key={a.id}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.04, duration: 0.3 }}
                className="grid grid-cols-[60px_1fr_100px_80px_140px] gap-4 px-5 py-3.5 border-b border-border-dim/50 hover:bg-surface-1/50 transition-colors group"
              >
                <span className="font-mono text-xs text-text-muted">{a.id}</span>
                <span className="text-sm text-text-primary truncate font-body group-hover:text-cyan-glow transition-colors">
                  {a.question}
                </span>
                <span className="font-mono text-sm font-600" style={{ color: tier.color }}>
                  {a.overall_score.toFixed(2)}
                </span>
                <span className="flex items-center gap-1.5">
                  <span
                    className="w-1.5 h-1.5 rounded-full"
                    style={{ backgroundColor: tier.color, boxShadow: `0 0 6px ${tier.color}` }}
                  />
                  <span className="font-mono text-[10px] tracking-wider" style={{ color: tier.color }}>
                    {tier.label}
                  </span>
                </span>
                <span className="font-mono text-xs text-text-muted">
                  {a.created_at ? new Date(a.created_at).toLocaleDateString('en-US', {
                    month: 'short',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit',
                  }) : '-'}
                </span>
              </motion.div>
            );
          })}
        </div>
      )}
    </div>
  );
}
