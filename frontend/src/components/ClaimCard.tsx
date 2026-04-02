import { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { submitFeedback, type ClaimResponse } from '../api/client';

const TIER_CONFIG = {
  supported: {
    color: 'var(--color-verdict-safe)',
    bgClass: 'bg-verdict-safe/5',
    borderClass: 'border-verdict-safe/20',
    label: 'SUPPORTED',
    icon: (
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
        <circle cx="7" cy="7" r="6" stroke="currentColor" strokeWidth="1.2" />
        <path d="M4.5 7L6.5 9L9.5 5.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  uncertain: {
    color: 'var(--color-verdict-warn)',
    bgClass: 'bg-verdict-warn/5',
    borderClass: 'border-verdict-warn/20',
    label: 'UNCERTAIN',
    icon: (
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
        <circle cx="7" cy="7" r="6" stroke="currentColor" strokeWidth="1.2" />
        <path d="M7 4.5V8" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
        <circle cx="7" cy="10" r="0.7" fill="currentColor" />
      </svg>
    ),
  },
  hallucinated: {
    color: 'var(--color-verdict-danger)',
    bgClass: 'bg-verdict-danger/5',
    borderClass: 'border-verdict-danger/20',
    label: 'HALLUCINATED',
    icon: (
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
        <circle cx="7" cy="7" r="6" stroke="currentColor" strokeWidth="1.2" />
        <path d="M5 5L9 9M9 5L5 9" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
      </svg>
    ),
  },
};

interface Props {
  claim: ClaimResponse;
  index: number;
  analysisId: number | null;
}

export function ClaimCard({ claim, index, analysisId }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [feedbackNote, setFeedbackNote] = useState('');
  const [feedbackSent, setFeedbackSent] = useState(false);
  const tier = TIER_CONFIG[claim.tier];

  const handleFeedback = async (isCorrect: boolean) => {
    if (!analysisId) return;
    await submitFeedback(analysisId, index, isCorrect, feedbackNote || undefined);
    setFeedbackSent(true);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.08, duration: 0.35 }}
      className={`rounded-xl border ${tier.borderClass} ${tier.bgClass} overflow-hidden transition-colors duration-300`}
    >
      {/* Header — always visible */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-start gap-3 px-4 py-3.5 text-left group"
      >
        {/* Tier indicator bar */}
        <div
          className="mt-0.5 shrink-0 transition-transform duration-200 group-hover:scale-110"
          style={{ color: tier.color }}
        >
          {tier.icon}
        </div>

        <p className="flex-1 text-sm text-text-primary leading-relaxed font-body">
          {claim.claim}
        </p>

        <div className="shrink-0 flex items-center gap-2 mt-0.5">
          <span
            className="font-mono text-[10px] font-600 tracking-wider"
            style={{ color: tier.color }}
          >
            {tier.label}
          </span>
          <span className="font-mono text-xs text-text-muted">
            {claim.hallucination_score.toFixed(2)}
          </span>
          <svg
            width="12" height="12" viewBox="0 0 12 12"
            className={`text-text-muted transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`}
          >
            <path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" fill="none" />
          </svg>
        </div>
      </button>

      {/* Expandable detail panel */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: 'easeInOut' }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 space-y-3 border-t border-border-dim pt-3">
              {/* Verifier readouts */}
              <div className="grid grid-cols-3 gap-2">
                <VerifierCell
                  label="NLI"
                  content={
                    <>
                      <Row label="E" value={claim.verifier_details.nli?.entailment} color="var(--color-verdict-safe)" />
                      <Row label="N" value={claim.verifier_details.nli?.neutral} color="var(--color-verdict-warn)" />
                      <Row label="C" value={claim.verifier_details.nli?.contradiction} color="var(--color-verdict-danger)" />
                    </>
                  }
                />
                <VerifierCell
                  label="Consistency"
                  content={
                    <p className="font-mono text-sm text-text-primary">
                      {claim.verifier_details.consistency?.appearances ?? '?'}
                      <span className="text-text-muted">/</span>
                      {claim.verifier_details.consistency?.n_samples ?? '?'}
                      <span className="text-text-muted text-[10px] ml-1">samples</span>
                    </p>
                  }
                />
                <VerifierCell
                  label="Similarity"
                  content={
                    <p className="font-mono text-sm text-text-primary">
                      {claim.verifier_details.similarity?.max_similarity?.toFixed(3) ?? '-'}
                    </p>
                  }
                />
              </div>

              {/* Matched source chunk */}
              {claim.verifier_details.nli?.matched_chunk && (
                <div className="glass-panel-inset p-3">
                  <p className="font-mono text-[10px] text-text-muted tracking-wider uppercase mb-1.5">Best Source Match</p>
                  <p className="text-xs text-text-secondary leading-relaxed italic">
                    "{claim.verifier_details.nli.matched_chunk}"
                  </p>
                </div>
              )}

              {/* Feedback */}
              {!feedbackSent ? (
                <div className="flex items-center gap-2 pt-1">
                  <span className="font-mono text-[10px] text-text-muted tracking-wider uppercase">Feedback</span>
                  <button
                    onClick={() => handleFeedback(true)}
                    className="rounded-lg bg-verdict-safe/10 border border-verdict-safe/20 px-2.5 py-1 text-xs text-verdict-safe hover:bg-verdict-safe/20 transition-colors"
                  >
                    Correct
                  </button>
                  <button
                    onClick={() => handleFeedback(false)}
                    className="rounded-lg bg-verdict-danger/10 border border-verdict-danger/20 px-2.5 py-1 text-xs text-verdict-danger hover:bg-verdict-danger/20 transition-colors"
                  >
                    Wrong
                  </button>
                  <input
                    value={feedbackNote}
                    onChange={(e) => setFeedbackNote(e.target.value)}
                    placeholder="Optional note..."
                    className="flex-1 scope-input px-2.5 py-1 text-xs rounded-lg"
                  />
                </div>
              ) : (
                <p className="font-mono text-[10px] text-cyan-dim tracking-wider">FEEDBACK RECORDED</p>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

function VerifierCell({ label, content }: { label: string; content: React.ReactNode }) {
  return (
    <div className="glass-panel-inset p-2.5">
      <p className="font-mono text-[10px] text-text-muted tracking-wider uppercase mb-1.5">{label}</p>
      {content}
    </div>
  );
}

function Row({ label, value, color }: { label: string; value?: number; color: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="font-mono text-[10px] text-text-muted">{label}</span>
      <span className="font-mono text-xs" style={{ color }}>
        {value?.toFixed(2) ?? '-'}
      </span>
    </div>
  );
}
