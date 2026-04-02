import { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { analyzeDocument, type AnalyzeResponse } from '../api/client';
import { ClaimCard } from '../components/ClaimCard';
import { DocumentInput } from '../components/DocumentInput';
import { ScoreGauge } from '../components/ScoreGauge';
import { ChunkViewer } from '../components/ChunkViewer';

const DEMO_DOCUMENT = `Albert Einstein was born on March 14, 1879, in Ulm, in the Kingdom of Württemberg in the German Empire. He developed the theory of special relativity in 1905 while working as a patent clerk in Bern, Switzerland. Einstein received the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect. He emigrated to the United States in 1933 and accepted a position at the Institute for Advanced Study in Princeton, New Jersey, where he worked until his death on April 18, 1955.`;
const DEMO_QUESTION = "Tell me about Albert Einstein's life and achievements.";

export function AnalysisPage() {
  const [documentText, setDocumentText] = useState('');
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await analyzeDocument(documentText, question);
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const handleDemo = () => {
    setDocumentText(DEMO_DOCUMENT);
    setQuestion(DEMO_QUESTION);
  };

  return (
    <div className="space-y-8">
      {/* Hero heading */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center space-y-2"
      >
        <h1 className="font-display text-3xl font-800 tracking-tight text-text-primary">
          Analyze for Hallucinations
        </h1>
        <p className="font-body text-sm text-text-secondary max-w-lg mx-auto">
          Paste a source document and the question that was asked. HalluciScope will extract claims
          from the AI answer and verify each one against the source.
        </p>
      </motion.div>

      <div className="grid grid-cols-1 gap-8 lg:grid-cols-5">
        {/* ── Left Panel: Input (2/5) ── */}
        <motion.div
          initial={{ opacity: 0, x: -12 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4, delay: 0.1 }}
          className="lg:col-span-2 space-y-5"
        >
          <div className="glass-panel p-5 space-y-5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-cyan-glow animate-pulse" />
                <span className="font-mono text-[10px] text-text-muted tracking-[0.15em] uppercase">
                  Input
                </span>
              </div>
              <button
                onClick={handleDemo}
                className="font-mono text-[10px] tracking-wider text-cyan-dim border border-cyan-dim/20 rounded-full px-3 py-1 hover:bg-cyan-glow/5 hover:border-cyan-dim/40 transition-all duration-200"
              >
                LOAD DEMO
              </button>
            </div>

            <DocumentInput
              documentText={documentText}
              question={question}
              onDocumentChange={setDocumentText}
              onQuestionChange={setQuestion}
            />

            <button
              onClick={handleAnalyze}
              disabled={loading || !documentText || !question}
              className="relative w-full rounded-xl py-3.5 font-display font-600 text-sm tracking-wide transition-all duration-300 overflow-hidden disabled:opacity-40 disabled:cursor-not-allowed group"
              style={{
                background: loading
                  ? 'var(--color-surface-2)'
                  : 'linear-gradient(135deg, var(--color-cyan-dim), var(--color-cyan-glow))',
                color: loading ? 'var(--color-text-muted)' : 'var(--color-obsidian)',
              }}
            >
              {/* Shimmer overlay on hover */}
              <span className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent translate-x-[-200%] group-hover:translate-x-[200%] transition-transform duration-700" />
              <span className="relative">
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" opacity="0.3" />
                      <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                    </svg>
                    Scanning Claims...
                  </span>
                ) : (
                  'Run Analysis'
                )}
              </span>
            </button>

            <AnimatePresence>
              {error && (
                <motion.p
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="text-verdict-danger text-xs font-mono bg-verdict-danger/10 border border-verdict-danger/20 rounded-lg px-3 py-2"
                >
                  {error}
                </motion.p>
              )}
            </AnimatePresence>
          </div>
        </motion.div>

        {/* ── Right Panel: Results (3/5) ── */}
        <motion.div
          initial={{ opacity: 0, x: 12 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4, delay: 0.2 }}
          className="lg:col-span-3 space-y-5"
        >
          <AnimatePresence mode="wait">
            {result ? (
              <motion.div
                key="results"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.4 }}
                className="space-y-5"
              >
                {/* Score + answer header */}
                <div className="glass-panel p-5">
                  <div className="flex items-start gap-6">
                    <ScoreGauge score={result.overall_score} />
                    <div className="flex-1 min-w-0 space-y-3">
                      <div>
                        <span className="font-mono text-[10px] text-text-muted tracking-[0.15em] uppercase">
                          AI Answer
                        </span>
                        <p className="text-sm text-text-primary leading-relaxed mt-1 font-body">
                          {result.answer}
                        </p>
                      </div>
                      <div className="flex items-center gap-4 text-[10px] font-mono text-text-muted">
                        <span>{result.scored_claims.length} claims extracted</span>
                        <span className="w-px h-3 bg-border-subtle" />
                        <span>{result.retrieved_chunks.length} chunks retrieved</span>
                        {result.id && (
                          <>
                            <span className="w-px h-3 bg-border-subtle" />
                            <span>ID: {result.id}</span>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Claims list */}
                <div className="space-y-2.5">
                  <span className="font-mono text-[10px] text-text-muted tracking-[0.15em] uppercase">
                    Claim Verdicts
                  </span>
                  {result.scored_claims.map((claim, i) => (
                    <ClaimCard
                      key={i}
                      claim={claim}
                      index={i}
                      analysisId={result.id}
                    />
                  ))}
                </div>

                {/* Retrieved chunks */}
                <ChunkViewer chunks={result.retrieved_chunks} />
              </motion.div>
            ) : (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="glass-panel flex flex-col items-center justify-center py-24 px-8"
              >
                {/* Scope crosshair illustration */}
                <svg width="80" height="80" viewBox="0 0 80 80" fill="none" className="mb-6 opacity-20">
                  <circle cx="40" cy="40" r="30" stroke="var(--color-cyan-glow)" strokeWidth="1" strokeDasharray="4 4" />
                  <circle cx="40" cy="40" r="18" stroke="var(--color-cyan-glow)" strokeWidth="0.5" />
                  <circle cx="40" cy="40" r="4" fill="var(--color-cyan-glow)" opacity="0.3" />
                  <line x1="40" y1="4" x2="40" y2="20" stroke="var(--color-cyan-glow)" strokeWidth="0.5" />
                  <line x1="40" y1="60" x2="40" y2="76" stroke="var(--color-cyan-glow)" strokeWidth="0.5" />
                  <line x1="4" y1="40" x2="20" y2="40" stroke="var(--color-cyan-glow)" strokeWidth="0.5" />
                  <line x1="60" y1="40" x2="76" y2="40" stroke="var(--color-cyan-glow)" strokeWidth="0.5" />
                </svg>
                <p className="font-display text-sm font-600 text-text-muted mb-1">
                  No analysis yet
                </p>
                <p className="font-body text-xs text-text-muted/60 text-center max-w-xs">
                  Paste a document and question, then hit Run Analysis to scan for hallucinations
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>
    </div>
  );
}
