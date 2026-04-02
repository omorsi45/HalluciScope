import { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';

interface Props {
  chunks: string[];
}

export function ChunkViewer({ chunks }: Props) {
  const [open, setOpen] = useState(false);

  if (chunks.length === 0) return null;

  return (
    <div className="glass-panel overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-5 py-3.5 group"
      >
        <div className="flex items-center gap-2">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" className="text-cyan-dim">
            <rect x="2" y="3" width="12" height="10" rx="1.5" stroke="currentColor" strokeWidth="1.2" />
            <line x1="5" y1="6.5" x2="11" y2="6.5" stroke="currentColor" strokeWidth="1" opacity="0.5" />
            <line x1="5" y1="9" x2="9" y2="9" stroke="currentColor" strokeWidth="1" opacity="0.5" />
          </svg>
          <span className="font-mono text-xs text-text-secondary tracking-wider uppercase">
            Retrieved Chunks
          </span>
          <span className="font-mono text-[10px] text-text-muted bg-surface-2 px-2 py-0.5 rounded-full">
            {chunks.length}
          </span>
        </div>
        <svg
          width="12" height="12" viewBox="0 0 12 12"
          className={`text-text-muted transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
        >
          <path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" fill="none" />
        </svg>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className="border-t border-border-dim px-5 py-4 space-y-2.5">
              {chunks.map((chunk, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.05 }}
                  className="glass-panel-inset p-3 flex gap-3"
                >
                  <span className="font-mono text-[10px] text-cyan-dim shrink-0 mt-0.5">
                    {String(i + 1).padStart(2, '0')}
                  </span>
                  <p className="text-xs text-text-secondary leading-relaxed">
                    {chunk}
                  </p>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
