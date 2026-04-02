import { useEffect, useRef } from 'react';

interface Props {
  score: number;
  size?: number;
}

export function ScoreGauge({ score, size = 120 }: Props) {
  const circleRef = useRef<SVGCircleElement>(null);
  const radius = (size - 12) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - score * circumference;

  const color =
    score < 0.2 ? 'var(--color-verdict-safe)' :
    score > 0.6 ? 'var(--color-verdict-danger)' :
    'var(--color-verdict-warn)';

  const label =
    score < 0.2 ? 'LOW RISK' :
    score > 0.6 ? 'HIGH RISK' :
    'UNCERTAIN';

  useEffect(() => {
    const el = circleRef.current;
    if (!el) return;
    el.style.setProperty('--gauge-circumference', String(circumference));
    el.style.setProperty('--gauge-offset', String(offset));
    el.style.strokeDashoffset = String(circumference);
    requestAnimationFrame(() => {
      el.style.transition = 'stroke-dashoffset 1.2s cubic-bezier(0.25, 0.46, 0.45, 0.94)';
      el.style.strokeDashoffset = String(offset);
    });
  }, [score, circumference, offset]);

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          {/* Track */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="var(--color-surface-2)"
            strokeWidth="6"
          />
          {/* Ticks */}
          {Array.from({ length: 24 }).map((_, i) => {
            const angle = (i / 24) * 2 * Math.PI - Math.PI / 2;
            const x1 = size / 2 + (radius - 4) * Math.cos(angle);
            const y1 = size / 2 + (radius - 4) * Math.sin(angle);
            const x2 = size / 2 + (radius + 1) * Math.cos(angle);
            const y2 = size / 2 + (radius + 1) * Math.sin(angle);
            return (
              <line
                key={i}
                x1={x1} y1={y1} x2={x2} y2={y2}
                stroke="var(--color-text-muted)"
                strokeWidth="0.5"
                opacity="0.5"
              />
            );
          })}
          {/* Value arc */}
          <circle
            ref={circleRef}
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth="6"
            strokeDasharray={circumference}
            strokeLinecap="round"
            style={{ filter: `drop-shadow(0 0 6px ${color})` }}
          />
        </svg>
        {/* Center readout */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span
            className="font-mono text-2xl font-600"
            style={{ color }}
          >
            {score.toFixed(2)}
          </span>
        </div>
      </div>
      <span
        className="font-mono text-[10px] font-500 tracking-[0.2em]"
        style={{ color }}
      >
        {label}
      </span>
    </div>
  );
}
