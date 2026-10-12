import { goalDist, pct, xgColor } from '../utils';

const W = 340, H = 170, P = { l: 36, r: 10, t: 14, b: 24 };
const iw = W - P.l - P.r, ih = H - P.t - P.b;

function Axes({ max, fmt }) {
  return [0, 0.5, 1].map((t) => {
    const y = P.t + ih * (1 - t);
    return (
      <g key={t}>
        <line x1={P.l} x2={W - P.r} y1={y} y2={y} className="grid" />
        <text x={P.l - 5} y={y + 3} className="tick" textAnchor="end">{fmt(max * t)}</text>
      </g>
    );
  });
}

export function Cumulative({ xs }) {
  const cum = xs.reduce((a, p) => [...a, (a.length ? a[a.length - 1] : 0) + p], []);
  const max = Math.max(0.5, cum[cum.length - 1] || 0) * 1.1;
  const px = (i) => P.l + (xs.length < 2 ? iw / 2 : (i / (xs.length - 1)) * iw);
  const py = (v) => P.t + ih * (1 - v / max);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart">
      <Axes max={max} fmt={(v) => v.toFixed(2)} />
      <polyline points={cum.map((v, i) => `${px(i)},${py(v)}`).join(' ')} fill="none" stroke="#38bdf8" strokeWidth="2" />
      {cum.map((v, i) => (
        <circle key={i} cx={px(i)} cy={py(v)} r="3.5" fill={xgColor(xs[i])} stroke="#0b1220">
          <title>{`Tiro ${i + 1}: xG acumulado ${v.toFixed(2)}`}</title>
        </circle>
      ))}
      <text x={P.l} y={H - 6} className="tick">Tiro 1</text>
      <text x={W - P.r} y={H - 6} className="tick" textAnchor="end">Tiro {xs.length}</text>
    </svg>
  );
}

function Bars({ vals, labels, colors, fmt, max }) {
  const m = max ?? Math.max(...vals, 0.01) * 1.2;
  const bw = iw / vals.length;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart">
      <Axes max={m} fmt={fmt} />
      {vals.map((v, i) => {
        const h = (ih * v) / m, x = P.l + i * bw + bw * 0.15;
        return (
          <g key={i}>
            <rect x={x} y={P.t + ih - h} width={bw * 0.7} height={h} rx="2" fill={colors[i]} />
            {vals.length <= 8 && <text x={x + bw * 0.35} y={P.t + ih - h - 3} className="tick" textAnchor="middle">{fmt(v)}</text>}
            <text x={x + bw * 0.35} y={H - 8} className="tick" textAnchor="middle">{labels[i]}</text>
          </g>
        );
      })}
    </svg>
  );
}

export function ShotBars({ xs }) {
  const start = Math.max(0, xs.length - 12);
  const v = xs.slice(start);
  return <Bars vals={v} labels={v.map((_, i) => start + i + 1)} colors={v.map(xgColor)} fmt={(n) => n.toFixed(2)} />;
}

export function GoalDist({ xs }) {
  const d = goalDist(xs).slice(0, 7);
  return <Bars vals={d} labels={d.map((_, k) => k)} colors={d.map(() => '#a78bfa')} fmt={(n) => pct(n, 0)} max={Math.max(...d) * 1.2} />;
}
