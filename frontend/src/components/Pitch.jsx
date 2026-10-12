import { useRef, useState } from 'react';
import { geo, pct, xgColor } from '../utils';

const LINE = { stroke: 'rgba(255,255,255,.75)', strokeWidth: 0.35, fill: 'none' };
const KIND = { def: ['#ef4444', 'D'], team: ['#38bdf8', 'C'], gk: ['#facc15', 'P'] };
const kindOf = (p) => (p.gk ? 'gk' : p.teammate ? 'team' : 'def');
const HINT = {
  shot: 'Clic para simular un tiro desde ese punto',
  def: 'Clic para colocar un defensa · arrastra para moverlo · clic derecho para quitarlo',
  gk: 'Clic para colocar el portero (solo uno) · arrastra para moverlo',
  team: 'Clic para colocar un compañero (puede tapar el tiro) · arrastra para moverlo',
};

export default function Pitch({ shots, onShot, full, loading, players, setPlayers, mode, onAddPlayer }) {
  const ref = useRef(null);
  const skip = useRef(false);
  const [hover, setHover] = useState(null);
  const [drag, setDrag] = useState(null);
  const x0 = full ? 0 : 58;

  const toPoint = (e) => {
    const svg = ref.current;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const p = pt.matrixTransform(svg.getScreenCTM().inverse());
    return { x: Math.min(120, Math.max(x0, p.x)), y: Math.min(80, Math.max(0, p.y)) };
  };
  const onMove = (e) => {
    const p = toPoint(e);
    setHover(p);
    if (drag != null) setPlayers((ps) => ps.map((q) => (q.id === drag ? { ...q, x: p.x, y: p.y } : q)));
  };
  const onUp = () => { if (drag != null) { skip.current = true; setDrag(null); } };
  const onClick = (e) => {
    if (skip.current) { skip.current = false; return; }
    const { x, y } = toPoint(e);
    if (mode === 'shot') onShot(x, y); else onAddPlayer(x, y);
  };
  const h = hover && geo(hover.x, hover.y);
  const showCone = hover && mode === 'shot' && drag == null;

  return (
    <div className="pitch-wrap">
      <svg
        ref={ref}
        viewBox={`${x0 - 2} -3 ${126 - x0} 86`}
        className={`pitch ${loading ? 'busy' : ''}`}
        onMouseMove={onMove}
        onMouseUp={onUp}
        onMouseLeave={() => { setHover(null); setDrag(null); }}
        onClick={onClick}
      >
        <rect x={x0 - 2} y={-3} width={126 - x0} height={86} fill="#0f3d2e" />
        <rect x={x0} y={0} width={120 - x0} height={80} fill="#14573f" />
        <g {...LINE}>
          <rect x={x0} y={0} width={120 - x0} height={80} />
          {full && <><line x1={60} y1={0} x2={60} y2={80} /><circle cx={60} cy={40} r={10} /></>}
          {!full && <line x1={60} y1={0} x2={60} y2={80} strokeDasharray="1 1" opacity={0.4} />}
          <rect x={102} y={18} width={18} height={44} />
          <rect x={114} y={30} width={6} height={20} />
          <path d="M102 32 A10 10 0 0 0 102 48" />
          <circle cx={108} cy={40} r={0.5} fill="white" />
          {full && <><rect x={0} y={18} width={18} height={44} /><rect x={0} y={30} width={6} height={20} /></>}
        </g>
        <rect x={120} y={36} width={2} height={8} fill="#e2e8f0" opacity={0.9} />

        {showCone && (
          <g pointerEvents="none">
            <polygon points={`${hover.x},${hover.y} 120,36 120,44`} fill="rgba(250,204,21,.28)" stroke="#facc15" strokeWidth={0.25} />
            <circle cx={hover.x} cy={hover.y} r={1.1} fill="none" stroke="#facc15" strokeWidth={0.4} />
          </g>
        )}

        {shots.map((s, i) => (
          <g key={s.id}>
            <line x1={s.x} y1={s.y} x2={120} y2={40} stroke={xgColor(s.xg)} strokeWidth={0.25} opacity={0.45} strokeDasharray="1 0.8" />
            <circle cx={s.x} cy={s.y} r={1.3 + s.xg * 4.5} fill={xgColor(s.xg)} fillOpacity={0.85} stroke="#0b1220" strokeWidth={0.35}>
              <title>{`#${i + 1} · xG ${s.xg.toFixed(3)} · ${s.dist.toFixed(1)} yd · ${s.angle.toFixed(0)}°${s.free != null ? ` · portería libre ${pct(s.free, 0)}` : ''}`}</title>
            </circle>
            <text x={s.x} y={s.y + 0.9} fontSize={2.4} textAnchor="middle" fill="#0b1220" fontWeight="700" pointerEvents="none">{i + 1}</text>
          </g>
        ))}

        {players.map((p) => {
          const [color, label] = KIND[kindOf(p)];
          return (
            <g
              key={p.id}
              className="player"
              onMouseDown={(e) => { e.stopPropagation(); setDrag(p.id); }}
              onClick={(e) => { e.stopPropagation(); skip.current = false; }}
              onContextMenu={(e) => { e.preventDefault(); e.stopPropagation(); setPlayers((ps) => ps.filter((q) => q.id !== p.id)); }}
            >
              <circle cx={p.x} cy={p.y} r={1.8} fill={color} stroke="#fff" strokeWidth={0.35} />
              <text x={p.x} y={p.y + 0.9} fontSize={2.3} textAnchor="middle" fill="#0b1220" fontWeight="700" pointerEvents="none">{label}</text>
            </g>
          );
        })}
      </svg>
      <div className="pitch-info">
        {mode === 'shot' && h
          ? <>Distancia <b>{h.dist.toFixed(1)} yd</b> · Ángulo de tiro <b>{h.angle.toFixed(1)}°</b> · {HINT.shot}</>
          : HINT[mode]}
      </div>
    </div>
  );
}