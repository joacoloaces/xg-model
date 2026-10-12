import { BASE, PRESETS, SITUATION, defenders } from '../utils';

const BODY = [['Right Foot', 'Pie der.'], ['Left Foot', 'Pie izq.'], ['Head', 'Cabeza']];
const TECH = ['Normal', 'Volley', 'Half Volley', 'Lob', 'Backheel', 'Diving Header', 'Overhead Kick'];
const PATTERNS = Object.keys(SITUATION);

const Seg = ({ value, options, onChange }) => (
  <div className="seg">
    {options.map(([v, label]) => (
      <button key={v} type="button" className={value === v ? 'on' : ''} onClick={() => onChange(v)}>{label}</button>
    ))}
  </div>
);

const Toggle = ({ label, checked, onChange }) => (
  <label className="toggle">
    <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
    <span>{label}</span>
  </label>
);

const Slider = ({ label, value, min, max, onChange }) => (
  <label className="slider">
    <span>{label} <b>{value}</b></span>
    <input type="range" min={min} max={max} value={value} onChange={(e) => onChange(+e.target.value)} />
  </label>
);

export default function Controls({ params: p, setParams, options, manual, nPlayers, hasGk }) {
  const set = (k, v) => setParams({ ...p, [k]: v });
  const tech = options?.shot_technique ?? TECH;
  const patterns = options?.play_pattern ?? PATTERNS;

  return (
    <div className="controls">
      <h3>Situación</h3>
      <div className="chips">
        {PRESETS.map((s) => (
          <button key={s.label} type="button" onClick={() => setParams({ ...BASE, ...s.p })}>{s.label}</button>
        ))}
      </div>

      <h3>Remate</h3>
      <Seg value={p.shot_body_part} options={BODY} onChange={(v) => set('shot_body_part', v)} />
      <select value={p.shot_technique} onChange={(e) => set('shot_technique', e.target.value)}>
        {tech.map((t) => <option key={t}>{t}</option>)}
      </select>

      <h3>Defensa en la línea de tiro</h3>
      {manual ? (
        <p className="hint">
          Usando los {nPlayers} jugadores colocados en el campo (se calculan igual que en el entrenamiento).
          {!hasGk && ' Falta el portero: añádelo para una estimación fiable.'}
        </p>
      ) : (
        <>
          <Seg
            value={p.n_defenders_in_cone}
            options={[0, 1, 2, 3, 4, 5].map((n) => [n, String(n)])}
            onChange={(n) => setParams({ ...p, ...defenders(n) })}
          />
          <Toggle label="Portero en la línea de tiro" checked={p.gk_in_cone} onChange={(v) => set('gk_in_cone', v)} />
        </>
      )}
      <Toggle label="Tirador presionado" checked={p.under_pressure} onChange={(v) => set('under_pressure', v)} />

      <h3>Jugada</h3>
      <Toggle label="Mano a mano con el portero" checked={p.shot_one_on_one} onChange={(v) => set('shot_one_on_one', v)} />
      <Toggle label="Portería vacía" checked={p.shot_open_goal} onChange={(v) => set('shot_open_goal', v)} />
      <Toggle label="Remate de primeras" checked={p.shot_first_time} onChange={(v) => set('shot_first_time', v)} />

      <h3>Contexto</h3>
      <Slider label="Minuto" value={p.minute} min={0} max={120} onChange={(v) => set('minute', v)} />
      <Slider label="Diferencia de goles" value={p.score_diff} min={-3} max={3} onChange={(v) => set('score_diff', v)} />

      <details>
        <summary>Más opciones</summary>
        <select value={p.play_pattern} onChange={(e) => set('play_pattern', e.target.value)}>
          {patterns.map((t) => <option key={t} value={t}>{SITUATION[t] ?? t}</option>)}
        </select>
        <Toggle label="Viene de un pase (pase clave)" checked={p.has_key_pass} onChange={(v) => set('has_key_pass', v)} />
      </details>
    </div>
  );
}