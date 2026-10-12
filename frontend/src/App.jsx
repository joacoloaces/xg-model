import { useCallback, useEffect, useMemo, useState } from 'react';
import { checkHealth, getFeatureOptions, predictXg } from './api';
import Pitch from './components/Pitch';
import Controls from './components/Controls';
import { Cumulative, GoalDist, ShotBars } from './components/Charts';
import { BASE, SITUATION, geo, pAtLeastOne, pct, toCsv, xgColor, zone } from './utils';

let nextId = 1;
let pid = 1;
const MODES = [['shot', 'Tiro'], ['def', 'Defensa'], ['gk', 'Portero'], ['team', 'Compañero']];
const TABS = [['charts', 'Gráficos'], ['situation', 'Situaciones'], ['log', 'Registro']];

const Kpi = ({ label, value, title }) => (
  <div className="kpi" title={title}><span>{label}</span><b>{value}</b></div>
);
const Box = ({ title, children }) => (
  <div className="chartbox"><h3>{title}</h3>{children}</div>
);

export default function App() {
  const [shots, setShots] = useState([]);
  const [params, setParams] = useState(BASE);
  const [options, setOptions] = useState(null);
  const [online, setOnline] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [full, setFull] = useState(false);
  const [tab, setTab] = useState('charts');
  const [players, setPlayers] = useState([]);
  const [mode, setMode] = useState('shot');

  useEffect(() => {
    let alive = true;
    const ping = async () => { const ok = await checkHealth(); if (alive) setOnline(ok); };
    ping();
    const t = setInterval(ping, 15000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  useEffect(() => {
    if (online && !options) getFeatureOptions().then(setOptions).catch(() => { });
  }, [online, options]);

  const addShot = useCallback(async (x, y) => {
    setLoading(true);
    setError('');
    try {
      const payload = { ...params, x: +x.toFixed(1), y: +y.toFixed(1) };
      if (players.length) {
        payload.players = players.map((q) => ({ x: +q.x.toFixed(1), y: +q.y.toFixed(1), teammate: q.teammate, gk: q.gk }));
      }
      const r = await predictXg(payload);
      const free = r.features_used?.freeze?.pct_libre_total ?? null;
      setShots((s) => [...s, { id: nextId++, x, y, xg: r.xg, free, ...geo(x, y), zone: zone(x, y), p: params }]);
    } catch (e) {
      setError(e.message || 'No se pudo conectar con la API');
    } finally {
      setLoading(false);
    }
  }, [params, players]);

  const addPlayer = (x, y) => setPlayers((ps) => {
    const q = { id: pid++, x, y, teammate: mode === 'team', gk: mode === 'gk' };
    return mode === 'gk' ? [...ps.filter((o) => !o.gk), q] : [...ps, q];
  });

  const xs = useMemo(() => shots.map((s) => s.xg), [shots]);
  const total = xs.reduce((a, b) => a + b, 0);
  const best = shots.reduce((m, s) => (s.xg > (m?.xg ?? -1) ? s : m), null);
  const bySituation = useMemo(() => {
    const g = {};
    shots.forEach((s) => {
      const k = SITUATION[s.p.play_pattern] ?? s.p.play_pattern;
      g[k] = g[k] || { n: 0, xg: 0 };
      g[k].n += 1;
      g[k].xg += s.xg;
    });
    return Object.entries(g).sort((a, b) => b[1].xg - a[1].xg);
  }, [shots]);

  const exportCsv = () => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([toCsv(shots)], { type: 'text/csv' }));
    a.download = 'simulacion_xg.csv';
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <div className="app">
      <header>
        <div className="brand"><h1>xG Lab</h1><p>Simulador de ocasiones de gol</p></div>
        <div className="kpis">
          <Kpi label="Tiros" value={shots.length} />
          <Kpi label="xG total" value={total.toFixed(2)} />
          <Kpi label="xG medio" value={shots.length ? (total / shots.length).toFixed(3) : '–'} />
          <Kpi label="P(≥ 1 gol)" value={shots.length ? pct(pAtLeastOne(xs), 0) : '–'} title="Probabilidad de marcar al menos un gol" />
          <Kpi label="Mejor ocasión" value={best ? best.xg.toFixed(3) : '–'} title={best ? `${best.dist.toFixed(1)} yd · ${best.zone}` : ''} />
        </div>
        <div className="actions">
          <span className={`status ${online ? 'ok' : online === false ? 'off' : ''}`}>
            {online ? 'API online' : online === false ? 'Sin conexión' : 'Conectando…'}
          </span>
          <button onClick={() => setFull((f) => !f)}>{full ? 'Medio campo' : 'Campo completo'}</button>
          <button disabled={!shots.length} onClick={() => setShots((s) => s.slice(0, -1))}>Deshacer</button>
          <button disabled={!shots.length} onClick={() => setShots([])}>Limpiar</button>
          <button className="primary" disabled={!shots.length} onClick={exportCsv}>CSV</button>
        </div>
      </header>

      {error && <div className="error" onClick={() => setError('')}>{error}</div>}

      <div className="main">
        <section className="card side">
          <h2>Parámetros</h2>
          <Controls params={params} setParams={setParams} options={options}
            manual={players.length > 0} nPlayers={players.length} hasGk={players.some((q) => q.gk)} />
        </section>

        <section className="card center">
          <h2>Campo</h2>
          <div className="tools">
            {MODES.map(([k, label]) => (
              <button key={k} className={mode === k ? 'on' : ''} onClick={() => setMode(k)}>{label}</button>
            ))}
            <button disabled={!players.length} onClick={() => setPlayers([])}>Quitar jugadores</button>
          </div>
          <Pitch shots={shots} onShot={addShot} full={full} loading={loading}
            players={players} setPlayers={setPlayers} mode={mode} onAddPlayer={addPlayer} />
        </section>

        <section className="card right">
          <div className="tabs">
            {TABS.map(([k, label]) => (
              <button key={k} className={tab === k ? 'on' : ''} onClick={() => setTab(k)}>{label}</button>
            ))}
          </div>
          {!shots.length ? (
            <p className="empty">Simula tiros en el campo para ver el xG acumulado, la probabilidad de marcar y los gráficos por situación.</p>
          ) : tab === 'charts' ? (
            <div className="tabbody stack">
              <Box title="xG acumulado"><Cumulative xs={xs} /></Box>
              <Box title="xG por tiro (últimos 12)"><ShotBars xs={xs} /></Box>
              <Box title="Distribución de goles esperados"><GoalDist xs={xs} /></Box>
            </div>
          ) : tab === 'situation' ? (
            <div className="tabbody scroll">
              <table>
                <thead><tr><th>Situación</th><th>Tiros</th><th>xG</th><th>xG/tiro</th></tr></thead>
                <tbody>{bySituation.map(([k, v]) => (
                  <tr key={k}><td>{k}</td><td>{v.n}</td><td>{v.xg.toFixed(2)}</td><td>{(v.xg / v.n).toFixed(3)}</td></tr>
                ))}</tbody>
              </table>
            </div>
          ) : (
            <div className="tabbody scroll">
              <table>
                <thead><tr><th>#</th><th>Zona</th><th>Dist.</th><th>Áng.</th><th>xG</th></tr></thead>
                <tbody>{shots.map((s, i) => (
                  <tr key={s.id}>
                    <td>{i + 1}</td><td>{s.zone}</td><td>{s.dist.toFixed(1)} yd</td><td>{s.angle.toFixed(0)}°</td>
                    <td><span className="dot" style={{ background: xgColor(s.xg) }} />{s.xg.toFixed(3)}</td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          )}
        </section>
      </div>

      <footer>
        Modelo entrenado con datos abiertos de StatsBomb · Simulador: el contexto y los defensas se aproximan, el xG es una estimación.
      </footer>
    </div>
  );
}