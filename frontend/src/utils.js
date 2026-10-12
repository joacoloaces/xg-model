// Utilidades del simulador xG (coordenadas StatsBomb: campo 120x80, portería en x=120)

export const defenders = (n) => ({
  n_defenders_in_cone: n,
  n_defenders_nearby: Math.min(n + 1, 8),
  n_defenders_ahead: Math.min(n + 1, 11),
  n_defenders_in_box: Math.min(n + 1, 11),
  n_defenders_pressure: Math.min(n, 2),
});

export const BASE = {
  minute: 45, period: 1, score_diff: 0,
  under_pressure: false, shot_first_time: false, shot_one_on_one: false,
  shot_open_goal: false, shot_aerial_won: false,
  shot_body_part: 'Right Foot', shot_technique: 'Normal', shot_type: 'Open Play',
  play_pattern: 'Regular Play', gk_in_cone: true,
  has_key_pass: true, kp_cross: false, kp_through_ball: false, kp_cut_back: false,
  kp_height: 'Ground Pass', kp_technique: 'None', kp_type: 'None',
  ...defenders(2),
};

export const PRESETS = [
  { label: 'Contraataque 1v1', p: { play_pattern: 'From Counter', shot_one_on_one: true, kp_through_ball: true, kp_technique: 'Through Ball', ...defenders(0) } },
  { label: 'Remate de cabeza', p: { shot_body_part: 'Head', shot_aerial_won: true, kp_cross: true, kp_height: 'High Pass', ...defenders(2) } },
  { label: 'Córner', p: { play_pattern: 'From Corner', shot_body_part: 'Head', kp_type: 'Corner', kp_height: 'High Pass', ...defenders(3), n_defenders_in_box: 6 } },
  { label: 'Pase atrás', p: { shot_first_time: true, kp_cut_back: true, ...defenders(1) } },
  { label: 'Disparo lejano', p: { ...defenders(1) } },
  { label: 'Tiro libre', p: { play_pattern: 'From Free Kick', shot_type: 'Free Kick', has_key_pass: false, ...defenders(4) } },
];

export const SITUATION = {
  'Regular Play': 'Juego abierto', 'From Corner': 'Córner', 'From Free Kick': 'Falta',
  'From Throw In': 'Saque de banda', 'From Counter': 'Contraataque', 'From Goal Kick': 'Saque de puerta',
  'From Keeper': 'Saque del portero', 'From Kick Off': 'Saque inicial', Other: 'Otra',
};

export const xgColor = (xg) =>
  xg < 0.05 ? '#5eead4' : xg < 0.15 ? '#fde047' : xg < 0.35 ? '#fb923c' : xg < 0.6 ? '#f87171' : '#ef4444';

export const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;

export function geo(x, y) {
  const dx = 120 - x;
  const angle = Math.abs(Math.atan2(44 - y, dx) - Math.atan2(36 - y, dx));
  return { dist: Math.hypot(dx, 40 - y), angle: (angle * 180) / Math.PI };
}

export function zone(x, y) {
  if (x >= 114 && y >= 30 && y <= 50) return 'Área chica';
  if (x >= 102 && y >= 18 && y <= 62) return 'Área grande';
  return x >= 84 ? 'Zona de remate' : 'Larga distancia';
}

/** Probabilidad de marcar al menos un gol con estos tiros. */
export const pAtLeastOne = (xs) => 1 - xs.reduce((a, p) => a * (1 - p), 1);

/** Distribución del nº de goles (binomial de Poisson): d[k] = P(k goles). */
export function goalDist(xs) {
  let d = [1];
  for (const p of xs) {
    const n = Array(d.length + 1).fill(0);
    d.forEach((v, k) => { n[k] += v * (1 - p); n[k + 1] += v * p; });
    d = n;
  }
  return d;
}

export function toCsv(shots) {
  const head = 'n,x,y,distancia_yd,angulo_grados,zona,situacion,parte_cuerpo,minuto,xg';
  const rows = shots.map((s, i) =>
    [i + 1, s.x.toFixed(1), s.y.toFixed(1), s.dist.toFixed(1), s.angle.toFixed(1), s.zone,
    SITUATION[s.p.play_pattern] ?? s.p.play_pattern, s.p.shot_body_part, s.p.minute, s.xg].join(','));
  return [head, ...rows].join('\n');
}