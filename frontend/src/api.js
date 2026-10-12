// URL del backend:
//  - desarrollo (npm run dev): http://localhost:8000
//  - producción: la que pongas en VITE_API_URL al construir (p.ej. https://api.tudominio.com).
//    Si no se define, usa la misma origen (útil si un proxy inverso sirve /predict, /health...).
const API_URL = (
  import.meta.env.VITE_API_URL ?? (import.meta.env.DEV ? 'http://localhost:8000' : '')
).replace(/\/$/, '');

async function parseError(res) {
  const body = await res.json().catch(() => null);
  // 422 de FastAPI: detail es una lista de errores de validación
  if (Array.isArray(body?.detail)) {
    return body.detail.map((d) => d.msg).join('; ');
  }
  return body?.detail || `Error ${res.status}`;
}

export async function predictXg(shotData) {
  const res = await fetch(`${API_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(shotData),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function getFeatureOptions() {
  const res = await fetch(`${API_URL}/feature-options`);
  if (!res.ok) throw new Error('No se pudieron cargar las opciones');
  return res.json();
}

export async function checkHealth() {
  try {
    const res = await fetch(`${API_URL}/health`);
    return res.ok;
  } catch {
    return false;
  }
}
