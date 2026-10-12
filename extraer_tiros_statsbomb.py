import csv
import json
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
OUT = Path("comparacion_xg_statsbomb.csv")

def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "xG-comparison/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))

def api_predict(payload):
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "http://127.0.0.1:8000/predict",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode("utf-8")).get("xg")
    except Exception:
        return ""

print("Descargando el catálogo de competiciones...")
competitions = get_json(f"{BASE}/competitions.json")

# Prioriza La Liga española; si no aparece, usa la primera competición disponible.
comp = next(
    (c for c in competitions if "la liga" in c["competition_name"].lower()),
    competitions[0]
)
comp_id = comp["competition_id"]
season_id = comp["season_id"]

print(f"Competición: {comp['competition_name']} | Temporada: {comp['season_name']}")
matches = get_json(f"{BASE}/matches/{comp_id}/{season_id}.json")
matches = sorted(matches, key=lambda m: m.get("match_date", ""), reverse=True)

shots = []
match_info = None

# Busca un partido con tiros disponibles.
for match in matches[:20]:
    match_id = match["match_id"]
    try:
        events = get_json(f"{BASE}/events/{match_id}.json")
    except Exception as e:
        print(f"No se pudo descargar el partido {match_id}: {e}")
        continue

    current = []
    for e in events:
        s = e.get("shot")
        loc = e.get("location")
        if e.get("type", {}).get("name") != "Shot" or not s or not loc:
            continue
        if s.get("statsbomb_xg") is None:
            continue

        body = s.get("body_part", {}).get("name", "Right Foot")
        technique = s.get("technique", {}).get("name", "Normal")
        shot_type = s.get("type", {}).get("name", "Open Play")
        play_pattern = e.get("play_pattern", {}).get("name", "Regular Play")

        payload = {
            "x": loc[0],
            "y": loc[1],
            "minute": e.get("minute", 45),
            "period": e.get("period", 1),
            "under_pressure": e.get("under_pressure", False),
            "shot_first_time": s.get("first_time", False),
            "shot_one_on_one": s.get("one_on_one", False),
            "shot_open_goal": s.get("open_goal", False),
            "shot_redirect": s.get("redirect", False),
            "shot_follows_dribble": s.get("follows_dribble", False),
            "shot_aerial_won": s.get("aerial_won", False),
            "play_pattern": play_pattern,
            "shot_body_part": body,
            "shot_technique": technique,
            "shot_type": shot_type,
        }

        current.append({
            "fecha": match.get("match_date", ""),
            "partido": f"{match.get('home_team', {}).get('home_team_name', '')} - "
                       f"{match.get('away_team', {}).get('away_team_name', '')}",
            "equipo": e.get("team", {}).get("name", ""),
            "jugador": e.get("player", {}).get("name", ""),
            "minuto": e.get("minute", ""),
            "x": loc[0],
            "y": loc[1],
            "xg_statsbomb": s["statsbomb_xg"],
            "resultado": s.get("outcome", {}).get("name", ""),
            "parte_cuerpo": body,
            "tecnica": technique,
            "tipo_tiro": shot_type,
            "patron_juego": play_pattern,
            "bajo_presion": e.get("under_pressure", False),
            "xg_modelo": api_predict(payload),
        })

    if current:
        shots = current
        match_info = match
        break

if not shots:
    raise SystemExit("No se encontraron tiros. Revisa la conexión o prueba otra temporada.")

with OUT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=shots[0].keys())
    writer.writeheader()
    writer.writerows(shots)

print(f"\nPartido: {shots[0]['partido']}")
print(f"Tiros extraídos: {len(shots)}")
predicted = sum(1 for s in shots if s["xg_modelo"] != "")
print(f"Tiros con predicción de tu API: {predicted}/{len(shots)}")
print(f"Archivo creado: {OUT.resolve()}")
print("\nMuestra:")
for s in shots[:8]:
    print(f"{s['jugador'][:22]:22} x={s['x']:5.1f} y={s['y']:5.1f} "
          f"StatsBomb={s['xg_statsbomb']:.3f} modelo={s['xg_modelo']}")
