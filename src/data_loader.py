"""Descarga de tiros desde StatsBomb Open Data para múltiples competiciones/temporadas.

Cada partido se cachea en data/raw/shots_v2/<match_id>.parquet, así las ejecuciones
siguientes solo descargan partidos nuevos. Además de los atributos del tiro, se calcula
el CONTEXTO PREVIO al tiro a partir de los eventos del partido (marcador, pase clave,
secuencia de la posesión, acción anterior). Todo el contexto usa solo información anterior
al tiro, así que no hay fuga de información.
"""
import json
import logging
import math
import os
import re
import shutil
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pyarrow.parquet as pq
from statsbombpy import sb
from tqdm import tqdm

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# Si cambias la lógica de _context_features, sube la versión: se descartará la caché antigua.
CACHE_DIR_NAME = "shots_v2"

# Eventos que no cuentan como "acción previa" al tiro.
IGNORED_FOR_PREV = {
    "Ball Receipt*", "Camera On", "Camera off", "Tactical Shift", "Substitution",
    "Player On", "Player Off", "Injury Stoppage", "Referee Ball-Drop", "Starting XI",
}


def _season_year(name):
    m = re.match(r"(\d{4})", str(name))
    return int(m.group(1)) if m else None


def select_competitions(data_cfg):
    comps = sb.competitions()
    if data_cfg.get("gender"):
        comps = comps[comps["competition_gender"] == data_cfg["gender"]]
    if not data_cfg.get("include_youth", False):
        comps = comps[~comps["competition_youth"].astype(bool)]
    if data_cfg.get("competitions"):
        comps = comps[comps["competition_name"].isin(data_cfg["competitions"])]
    if data_cfg.get("exclude_competitions"):
        comps = comps[~comps["competition_name"].isin(data_cfg["exclude_competitions"])]
    if data_cfg.get("min_season"):
        years = comps["season_name"].map(_season_year)
        comps = comps[years >= int(data_cfg["min_season"])]
    cols = ["competition_id", "season_id", "competition_name", "season_name",
            "competition_gender", "country_name"]
    comps = comps[cols].drop_duplicates().reset_index(drop=True)
    log.info("Competiciones/temporadas seleccionadas: %d", len(comps))
    return comps


def fetch_matches(comps):
    frames = []
    for row in tqdm(comps.itertuples(index=False), total=len(comps), desc="Partidos"):
        try:
            m = sb.matches(competition_id=row.competition_id, season_id=row.season_id)
        except Exception as e:
            log.warning("Sin partidos para %s %s: %s", row.competition_name, row.season_name, e)
            continue
        m = m[["match_id", "match_date"]].copy()
        for col in comps.columns:
            m[col] = getattr(row, col)
        frames.append(m)
    matches = pd.concat(frames, ignore_index=True).drop_duplicates("match_id")
    log.info("Partidos encontrados: %d", len(matches))
    return matches


# ---------------------------------------------------------------------------
# Contexto previo al tiro
# ---------------------------------------------------------------------------
def _v(r, key):
    """Valor de una columna del evento o None si falta / es NaN."""
    v = r.get(key)
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _loc(r, key="location"):
    v = _v(r, key)
    return list(v) if isinstance(v, (list, tuple)) else None


def _clock(r):
    return int(_v(r, "minute") or 0) * 60 + int(_v(r, "second") or 0)


def _dist(a, b):
    if not a or not b:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _key_pass_features(p):
    """Características del pase que asistió al tiro (None si no hay pase clave)."""
    out = {
        "has_key_pass": p is not None, "kp_length": None, "kp_angle": None, "kp_height": None,
        "kp_type": None, "kp_technique": None, "kp_cross": False, "kp_through_ball": False,
        "kp_cut_back": False, "kp_switch": False, "kp_start_x": None, "kp_start_y": None,
    }
    if p is None:
        return out
    loc = _loc(p) or [None, None]
    out.update({
        "kp_length": _v(p, "pass_length"), "kp_angle": _v(p, "pass_angle"),
        "kp_height": _v(p, "pass_height"), "kp_type": _v(p, "pass_type"),
        "kp_technique": _v(p, "pass_technique"),
        "kp_cross": bool(_v(p, "pass_cross")), "kp_through_ball": bool(_v(p, "pass_through_ball")),
        "kp_cut_back": bool(_v(p, "pass_cut_back")), "kp_switch": bool(_v(p, "pass_switch")),
        "kp_start_x": loc[0], "kp_start_y": loc[1],
    })
    return out


def _context_features(events):
    """Una fila por tiro (clave `id`) con variables de contexto anteriores al tiro."""
    recs = events.sort_values("index").to_dict("records")
    by_id = {r["id"]: r for r in recs}
    team_ids = {r["team_id"] for r in recs if _v(r, "team_id") is not None}

    score = defaultdict(int)
    last_shot_clock = {}
    poss = {}
    prev = None
    rows = []

    for r in recs:
        etype = r["type"]
        team = _v(r, "team_id")
        clock = _clock(r)

        # 1) tiro: usar el estado ANTERIOR al tiro
        if etype == "Shot" and _loc(r):
            loc = _loc(r)
            opp = next((t for t in team_ids if t != team), None)
            ps = poss.get(_v(r, "possession"))
            start_x = ps["start_x"] if ps and ps["start_x"] is not None else loc[0]

            carry_len = None
            if (prev is not None and prev["type"] == "Carry"
                    and _v(prev, "player_id") == _v(r, "player_id")):
                carry_len = _dist(_loc(prev), _loc(prev, "carry_end_location"))

            kp_id = _v(r, "shot_key_pass_id")
            rows.append({
                "id": r["id"],
                "score_diff": score[team] - (score[opp] if opp is not None else 0),
                "prev_type": prev["type"] if prev is not None else None,
                "prev_same_team": bool(prev is not None and _v(prev, "team_id") == team),
                "secs_since_prev_event": (clock - _clock(prev)) if prev is not None else None,
                "poss_elapsed_s": clock - ps["start_clock"] if ps else 0,
                "poss_n_passes": ps["n_passes"] if ps else 0,
                "poss_progress_x": loc[0] - start_x,
                "secs_since_team_last_shot": (clock - last_shot_clock[team]) if team in last_shot_clock else None,
                "carry_length": carry_len,
                **_key_pass_features(by_id.get(kp_id) if kp_id else None),
            })
            last_shot_clock[team] = clock

        # 2) actualizar estado DESPUÉS de procesar el evento
        if etype == "Shot" and _v(r, "shot_outcome") == "Goal" and (_v(r, "period") or 0) <= 4:
            score[team] += 1
        elif etype == "Own Goal For":
            score[team] += 1

        pn = _v(r, "possession")
        if etype not in IGNORED_FOR_PREV and pn is not None:
            ps = poss.get(pn)
            if ps is None:
                ps = poss[pn] = {"start_clock": clock, "start_x": None, "n_passes": 0}
            if team == _v(r, "possession_team_id"):
                if etype == "Pass":
                    ps["n_passes"] += 1
                if ps["start_x"] is None and _loc(r):
                    ps["start_x"] = _loc(r)[0]
        if etype not in IGNORED_FOR_PREV:
            prev = r

    return pd.DataFrame(rows)


def _serialize(df):
    for col in df.columns:
        if df[col].dtype == object and df[col].map(lambda v: isinstance(v, (list, dict))).any():
            df[col] = df[col].map(lambda v: json.dumps(v) if isinstance(v, (list, dict)) else None)
    return df


def _is_valid_parquet(path):
    """True si el archivo existe y su pie de parquet se lee (detecta escrituras truncadas)."""
    try:
        pq.read_metadata(path)
        return True
    except Exception:
        return False


def _download_match(match_id, out_path):
    events = sb.events(match_id=match_id)
    shots = events[events["type"] == "Shot"].copy()
    shots["match_id"] = match_id
    ctx = _context_features(events)
    if len(ctx):
        shots = shots.merge(ctx, on="id", how="left")
    shots = _serialize(shots.reset_index(drop=True))
    # Escritura atómica: si el disco se llena o el proceso se corta, nunca queda un
    # parquet a medias con el nombre definitivo.
    tmp = out_path.with_name(out_path.name + ".tmp")
    try:
        shots.to_parquet(tmp, index=False)
        os.replace(tmp, out_path)
    finally:
        tmp.unlink(missing_ok=True)
    return len(shots)


def download_shots(cfg):
    data_cfg = cfg["data"]
    shots_dir = cfg["paths"]["raw_dir"] / CACHE_DIR_NAME
    shots_dir.mkdir(parents=True, exist_ok=True)

    matches = fetch_matches(select_competitions(data_cfg))
    if data_cfg.get("max_matches"):
        matches = matches.sample(n=min(len(matches), data_cfg["max_matches"]), random_state=42)
    matches.to_parquet(cfg["paths"]["interim_dir"] / "matches.parquet", index=False)

    for leftover in shots_dir.glob("*.tmp"):   # restos de descargas interrumpidas
        leftover.unlink(missing_ok=True)
    free_gb = shutil.disk_usage(shots_dir).free / 1e9
    if free_gb < 2:
        raise RuntimeError(f"Solo quedan {free_gb:.1f} GB libres en el disco; libera espacio antes de descargar")

    refresh = data_cfg.get("refresh", False)
    pending, corrupt = [], 0
    for mid in matches["match_id"]:
        f = shots_dir / f"{mid}.parquet"
        if refresh or not f.exists():
            pending.append(mid)
        elif not _is_valid_parquet(f):   # caché dañada (p.ej. disco lleno): volver a descargar
            corrupt += 1
            pending.append(mid)
    if corrupt:
        log.warning("Archivos de caché dañados, se descargan de nuevo: %d", corrupt)
    log.info("Partidos a descargar: %d (cacheados: %d)", len(pending), len(matches) - len(pending))

    failed = []
    with ThreadPoolExecutor(max_workers=data_cfg.get("n_workers", 8)) as pool:
        futures = {pool.submit(_download_match, mid, shots_dir / f"{mid}.parquet"): mid for mid in pending}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Eventos"):
            try:
                fut.result()
            except Exception as e:
                failed.append(futures[fut])
                log.warning("Fallo match %s: %s", futures[fut], e)
    if failed:
        log.warning("Fallaron %d partidos (se reintentarán en la próxima ejecución)", len(failed))

    files = [shots_dir / f"{mid}.parquet" for mid in matches["match_id"]]
    frames = [pd.read_parquet(f) for f in files if f.exists()]
    shots = pd.concat([f for f in frames if len(f)], ignore_index=True)
    shots = shots.merge(matches, on="match_id", how="left")

    out = cfg["paths"]["interim_dir"] / "shots.parquet"
    shots.to_parquet(out, index=False)
    log.info("Tiros totales: %d -> %s", len(shots), out)
    return shots