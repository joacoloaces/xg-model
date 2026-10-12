"""Feature engineering: geometría del tiro, interacciones y freeze frame.

Coordenadas StatsBomb en yardas: campo 120x80, portería en x=120 con postes en y=36 e y=44.

Cambios respecto a la versión anterior (todo se calcula con las columnas que ya tienes):
  * FIX: `dist_rival_mas_cercano` incluía al portero (a corta distancia el "rival más cercano"
    era casi siempre el portero). Ahora solo cuenta rivales de campo.
  * NUEVO: fracción de portería visible desde el punto de tiro (descontando rivales, portero
    y compañeros que tapan la línea de tiro).
  * NUEVO: desplazamiento lateral y salida del portero, rivales por delante, ocupación del área.
"""
import json
import logging

import numpy as np
import pandas as pd

from src.preprocessing import BOOL_COLS, CAT_COLS, CONTEXT_BOOL, CONTEXT_CAT, CONTEXT_NUM

log = logging.getLogger(__name__)

GOAL_X, GOAL_Y = 120.0, 40.0
POST_LOW, POST_HIGH = 36.0, 44.0
PRESSURE_RADIUS = 3.0
NEAR_RADIUS = 6.0
BODY_RADIUS = 0.6      # "anchura" aproximada de un jugador de campo (yardas) para el bloqueo
GK_RADIUS = 0.9        # portero con brazos/piernas abiertos

GEOMETRY_COLS = [
    "x", "y", "distancia_arco", "angulo", "dist_x", "offset_y",
    "dentro_del_area", "dentro_area_chica", "franja_central",
]
INTERACTION_COLS = [
    "distancia_arco*franja_central", "distancia/angulo", "first_time_dentro_area",
    "central_one_on_one", "angulo_one_on_one",
]
FREEZE_COLS = [
    "has_freeze_frame", "n_rivales_en_cono", "n_companeros_en_cono", "gk_en_cono",
    "gk_dist_arco", "gk_dist_tirador", "gk_offset_linea", "dist_rival_mas_cercano",
    "n_rivales_presion",
    # --- nuevos
    "pct_libre_def", "pct_libre_total", "gk_desplaz_lateral", "gk_salida",
    "n_rivales_cerca", "n_rivales_adelante", "n_rivales_area", "n_companeros_area",
]
NUM_FEATURES = (["minute"] + BOOL_COLS + GEOMETRY_COLS + INTERACTION_COLS + FREEZE_COLS
                + CONTEXT_NUM + CONTEXT_BOOL)
CAT_FEATURES = CAT_COLS + CONTEXT_CAT


def _geometry(df):
    dx = GOAL_X - df["x"]
    df["distancia_arco"] = np.hypot(dx, GOAL_Y - df["y"])
    ang = np.arctan2(POST_HIGH - df["y"], dx) - np.arctan2(POST_LOW - df["y"], dx)
    df["angulo"] = np.degrees(np.abs(ang))
    df["dist_x"] = dx
    df["offset_y"] = (df["y"] - GOAL_Y).abs()
    df["franja_central"] = df["y"].between(18, 62).astype(int)
    df["dentro_del_area"] = ((df["x"] >= 102) & df["y"].between(18, 62)).astype(int)
    df["dentro_area_chica"] = ((df["x"] >= 114) & df["y"].between(30, 50)).astype(int)
    return df


def _interactions(df):
    df["distancia_arco*franja_central"] = df["distancia_arco"] * df["franja_central"]
    df["distancia/angulo"] = df["distancia_arco"] / (df["angulo"] + 1)
    df["first_time_dentro_area"] = df["dentro_del_area"] * df["shot_first_time"]
    df["central_one_on_one"] = df["franja_central"] * df["shot_one_on_one"]
    df["angulo_one_on_one"] = df["angulo"] * df["shot_one_on_one"]
    return df


def _in_triangle(pts, a, b, c):
    def sign(p, q, r):
        return (p[:, 0] - r[0]) * (q[1] - r[1]) - (q[0] - r[0]) * (p[:, 1] - r[1])
    d1, d2, d3 = sign(pts, a, b), sign(pts, b, c), sign(pts, c, a)
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(neg & pos)


def _position_name(p):
    pos = p.get("position")
    if isinstance(pos, dict):
        return pos.get("name")
    return p.get("position_name", pos)


def _fraccion_libre(shooter, blockers, radii):
    """Fracción (0-1) del ángulo de portería que se ve libre desde `shooter`.

    Cada bloqueador (jugador situado entre el tirador y la línea de gol) tapa un intervalo
    angular de semiancho asin(r/d). Se une la cobertura y se recorta al ángulo de portería.
    """
    a1 = np.arctan2(POST_LOW - shooter[1], GOAL_X - shooter[0])
    a2 = np.arctan2(POST_HIGH - shooter[1], GOAL_X - shooter[0])
    lo, hi = min(a1, a2), max(a1, a2)
    total = hi - lo
    if total < 1e-9:
        return np.nan
    intervals = []
    for (bx, by), r in zip(blockers, radii):
        vx, vy = bx - shooter[0], by - shooter[1]
        d = np.hypot(vx, vy)
        if d < 0.1 or vx <= 0 or bx > GOAL_X + 0.5:   # detrás del tirador o fuera del campo
            continue
        half = np.arcsin(min(r / d, 1.0))
        c = np.arctan2(vy, vx)
        s, e = max(c - half, lo), min(c + half, hi)
        if e > s:
            intervals.append((s, e))
    if not intervals:
        return 1.0
    intervals.sort()
    covered, cur_s, cur_e = 0.0, *intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            covered += cur_e - cur_s
            cur_s, cur_e = s, e
    covered += cur_e - cur_s
    return float(max(0.0, 1.0 - covered / total))


def _freeze_frame_row(ff, x, y):
    empty = dict.fromkeys(FREEZE_COLS, np.nan)
    empty["has_freeze_frame"] = 0
    if not isinstance(ff, str) or not ff:
        return empty
    try:
        players = json.loads(ff)
    except json.JSONDecodeError:
        return empty
    if not players:
        return empty

    shooter = np.array([x, y])
    tri = (shooter, np.array([GOAL_X, POST_LOW]), np.array([GOAL_X, POST_HIGH]))
    locs = np.array([p["location"][:2] for p in players], dtype=float)
    mates = np.array([bool(p.get("teammate")) for p in players])
    is_gk = np.array([(not m) and _position_name(p) == "Goalkeeper" for p, m in zip(players, mates)])
    rivals = ~mates & ~is_gk

    in_cone = _in_triangle(locs, *tri)
    dist_shooter = np.hypot(*(locs - shooter).T)
    in_box = (locs[:, 0] >= 102) & (locs[:, 1] >= 18) & (locs[:, 1] <= 62)

    out = {
        "has_freeze_frame": 1,
        "n_rivales_en_cono": int((in_cone & rivals).sum()),
        "n_companeros_en_cono": int((in_cone & mates).sum()),
        "gk_en_cono": np.nan,
        "gk_dist_arco": np.nan,
        "gk_dist_tirador": np.nan,
        "gk_offset_linea": np.nan,
        # FIX: solo rivales de campo (antes incluía al portero)
        "dist_rival_mas_cercano": float(dist_shooter[rivals].min()) if rivals.any() else np.nan,
        "n_rivales_presion": int((rivals & (dist_shooter <= PRESSURE_RADIUS)).sum()),
        "gk_desplaz_lateral": np.nan,
        "gk_salida": np.nan,
        "n_rivales_cerca": int((rivals & (dist_shooter <= NEAR_RADIUS)).sum()),
        "n_rivales_adelante": int((rivals & (locs[:, 0] > x)).sum()),
        "n_rivales_area": int((rivals & in_box).sum()),
        "n_companeros_area": int((mates & in_box).sum()),
    }

    # fracción de portería libre: (a) solo rivales de campo, (b) rivales + portero + compañeros
    rv_idx = np.where(rivals)[0]
    out["pct_libre_def"] = _fraccion_libre(shooter, locs[rv_idx], [BODY_RADIUS] * len(rv_idx))
    all_idx = np.arange(len(players))
    radii = [GK_RADIUS if is_gk[i] else BODY_RADIUS for i in all_idx]
    out["pct_libre_total"] = _fraccion_libre(shooter, locs[all_idx], radii)

    if is_gk.any():
        gk = locs[is_gk][0]
        out["gk_en_cono"] = int(in_cone[is_gk][0])
        out["gk_dist_arco"] = float(np.hypot(GOAL_X - gk[0], GOAL_Y - gk[1]))
        out["gk_dist_tirador"] = float(dist_shooter[is_gk][0])
        out["gk_salida"] = float(GOAL_X - gk[0])
        # + si el portero está desplazado hacia el lado del tirador (cubre el palo cercano)
        side = 1.0 if y >= GOAL_Y else -1.0
        out["gk_desplaz_lateral"] = float((gk[1] - GOAL_Y) * side)
        line = np.array([GOAL_X, GOAL_Y]) - shooter
        norm = np.linalg.norm(line)
        if norm > 0:
            out["gk_offset_linea"] = float(abs(line[0] * (gk - shooter)[1] - line[1] * (gk - shooter)[0]) / norm)
    return out


def _context(df):
    """Rellena con significado los vacíos del contexto previo (así no dependen del imputador)."""
    # sin tiro anterior del equipo = "hace mucho"; tope de 10 min
    df["secs_since_team_last_shot"] = df["secs_since_team_last_shot"].fillna(600).clip(upper=600)
    # sin conducción previa = longitud 0
    df["carry_length"] = df["carry_length"].fillna(0)
    return df


def _freeze_frame(df):
    feats = [_freeze_frame_row(ff, x, y) for ff, x, y in zip(df["shot_freeze_frame"], df["x"], df["y"])]
    return pd.concat([df, pd.DataFrame(feats, index=df.index)], axis=1)


def build_features(df):
    df = df.copy()
    df = _geometry(df)
    df = _interactions(df)
    df = _context(df)
    df = _freeze_frame(df)
    df = df.drop(columns=["shot_freeze_frame"])
    log.info("Features: %d numéricas, %d categóricas | freeze frame en %.1f%% de tiros",
             len(NUM_FEATURES), len(CAT_FEATURES), 100 * df["has_freeze_frame"].mean())
    return df