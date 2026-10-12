"""Limpieza de tiros: coordenadas, booleanos StatsBomb, categóricas y target."""
import json
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Atributos del tiro conocidos ANTES de que el balón llegue a puerta.
# (Se quitó "shot_deflected": un desvío ocurre después del golpeo, es información post-tiro.)
BOOL_COLS = [
    "under_pressure", "shot_first_time", "shot_one_on_one", "shot_open_goal",
    "shot_redirect", "shot_follows_dribble", "shot_aerial_won",
]
CAT_COLS = ["play_pattern", "shot_body_part", "shot_technique", "shot_type"]

# Contexto previo al tiro (lo calcula data_loader._context_features).
CONTEXT_NUM = [
    "period", "score_diff", "secs_since_prev_event", "poss_elapsed_s", "poss_n_passes",
    "poss_progress_x", "secs_since_team_last_shot", "carry_length",
    "kp_length", "kp_angle", "kp_start_x", "kp_start_y",
]
CONTEXT_BOOL = ["has_key_pass", "kp_cross", "kp_through_ball", "kp_cut_back", "kp_switch", "prev_same_team"]
CONTEXT_CAT = ["prev_type", "kp_height", "kp_type", "kp_technique"]

META_COLS = [
    "id", "match_id", "match_date", "competition_name", "season_name",
    "competition_gender", "period", "minute", "shot_statsbomb_xg", "shot_freeze_frame",
]


def _xy(value):
    try:
        loc = json.loads(value) if isinstance(value, str) else value
        return float(loc[0]), float(loc[1])
    except Exception:
        return np.nan, np.nan


def clean_shots(df, cfg):
    df = df.copy()
    n0 = len(df)

    df = df[df["period"] != 5]
    if cfg["preprocessing"].get("exclude_penalties", True):
        df = df[df["shot_type"] != "Penalty"]

    xy = df["location"].map(_xy)
    df["x"] = xy.map(lambda t: t[0])
    df["y"] = xy.map(lambda t: t[1])
    df = df[df["x"].between(0, 120) & df["y"].between(0, 80)]

    for col in BOOL_COLS + CONTEXT_BOOL:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(False).astype(bool).astype(int)

    for col in CAT_COLS:
        df[col] = df[col].fillna("Unknown").astype(str)
    for col in CONTEXT_CAT:
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].fillna("None").astype(str)

    df["goal"] = (df["shot_outcome"] == "Goal").astype(int)
    df["match_date"] = pd.to_datetime(df["match_date"])

    for col in META_COLS:
        if col not in df.columns:
            df[col] = np.nan
    for col in CONTEXT_NUM:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    keep = (META_COLS + ["x", "y"] + BOOL_COLS + CAT_COLS
            + [c for c in CONTEXT_NUM if c not in META_COLS] + CONTEXT_BOOL + CONTEXT_CAT + ["goal"])
    df = df[keep].reset_index(drop=True)
    log.info("Tiros tras limpieza: %d de %d (tasa de gol %.2f%%)", len(df), n0, 100 * df["goal"].mean())
    return df