"""FastAPI backend para inferencia de xG (versión para producción).

Configuración por variables de entorno:
  MODEL_PATH     ruta del .joblib            (def.: models/xg_model.joblib)
  MODEL_VERSION  etiqueta de versión         (def.: "dev")
  CORS_ORIGINS   orígenes permitidos, coma   (def.: http://localhost:5173,http://127.0.0.1:5173)
  LOG_LEVEL      INFO | DEBUG ...            (def.: INFO)
  DISABLE_DOCS   "1" desactiva /docs         (def.: docs activas)
"""
import bisect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # para importar src.* (y para deserializar el modelo)

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from src.features import CAT_FEATURES, NUM_FEATURES, _freeze_frame_row, _geometry, _interactions

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
log = logging.getLogger("xg-api")

# ---------------------------------------------------------------------------
# Configuración y artefactos (se cargan una sola vez al arrancar; si algo falta, el
# servicio no arranca en lugar de fallar en la primera petición)
# ---------------------------------------------------------------------------
API_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "models" / "xg_model.joblib"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "dev")
CORS_ORIGINS = [o.strip() for o in os.getenv(
    "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]

OPTIONS = json.loads((API_DIR / "feature_options.json").read_text(encoding="utf-8"))
PRIORS = json.loads((API_DIR / "feature_priors.json").read_text(encoding="utf-8"))

if not MODEL_PATH.exists():
    raise RuntimeError(f"No existe el modelo en {MODEL_PATH}. Define MODEL_PATH o copia el .joblib.")
model = joblib.load(MODEL_PATH)
log.info("Modelo cargado: %s (versión %s)", MODEL_PATH.name, MODEL_VERSION)

app = FastAPI(
    title="xG Model API",
    version="1.1",
    docs_url=None if os.getenv("DISABLE_DOCS") == "1" else "/docs",
    redoc_url=None,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

CATEGORICAL_FIELDS = ["play_pattern", "shot_body_part", "shot_technique", "shot_type",
                      "prev_type", "kp_height", "kp_type", "kp_technique"]
KP_NUM = ["kp_length", "kp_angle", "kp_start_x", "kp_start_y"]
KP_BOOL = ["kp_cross", "kp_through_ball", "kp_cut_back", "kp_switch"]
KP_CAT = ["kp_height", "kp_type", "kp_technique"]


# ---------------------------------------------------------------------------
# Esquema de entrada
# ---------------------------------------------------------------------------
class Player(BaseModel):
    """Jugador colocado en el campo (freeze frame manual)."""
    x: float = Field(..., ge=0, le=120)
    y: float = Field(..., ge=0, le=80)
    teammate: bool = False   # True = compañero del tirador; False = rival
    gk: bool = False         # True = portero (rival)


class ShotInput(BaseModel):
    """Datos de un tiro simulado por el usuario (coordenadas StatsBomb: campo 120x80)."""
    x: float = Field(..., ge=0, le=120, description="Posición X del tiro (0-120 yardas)")
    y: float = Field(..., ge=0, le=80, description="Posición Y del tiro (0-80 yardas)")

    minute: int = Field(default=45, ge=0, le=130)
    period: int = Field(default=1, ge=1, le=4)

    under_pressure: bool = False
    shot_first_time: bool = False
    shot_one_on_one: bool = False
    shot_open_goal: bool = False
    shot_redirect: bool = False
    shot_follows_dribble: bool = False
    shot_aerial_won: bool = False

    play_pattern: str = "Regular Play"
    shot_body_part: str = "Right Foot"
    shot_technique: str = "Normal"
    shot_type: str = "Open Play"

    score_diff: int = Field(default=0, ge=-15, le=15)
    secs_since_prev_event: float = Field(default=5.0, ge=0, le=3600)
    poss_elapsed_s: float = Field(default=10.0, ge=0, le=3600)
    poss_n_passes: int = Field(default=3, ge=0, le=200)
    poss_progress_x: float = Field(default=30.0, ge=-120, le=120)
    secs_since_team_last_shot: float = Field(default=120.0, ge=0, le=10000)
    carry_length: float = Field(default=0.0, ge=0, le=120)

    has_key_pass: bool = True
    kp_length: float = Field(default=10.0, ge=0, le=130)
    kp_angle: float = Field(default=0.5, ge=-3.5, le=3.5)
    kp_start_x: float = Field(default=90.0, ge=0, le=120)
    kp_start_y: float = Field(default=40.0, ge=0, le=80)
    kp_cross: bool = False
    kp_through_ball: bool = False
    kp_cut_back: bool = False
    kp_switch: bool = False
    prev_same_team: bool = True
    prev_type: str = "Pass"
    kp_height: str = "Ground Pass"
    kp_type: str = "None"
    kp_technique: str = "None"

    n_defenders_in_cone: int = Field(default=2, ge=0, le=11)
    n_teammates_in_cone: int = Field(default=0, ge=0, le=10)
    gk_in_cone: bool = True
    gk_distance_to_goal: float = Field(default=2.0, ge=0, le=20)
    gk_distance_to_shooter: Optional[float] = Field(default=None, ge=0, le=130)
    gk_offset_line: float = Field(default=1.5, ge=0, le=10)
    gk_lateral_displacement: float = Field(default=0.0, ge=-5, le=5)
    gk_advance: float = Field(default=2.0, ge=0, le=15)
    nearest_defender_dist: float = Field(default=3.0, ge=0, le=30)
    n_defenders_pressure: int = Field(default=1, ge=0, le=5)
    n_defenders_nearby: int = Field(default=2, ge=0, le=8)
    n_defenders_ahead: int = Field(default=3, ge=0, le=11)
    n_defenders_in_box: int = Field(default=3, ge=0, le=11)
    n_teammates_in_box: int = Field(default=1, ge=0, le=10)

    # Si se envían jugadores, TODAS las variables de freeze frame se calculan con ellos
    # (mismo código que en entrenamiento) y se ignoran los controles aproximados.
    players: Optional[list[Player]] = Field(default=None, max_length=22)

    @model_validator(mode="after")
    def _check_categories(self):
        """Rechaza categorías que el modelo no vio al entrenar (antes se ignoraban sin avisar)."""
        for field in CATEGORICAL_FIELDS:
            value = getattr(self, field)
            if value not in OPTIONS[field]:
                raise ValueError(f"{field}='{value}' no es válido; opciones: {OPTIONS[field]}")
        return self


class ShotResult(BaseModel):
    xg: float
    model_version: str
    features_used: dict


# ---------------------------------------------------------------------------
# Construcción de features (misma lógica que en entrenamiento)
# ---------------------------------------------------------------------------
def _geometry_features(shot: ShotInput) -> dict:
    """Reutiliza src.features para que geometría e interacciones sean idénticas a las de entrenamiento."""
    frame = pd.DataFrame([{
        "x": shot.x, "y": shot.y,
        "shot_first_time": int(shot.shot_first_time), "shot_one_on_one": int(shot.shot_one_on_one),
    }])
    frame = _interactions(_geometry(frame))
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in frame.iloc[0].to_dict().items()
            if k not in ("shot_first_time", "shot_one_on_one")}


def _free_goal_fractions(n_in_cone: int, gk_in_cone: bool, distance: float) -> tuple[float, float]:
    """Fracción de portería libre (sin / con portero y compañeros) a partir de los conteos del simulador.

    El simulador no conoce las posiciones exactas de los jugadores, así que se usa la media real
    observada en los datos de entrenamiento para ese nº de rivales en el cono, portero en el cono
    y distancia al arco (tabla en feature_priors.json). Sustituye a la heurística de "3,5° por defensor",
    que se alejaba mucho de la distribución real.
    """
    n = min(n_in_cone, PRIORS["n_max"])
    b = bisect.bisect_right(PRIORS["dist_bins"], distance)
    cell = PRIORS["cells"].get(f"{n}|{int(gk_in_cone)}|{b}") or PRIORS["coarse"][f"{n}|{int(gk_in_cone)}"]
    return float(cell[0]), float(cell[1])


def build_row(shot: ShotInput) -> dict:
    geom = _geometry_features(shot)
    pct_def, pct_total = _free_goal_fractions(shot.n_defenders_in_cone, shot.gk_in_cone, geom["distancia_arco"])

    gk_dist_shooter = shot.gk_distance_to_shooter
    if gk_dist_shooter is None:
        gk_dist_shooter = float(np.hypot(120.0 - shot.gk_advance - shot.x, 40.0 - shot.y))

    row = {
        "minute": shot.minute,
        "under_pressure": int(shot.under_pressure),
        "shot_first_time": int(shot.shot_first_time),
        "shot_one_on_one": int(shot.shot_one_on_one),
        "shot_open_goal": int(shot.shot_open_goal),
        "shot_redirect": int(shot.shot_redirect),
        "shot_follows_dribble": int(shot.shot_follows_dribble),
        "shot_aerial_won": int(shot.shot_aerial_won),
        **geom,
        "has_freeze_frame": 1,
        "n_rivales_en_cono": shot.n_defenders_in_cone,
        "n_companeros_en_cono": shot.n_teammates_in_cone,
        "gk_en_cono": int(shot.gk_in_cone),
        "gk_dist_arco": shot.gk_distance_to_goal,
        "gk_dist_tirador": float(gk_dist_shooter),
        "gk_offset_linea": shot.gk_offset_line,
        "dist_rival_mas_cercano": shot.nearest_defender_dist,
        "n_rivales_presion": shot.n_defenders_pressure,
        "pct_libre_def": pct_def,
        "pct_libre_total": pct_total,
        "gk_desplaz_lateral": shot.gk_lateral_displacement,
        "gk_salida": shot.gk_advance,
        "n_rivales_cerca": shot.n_defenders_nearby,
        "n_rivales_adelante": shot.n_defenders_ahead,
        "n_rivales_area": shot.n_defenders_in_box,
        "n_companeros_area": shot.n_teammates_in_box,
        "period": shot.period,
        "score_diff": shot.score_diff,
        "secs_since_prev_event": shot.secs_since_prev_event,
        "poss_elapsed_s": shot.poss_elapsed_s,
        "poss_n_passes": shot.poss_n_passes,
        "poss_progress_x": shot.poss_progress_x,
        "secs_since_team_last_shot": min(shot.secs_since_team_last_shot, 600.0),  # tope de entrenamiento
        "carry_length": shot.carry_length,
        "has_key_pass": int(shot.has_key_pass),
        "prev_same_team": int(shot.prev_same_team),
        "play_pattern": shot.play_pattern,
        "shot_body_part": shot.shot_body_part,
        "shot_technique": shot.shot_technique,
        "shot_type": shot.shot_type,
        "prev_type": shot.prev_type,
    }
    if shot.has_key_pass:
        row.update({c: getattr(shot, c) for c in KP_NUM + KP_CAT})
        row.update({c: int(getattr(shot, c)) for c in KP_BOOL})
    else:
        # Igual que en entrenamiento: sin pase clave, los datos del pase no existen (vacíos / "None")
        row.update({c: np.nan for c in KP_NUM})
        row.update({c: "None" for c in KP_CAT})
        row.update({c: 0 for c in KP_BOOL})
    if shot.players:
        ff = [{"location": [p.x, p.y], "teammate": p.teammate,
               "position": {"name": "Goalkeeper" if (p.gk and not p.teammate) else "Other"}}
              for p in shot.players]
        row.update(_freeze_frame_row(json.dumps(ff), shot.x, shot.y))
    return row


def predict_xg_value(shot: ShotInput) -> tuple[float, dict]:
    row = build_row(shot)
    frame = pd.DataFrame([row])[NUM_FEATURES + CAT_FEATURES]  # orden exacto de columnas
    proba = float(model.predict_proba(frame)[0, 1])
    return proba, row


FREEZE_SUMMARY = ("pct_libre_def", "pct_libre_total", "n_rivales_en_cono", "dist_rival_mas_cercano", "gk_dist_arco")


def _num(v):
    """NaN no es JSON válido: se devuelve None."""
    return None if v is None or (isinstance(v, float) and v != v) else v


# Autocomprobación al arrancar: si las features no coinciden con el modelo, falla ahora, no en producción
_p, _ = predict_xg_value(ShotInput(x=108, y=40))
log.info("Autocomprobación OK (xG de prueba = %.4f)", _p)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=ShotResult)
def predict_xg(shot: ShotInput):
    try:
        xg, row = predict_xg_value(shot)
    except Exception:
        log.exception("Error al predecir")
        raise HTTPException(status_code=500, detail="Error interno al calcular el xG")
    return ShotResult(
        xg=round(xg, 4),
        model_version=MODEL_VERSION,
        features_used={"geometry": {k: row[k] for k in ("x", "y", "distancia_arco", "angulo")},
                       "freeze": {k: _num(row[k]) for k in FREEZE_SUMMARY},
                       "manual_players": bool(shot.players),
                       "total_features": len(NUM_FEATURES) + len(CAT_FEATURES)},
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH.name, "version": MODEL_VERSION}


@app.get("/model-info")
def model_info():
    return {
        "version": MODEL_VERSION,
        "model_file": MODEL_PATH.name,
        "members": getattr(model, "names", [type(model).__name__]),
        "n_features": len(NUM_FEATURES) + len(CAT_FEATURES),
        "note": "Simulador: freeze frame y contexto se aproximan a partir de los parámetros introducidos.",
    }


@app.get("/feature-options")
def feature_options():
    """Opciones válidas de los campos categóricos (las categorías vistas en entrenamiento)."""
    return OPTIONS