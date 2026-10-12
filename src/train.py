"""Entrenamiento: RandomizedSearchCV con StratifiedGroupKFold por partido, predicciones
out-of-fold (OOF) de cada mejor modelo y ensembles por soft voting de combinaciones de modelos.

Todas las métricas del leaderboard se calculan sobre OOF de train, así modelos individuales
y ensembles se comparan en igualdad de condiciones sin tocar el test.
"""
import logging
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.ensemble import SoftVotingEnsemble, ensemble_name
from src.features import CAT_FEATURES, NUM_FEATURES
from src.models import build_preprocessor, get_models
from src.report import best_f1, get_thresholds

log = logging.getLogger(__name__)

CV_GROUPS = {"match": "match_id", "competition": "competition_name"}
SCORING = {"log_loss": "neg_log_loss", "brier": "neg_brier_score", "roc_auc": "roc_auc"}
EPS = 1e-6


def _to_native(params):
    """np.float64 -> float, np.int64 -> int, etc.

    RandomizedSearchCV devuelve hiperparámetros como tipos numpy (salen de distribuciones de
    scipy). CatBoost los convierte internamente a tipos nativos de Python, y entonces
    sklearn.clone() falla con "constructor either does not set or modifies parameter ...".
    """
    return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in params.items()}


def _oof_metrics(y, p, thresholds):
    p = np.clip(p, EPS, 1 - EPS)
    f1, thr = best_f1(y, p, thresholds)
    return {
        "oof_log_loss": log_loss(y, p),
        "oof_brier": brier_score_loss(y, p),
        "oof_roc_auc": roc_auc_score(y, p),
        "oof_best_f1": f1,
        "oof_best_f1_umbral": thr,
    }


def _fit_single(name, estimator, params, X, y, groups, cv, tcfg, reports_dir):
    pipe = Pipeline([("prep", build_preprocessor(NUM_FEATURES, CAT_FEATURES)), ("model", estimator)])
    search = RandomizedSearchCV(
        pipe,
        {f"model__{k}": v for k, v in params.items()},
        n_iter=tcfg.get("n_iter", 20),
        scoring=SCORING,
        refit="log_loss",
        cv=cv,
        n_jobs=tcfg.get("n_jobs", -1),
        random_state=tcfg.get("random_state", 42),
        error_score=np.nan,
    )
    search.fit(X, y, groups=groups)
    pd.DataFrame(search.cv_results_).to_csv(reports_dir / f"cv_results_{name}.csv", index=False)

    # Mejores hiperparámetros con tipos nativos y modelo reconstruido desde el pipeline base,
    # de modo que cross_val_predict (y cualquier clone posterior) funcione también con CatBoost.
    best_params = _to_native(search.best_params_)
    best_estimator = clone(pipe).set_params(**best_params)

    oof = cross_val_predict(best_estimator, X, y, groups=groups, cv=cv,
                            method="predict_proba", n_jobs=tcfg.get("n_jobs", -1))[:, 1]
    best_estimator.fit(X, y)  # ajuste final sobre todo el train

    params_out = {k.replace("model__", ""): v for k, v in best_params.items()}
    return best_estimator, oof, params_out


def train_models(train_df, cfg):
    tcfg = cfg["training"]
    rs = tcfg.get("random_state", 42)
    models_dir = cfg["paths"]["models_dir"]
    reports_dir = cfg["paths"]["reports_dir"]
    thresholds = get_thresholds(cfg)

    X = train_df[NUM_FEATURES + CAT_FEATURES]
    y = train_df["goal"].values
    cv_group = tcfg.get("cv_group", "match")
    groups = train_df[CV_GROUPS[cv_group]].values
    n_groups = len(np.unique(groups))
    if n_groups < tcfg.get("cv_folds", 5):
        raise ValueError(f"cv_group={cv_group}: hay {n_groups} grupos, menos que cv_folds")
    log.info("CV agrupada por %s (%d grupos)", cv_group, n_groups)
    cv = StratifiedGroupKFold(n_splits=tcfg.get("cv_folds", 5), shuffle=True, random_state=rs)

    fitted, oof, rows = {}, {}, []
    for name, (estimator, params) in get_models(tcfg["models"], rs).items():
        log.info("Entrenando %s ...", name)
        fitted[name], oof[name], params_out = _fit_single(name, estimator, params, X, y, groups, cv, tcfg, reports_dir)
        joblib.dump(fitted[name], models_dir / f"{name}.joblib")
        row = {"modelo": name, "tipo": "individual", "miembros": name,
               **_oof_metrics(y, oof[name], thresholds), "params": params_out}
        rows.append(row)
        log.info("%s | OOF log loss %.4f | AUC %.4f | best F1 %.4f @ %.2f", name,
                 row["oof_log_loss"], row["oof_roc_auc"], row["oof_best_f1"], row["oof_best_f1_umbral"])

    if not rows:
        raise RuntimeError("No se entrenó ningún modelo")

    k = tcfg.get("ensemble_size", 3)
    if len(fitted) >= k:
        for combo in combinations(fitted, k):
            name = ensemble_name(combo)
            p = np.mean([oof[m] for m in combo], axis=0)
            rows.append({"modelo": name, "tipo": "ensemble", "miembros": "+".join(combo),
                         **_oof_metrics(y, p, thresholds), "params": None})
        log.info("Ensembles evaluados (soft voting, %d modelos): %d", k, sum(r["tipo"] == "ensemble" for r in rows))

    pd.DataFrame(oof).assign(goal=y, match_id=train_df["match_id"].values).to_parquet(reports_dir / "oof_predicciones.parquet", index=False)
    leaderboard = pd.DataFrame(rows).sort_values("oof_log_loss").reset_index(drop=True)
    leaderboard.to_csv(reports_dir / "cv_leaderboard.csv", index=False)

    for label, col in (("best_model", "oof_log_loss"), ("best_f1_model", "oof_best_f1")):
        idx = leaderboard[col].idxmin() if col == "oof_log_loss" else leaderboard[col].idxmax()
        members = leaderboard.loc[idx, "miembros"].split("+")
        obj = fitted[members[0]] if len(members) == 1 else SoftVotingEnsemble({m: fitted[m] for m in members})
        joblib.dump(obj, models_dir / f"{label}.joblib")
        log.info("%s: %s", label, leaderboard.loc[idx, "modelo"])
    return leaderboard