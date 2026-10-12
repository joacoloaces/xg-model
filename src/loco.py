"""Leave-one-competition-out: para cada competición se reentrena el modelo elegido
(mismos hiperparámetros) con el resto y se predice la competición excluida.
Mide cuánto generaliza el modelo a competiciones que nunca vio, frente a StatsBomb."""
import logging

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone

from src.ensemble import members_of
from src.evaluate import compute_metrics
from src.features import CAT_FEATURES, NUM_FEATURES

log = logging.getLogger(__name__)
FEATURES = NUM_FEATURES + CAT_FEATURES


def _resolve_model(which, leaderboard):
    if which == "best_model":
        return leaderboard.loc[leaderboard["oof_log_loss"].idxmin(), "modelo"]
    if which == "best_f1_model":
        return leaderboard.loc[leaderboard["oof_best_f1"].idxmax(), "modelo"]
    return which


def _fold(comp, df, templates):
    tr = df[df["competition_name"] != comp]
    te = df[df["competition_name"] == comp]
    probs = []
    for model in templates.values():
        m = clone(model).fit(tr[FEATURES], tr["goal"].values)
        probs.append(m.predict_proba(te[FEATURES])[:, 1])
    return comp, te.index.values, np.mean(probs, axis=0)


def _metrics(y, p, prefix):
    if len(np.unique(y)) < 2:
        return {f"{prefix}_xg": float(np.sum(p))}
    m = compute_metrics(y, p)
    return {f"{prefix}_log_loss": m["log_loss"], f"{prefix}_brier": m["brier"],
            f"{prefix}_auc": m["roc_auc"], f"{prefix}_xg": m["xg_total"]}


def _row(name, y, p, sb):
    row = {"competicion": name, "tiros": len(y), "goles": int(y.sum())}
    row.update(_metrics(y, p, "modelo"))
    row.update(_metrics(y, sb, "sb"))
    row["dif_modelo_%"] = 100 * (row["modelo_xg"] / max(row["goles"], 1) - 1)
    row["dif_sb_%"] = 100 * (row["sb_xg"] / max(row["goles"], 1) - 1)
    if "modelo_log_loss" in row and "sb_log_loss" in row:
        row["gana_log_loss"] = "modelo" if row["modelo_log_loss"] < row["sb_log_loss"] else "statsbomb"
    return row


def run_loco(df, cfg, leaderboard):
    lcfg = cfg.get("loco", {})
    reports, models_dir = cfg["paths"]["reports_dir"], cfg["paths"]["models_dir"]

    name = _resolve_model(lcfg.get("model", "best_model"), leaderboard)
    templates = {m: joblib.load(models_dir / f"{m}.joblib") for m in members_of(name)}

    df = df[df["shot_statsbomb_xg"].notna()].reset_index(drop=True)
    counts = df["competition_name"].value_counts()
    comps = counts[counts >= lcfg.get("min_shots", 100)].index.tolist()
    skipped = counts[counts < lcfg.get("min_shots", 100)]
    if len(skipped):
        log.info("Competiciones omitidas por pocos tiros: %s", dict(skipped))
    log.info("LOCO con %s sobre %d competiciones", name, len(comps))

    results = Parallel(n_jobs=lcfg.get("n_jobs", -1), verbose=5)(
        delayed(_fold)(c, df, templates) for c in comps)

    preds = pd.Series(np.nan, index=df.index)
    for _, idx, p in results:
        preds.loc[idx] = p

    rows = []
    for comp, idx, p in results:
        part = df.loc[idx]
        rows.append(_row(comp, part["goal"].values, p, part["shot_statsbomb_xg"].values))
    mask = preds.notna()
    rows.append(_row("TOTAL", df.loc[mask, "goal"].values, preds[mask].values,
                     df.loc[mask, "shot_statsbomb_xg"].values))

    table = pd.DataFrame(rows)
    body = table[table["competicion"] != "TOTAL"].sort_values("tiros", ascending=False)
    table = pd.concat([body, table[table["competicion"] == "TOTAL"]], ignore_index=True)
    table.to_csv(reports / "loco_por_competicion.csv", index=False)
    df.loc[mask, ["id", "match_id", "competition_name", "season_name", "goal", "shot_statsbomb_xg"]] \
        .assign(xg_loco=preds[mask].values).to_parquet(reports / "loco_predicciones.parquet", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(f"\n{'=' * 78}\nLEAVE-ONE-COMPETITION-OUT | {name}\n{'=' * 78}")
        print(table.round(4).to_string(index=False))
    return table
