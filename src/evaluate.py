"""Evaluación en test: métricas probabilísticas, benchmark contra el xG de StatsBomb,
calibración, desglose por competición e importancia de variables."""
import json
import logging

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from src.ensemble import members_of
from src.features import CAT_FEATURES, NUM_FEATURES
from src.report import get_thresholds, threshold_report

log = logging.getLogger(__name__)
EPS = 1e-6


def expected_calibration_error(y, p, n_bins=10):
    bins = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return ece


def compute_metrics(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return {
        "log_loss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "ece": expected_calibration_error(y, p),
        "goles": int(y.sum()),
        "xg_total": float(p.sum()),
    }


def _predict_all(test_df, models_dir, leaderboard):
    X = test_df[NUM_FEATURES + CAT_FEATURES]
    preds, models = {}, {}
    singles = leaderboard.loc[leaderboard["tipo"] == "individual", "modelo"]
    for name in singles:
        path = models_dir / f"{name}.joblib"
        if path.exists():
            models[name] = joblib.load(path)
            preds[name] = models[name].predict_proba(X)[:, 1]
    for name in leaderboard.loc[leaderboard["tipo"] == "ensemble", "modelo"]:
        members = members_of(name)
        if all(m in preds for m in members):
            preds[name] = np.mean([preds[m] for m in members], axis=0)
    return preds, models


def _reliability_plot(y, preds, path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfecta")
    for name, p in preds.items():
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=name)
    ax.set_xlabel("xG predicho")
    ax.set_ylabel("Tasa de gol observada")
    ax.set_title("Curva de calibración (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _scatter_vs_sb(y, p, sb, name, path):
    fig, ax = plt.subplots(figsize=(9, 8))
    colors = np.where(y == 1, "limegreen", "red")
    ax.scatter(sb, p, c=colors, s=8, alpha=0.5)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("xG StatsBomb")
    ax.set_ylabel(f"xG {name}")
    ax.set_title(f"{name} vs StatsBomb (verde = gol)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _feature_importance(model, path):
    prep, est = model.named_steps["prep"], model.named_steps["model"]
    names = prep.get_feature_names_out()
    if hasattr(est, "feature_importances_"):
        imp = est.feature_importances_
    elif hasattr(est, "coef_"):
        imp = np.abs(est.coef_[0])
    else:
        return
    fi = pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False)
    fi.to_csv(path, index=False)


def evaluate(test_df, cfg, leaderboard):
    reports = cfg["paths"]["reports_dir"]
    y = test_df["goal"].values
    preds, models = _predict_all(test_df, cfg["paths"]["models_dir"], leaderboard)

    sb = test_df["shot_statsbomb_xg"].astype(float).values
    has_sb = ~np.isnan(sb)

    rows = []
    for name, p in preds.items():
        m = compute_metrics(y, p)
        m["rmse_vs_sb"] = float(np.sqrt(np.mean((p[has_sb] - sb[has_sb]) ** 2)))
        m["corr_vs_sb"] = float(np.corrcoef(p[has_sb], sb[has_sb])[0, 1])
        rows.append({"modelo": name, **m})
    if has_sb.any():
        rows.append({"modelo": "statsbomb_xg", **compute_metrics(y[has_sb], sb[has_sb]),
                     "rmse_vs_sb": 0.0, "corr_vs_sb": 1.0})

    results = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    results.to_csv(reports / "test_metrics.csv", index=False)

    best = leaderboard.loc[leaderboard["oof_log_loss"].idxmin(), "modelo"]
    best_f1_name = leaderboard.loc[leaderboard["oof_best_f1"].idxmax(), "modelo"]

    test_df = test_df.assign(xg_modelo=preds[best], xg_best_f1=preds[best_f1_name])
    by_comp = (test_df.groupby(["competition_name", "season_name"])
               .agg(tiros=("goal", "size"), goles=("goal", "sum"),
                    xg_modelo=("xg_modelo", "sum"), xg_statsbomb=("shot_statsbomb_xg", "sum"))
               .reset_index().sort_values("tiros", ascending=False))
    by_comp.to_csv(reports / "test_por_competicion.csv", index=False)
    test_df[["id", "match_id", "competition_name", "season_name", "goal",
             "shot_statsbomb_xg", "xg_modelo", "xg_best_f1"]].to_parquet(reports / "test_predicciones.parquet", index=False)

    plot_preds = {n: preds[n] for n in models}
    plot_preds[best] = preds[best]
    if has_sb.all():
        plot_preds["statsbomb_xg"] = sb
    _reliability_plot(y, plot_preds, reports / "calibracion.png")
    _scatter_vs_sb(y[has_sb], preds[best][has_sb], sb[has_sb], best, reports / "scatter_best_vs_sb.png")
    if best in models:
        _feature_importance(models[best], reports / f"feature_importance_{best}.csv")

    summary = {"mejor_modelo_log_loss": best, "mejor_modelo_f1": best_f1_name,
               "metricas_test": results.to_dict(orient="records")}
    with open(reports / "resumen.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        log.info("Resultados en test (top 15 por log loss):\n%s", results.head(15).round(4).to_string(index=False))
        log.info("Mejor por log loss (OOF): %s | Mejor por F1 (OOF): %s", best, best_f1_name)
        threshold_report(
            y[has_sb],
            {"statsbomb_xg": sb[has_sb], best_f1_name: preds[best_f1_name][has_sb]},
            get_thresholds(cfg),
            path=reports / "reporte_umbrales.csv",
        )
    return results
