"""Reporte por umbral: matriz de confusión, accuracy, recall, precision y F1."""
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

log = logging.getLogger(__name__)


def get_thresholds(cfg):
    t = cfg.get("report", {}).get("thresholds", {})
    start, stop, step = t.get("start", 0.1), t.get("stop", 0.9), t.get("step", 0.1)
    return np.round(np.arange(start, stop + step / 2, step), 4)


def threshold_metrics(y, p, thr):
    yhat = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "umbral": thr, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": accuracy_score(y, yhat),
        "recall": recall_score(y, yhat, zero_division=0),
        "precision": precision_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
    }


def best_f1(y, p, thresholds):
    rows = [threshold_metrics(y, p, t) for t in thresholds]
    best = max(rows, key=lambda r: r["f1"])
    return best["f1"], best["umbral"]


def _cm(r):
    return pd.DataFrame([[int(r["TN"]), int(r["FP"])], [int(r["FN"]), int(r["TP"])]],
                        index=["real_no_gol", "real_gol"], columns=["pred_no_gol", "pred_gol"])


def threshold_report(y, preds, thresholds, path=None):
    tables = {name: pd.DataFrame([threshold_metrics(y, p, t) for t in thresholds]) for name, p in preds.items()}
    names = list(preds)
    line = "=" * 78

    print(f"\n{line}\nREPORTE POR UMBRAL (test) | {' vs '.join(names)}\n{line}")
    for i, thr in enumerate(thresholds):
        rows = {n: tables[n].iloc[i] for n in names}
        print(f"\n--- Umbral {thr:.2f} ---")
        print(pd.concat({n: _cm(r) for n, r in rows.items()}, axis=1).to_string())
        metrics = pd.DataFrame({n: r[["accuracy", "recall", "precision", "f1"]] for n, r in rows.items()})
        print(metrics.astype(float).round(4).to_string())

    print(f"\n{line}\nRESUMEN\n{line}")
    summary = pd.concat(tables, names=["modelo"]).reset_index(level=0)
    print(summary.round(4).to_string(index=False))
    if path is not None:
        summary.to_csv(path, index=False)
    return summary
