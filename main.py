"""Orquestador del pipeline xG.

Pasos: download -> features -> train -> evaluate
Ejemplos:
    python main.py
    python main.py --steps train evaluate
    python main.py --competitions "La Liga" "Premier League" --min-season 2015
    python main.py --max-matches 50 --models logreg xgboost
    python main.py --steps train evaluate --split competition --cv-group competition
    python main.py --steps loco
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import load_config, setup_logging
from src.data_loader import download_shots
from src.evaluate import evaluate
from src.features import build_features
from src.loco import run_loco
from src.preprocessing import clean_shots
from src.split import split_shots
from src.train import train_models

STEPS = ["download", "features", "train", "evaluate"]
EXTRA_STEPS = ["loco"]
log = logging.getLogger("main")


def parse_args():
    ap = argparse.ArgumentParser(description="Pipeline de expected goals con StatsBomb Open Data")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.yaml"))
    ap.add_argument("--steps", nargs="+", choices=STEPS + EXTRA_STEPS + ["all"], default=["all"])
    ap.add_argument("--competitions", nargs="+")
    ap.add_argument("--min-season", type=int)
    ap.add_argument("--max-matches", type=int)
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--n-iter", type=int)
    ap.add_argument("--split", choices=["group", "temporal", "competition"])
    ap.add_argument("--test-competitions", nargs="+")
    ap.add_argument("--cv-group", choices=["match", "competition"])
    ap.add_argument("--refresh", action="store_true")
    return ap.parse_args()


def apply_overrides(cfg, args):
    data = cfg["data"]
    if args.competitions:
        data["competitions"] = args.competitions
    if args.min_season:
        data["min_season"] = args.min_season
    if args.max_matches:
        data["max_matches"] = args.max_matches
    if args.refresh:
        data["refresh"] = True
    if args.models:
        cfg["training"]["models"] = args.models
    if args.n_iter:
        cfg["training"]["n_iter"] = args.n_iter
    if args.split:
        cfg["split"]["strategy"] = args.split
    if args.test_competitions:
        cfg["split"]["test_competitions"] = args.test_competitions
    if args.cv_group:
        cfg["training"]["cv_group"] = args.cv_group
    return cfg


def main():
    setup_logging()
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    steps = STEPS if "all" in args.steps else [s for s in STEPS + EXTRA_STEPS if s in args.steps]
    paths = cfg["paths"]

    if "download" in steps:
        log.info("=== 1/4 Descarga ===")
        download_shots(cfg)

    if "features" in steps:
        log.info("=== 2/4 Limpieza y features ===")
        shots = pd.read_parquet(paths["interim_dir"] / "shots.parquet")
        df = build_features(clean_shots(shots, cfg))
        df.to_parquet(paths["processed_dir"] / "shots_features.parquet", index=False)

    if "train" in steps:
        log.info("=== 3/4 Split y entrenamiento ===")
        df = pd.read_parquet(paths["processed_dir"] / "shots_features.parquet")
        train, test = split_shots(df, cfg["split"])
        train.to_parquet(paths["processed_dir"] / "train.parquet", index=False)
        test.to_parquet(paths["processed_dir"] / "test.parquet", index=False)
        lb = train_models(train, cfg).drop(columns=["params", "miembros"])
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(lb.head(15).round(4).to_string(index=False))

    if "evaluate" in steps:
        log.info("=== 4/4 Evaluación en test ===")
        test = pd.read_parquet(paths["processed_dir"] / "test.parquet")
        leaderboard = pd.read_csv(paths["reports_dir"] / "cv_leaderboard.csv")
        evaluate(test, cfg, leaderboard)

    if "loco" in steps:
        log.info("=== Extra: leave-one-competition-out ===")
        df = pd.read_parquet(paths["processed_dir"] / "shots_features.parquet")
        leaderboard = pd.read_csv(paths["reports_dir"] / "cv_leaderboard.csv")
        run_loco(df, cfg, leaderboard)

    log.info("Pipeline terminado")


if __name__ == "__main__":
    main()
