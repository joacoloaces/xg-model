"""Split train/test: por partido (group), temporal o por competición (holdout de competiciones enteras)."""
import logging

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

log = logging.getLogger(__name__)


def split_shots(df, split_cfg):
    strategy = split_cfg.get("strategy", "group")
    if strategy == "temporal":
        cutoff = pd.Timestamp(split_cfg["temporal_cutoff"])
        is_test = df["match_date"] >= cutoff
        train, test = df[~is_test], df[is_test]
    elif strategy == "competition":
        test_comps = split_cfg.get("test_competitions") or []
        if not test_comps:
            raise ValueError("split.strategy=competition requiere split.test_competitions")
        unknown = set(test_comps) - set(df["competition_name"].unique())
        if unknown:
            log.warning("Competiciones de test no presentes en los datos: %s", sorted(unknown))
        is_test = df["competition_name"].isin(test_comps)
        train, test = df[~is_test], df[is_test]
    elif strategy == "group":
        gss = GroupShuffleSplit(n_splits=1, test_size=split_cfg.get("test_size", 0.2),
                                random_state=split_cfg.get("random_state", 42))
        tr_idx, te_idx = next(gss.split(df, groups=df["match_id"]))
        train, test = df.iloc[tr_idx], df.iloc[te_idx]
    else:
        raise ValueError(f"Estrategia de split desconocida: {strategy}")

    for name, part in (("train", train), ("test", test)):
        log.info("%s: %d tiros | %d partidos | tasa gol %.2f%%",
                 name, len(part), part["match_id"].nunique(), 100 * part["goal"].mean())
    if test.empty or train.empty:
        raise ValueError("Split vacío: revisa test_size / temporal_cutoff / test_competitions")
    return train.reset_index(drop=True), test.reset_index(drop=True)
