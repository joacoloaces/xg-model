import logging
from pathlib import Path

import yaml


def load_config(path="config.yaml"):
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, value in cfg["paths"].items():
        p = Path(value).expanduser()
        if not p.is_absolute():
            p = path.parent / p
        p.mkdir(parents=True, exist_ok=True)
        cfg["paths"][key] = p
    return cfg


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
