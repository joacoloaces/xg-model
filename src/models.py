"""Registro de modelos y espacios de búsqueda de hiperparámetros."""
import logging

from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

log = logging.getLogger(__name__)


def build_preprocessor(num_cols, cat_cols):
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False)
    return ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)])


def _logreg(rs):
    model = LogisticRegression(solver="liblinear", max_iter=2000, random_state=rs)
    params = {"C": loguniform(1e-3, 1e2), "l1_ratio": [0.0, 1.0]}
    return model, params


def _random_forest(rs):
    model = RandomForestClassifier(n_jobs=1, random_state=rs)
    params = {
        "n_estimators": randint(200, 600),
        "max_depth": [6, 8, 10, 12, None],
        "min_samples_leaf": randint(10, 150),
        "max_features": ["sqrt", 0.3, 0.5],
    }
    return model, params


def _xgboost(rs):
    from xgboost import XGBClassifier
    model = XGBClassifier(eval_metric="logloss", tree_method="hist", n_jobs=1, random_state=rs)
    params = {
        "n_estimators": randint(200, 800),
        "max_depth": randint(3, 7),
        "learning_rate": loguniform(0.01, 0.2),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.5, 0.5),
        "min_child_weight": randint(1, 30),
        "reg_lambda": loguniform(1e-2, 10),
    }
    return model, params


def _lightgbm(rs):
    from lightgbm import LGBMClassifier
    model = LGBMClassifier(n_jobs=1, random_state=rs, verbose=-1)
    params = {
        "n_estimators": randint(200, 800),
        "num_leaves": randint(8, 64),
        "learning_rate": loguniform(0.01, 0.2),
        "min_child_samples": randint(20, 200),
        "subsample": uniform(0.6, 0.4),
        "subsample_freq": [1],
        "colsample_bytree": uniform(0.5, 0.5),
        "reg_lambda": loguniform(1e-2, 10),
    }
    return model, params


def _catboost(rs):
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(verbose=0, thread_count=1, random_seed=rs, allow_writing_files=False)
    params = {
        "iterations": randint(300, 1000),
        "depth": randint(4, 8),
        "learning_rate": loguniform(0.01, 0.2),
        "l2_leaf_reg": loguniform(1, 10),
    }
    return model, params


def _mlp(rs):
    model = MLPClassifier(max_iter=500, early_stopping=True, n_iter_no_change=15, random_state=rs)
    params = {
        "hidden_layer_sizes": [
            (32,), (64,), (128,),
            (64, 32), (128, 64), (256, 128),
            (128, 64, 32), (256, 128, 64),
            (256, 128, 64, 32),
        ],
        "activation": ["relu", "tanh"],
        "alpha": loguniform(1e-5, 1e-1),
        "learning_rate_init": loguniform(1e-4, 1e-2),
        "batch_size": [128, 256, 512],
    }
    return model, params


REGISTRY = {
    "logreg": _logreg,
    "random_forest": _random_forest,
    "xgboost": _xgboost,
    "lightgbm": _lightgbm,
    "catboost": _catboost,
    "mlp": _mlp,
}


def get_models(names, random_state=42):
    models = {}
    for name in names:
        if name not in REGISTRY:
            log.warning("Modelo desconocido '%s', se omite", name)
            continue
        try:
            models[name] = REGISTRY[name](random_state)
        except ImportError as e:
            log.warning("No se pudo importar '%s' (%s), se omite", name, e)
    return models
