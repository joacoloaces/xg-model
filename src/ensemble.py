"""Ensemble por soft voting: promedia predict_proba de modelos ya entrenados."""
import numpy as np


class SoftVotingEnsemble:
    def __init__(self, models):
        self.models = dict(models)

    @property
    def names(self):
        return list(self.models)

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.models.values()], axis=0)
        return np.column_stack([1 - p, p])


def ensemble_name(members):
    return "ens_" + "+".join(members)


def members_of(name):
    return name.removeprefix("ens_").split("+") if name.startswith("ens_") else [name]
