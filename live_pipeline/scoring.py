"""
scoring.py — the two models -> the scores the strategy and the dashboard use.

  combo     = P_move x P_dir   (buy gate: combo >= combo_cutoff)
  raw_dir   = P_dir            (dashboard)
  raw_move  = P_move           (dashboard)

The research thresholds are percentiles within the tune-split predictions
(strategy.ipynb). The live bot does not need those predictions: a percentile
only grows with the raw value, so "combo percentile >= confidence_threshold"
is exactly "combo >= combo_cutoff", where combo_cutoff is the tune combo at that
rank. strategy.ipynb computes combo_cutoff once; it is hard-coded in strategy.py.

Files in model_dir (default artifacts/, copied from the live_artifacts/ folder of strategy.ipynb):
  move.ubj, direction.ubj    the two XGBoost models; each file carries its feature list and order
  strategy_<name>.json       the strategy tuned on them (strategy.py), with their fingerprints
"""
import hashlib
from pathlib import Path

import numpy as np
import xgboost as xgb


def _load_model(path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (copy the models written by the export cell "
                                f"of strategy.ipynb into the model folder)")
    m = xgb.Booster()
    m.load_model(str(path))
    if not m.feature_names:
        raise ValueError(f"{path} has no feature names; the research models always carry them")
    return m


def _iteration_range(model):
    """(0, best_iteration + 1) as in the research; all trees if no early stopping was recorded."""
    try:
        return (0, int(model.best_iteration) + 1)
    except (AttributeError, TypeError, ValueError):
        return (0, 0)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify_models(model_dir, strategy):
    """The strategy's combo_cutoff is a score of the models it was tuned on: raise unless
    move.ubj and direction.ubj are exactly those models (by sha256)."""
    d = Path(model_dir)
    for key, fname in (('move', 'move.ubj'), ('direction', 'direction.ubj')):
        want = strategy['models'][key]['sha256']
        have = file_sha256(d / fname)
        if have != want:
            raise ValueError(f"{d / fname} is not the {key} model strategy '{strategy.get('name')}' was tuned on "
                             f"(sha256 {have[:12]}..., expected {want[:12]}...); copy the strategy file together "
                             f"with its models from strategy.ipynb's live_artifacts/")


class LiveScorer:
    def __init__(self, model_dir):
        d = Path(model_dir)
        self.move = _load_model(d / "move.ubj")
        self.dir = _load_model(d / "direction.ubj")
        self.move_features = list(self.move.feature_names)
        self.dir_features = list(self.dir.feature_names)
        self._move_range = _iteration_range(self.move)
        self._dir_range = _iteration_range(self.dir)

    @staticmethod
    def _predict(model, feats, iteration_range, df):
        missing = [c for c in feats if c not in df.columns]
        if missing:
            raise KeyError(f"feature frame lacks {len(missing)} model features, e.g. {missing[:5]}")
        return model.predict(xgb.DMatrix(df[feats], feature_names=feats), iteration_range=iteration_range)

    def score(self, feature_df):
        """feature_df: rows from features.compute_live_features. Returns a dict of numpy arrays."""
        pm = self._predict(self.move, self.move_features, self._move_range, feature_df)
        pdir = self._predict(self.dir, self.dir_features, self._dir_range, feature_df)
        # The product in the models' own precision (float32), exactly as the
        # research computes it before ranking; compared in float64.
        combo = (pm * pdir).astype(np.float64)
        return {'raw_move': pm.astype(np.float64), 'raw_dir': pdir.astype(np.float64), 'combo': combo}
