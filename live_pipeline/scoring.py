"""
scoring.py — the two models -> the strategy's three percentile scores, exactly
as in the research sims (development_notebooks/strategy.ipynb, cell "MODELS + DATA").

  combo_pct = percentile of P_move * P_dir   (buy gate: combo_pct >= confidence_threshold)
  dir_pct   = percentile of P_dir            (signal exit, off by default; dashboard)
  move_pct  = percentile of P_move           (signal exit, off by default; dashboard)

Percentiles are taken within the TUNE-split predictions saved with the models
(ref_*.npy), never within live data, so a threshold means the same thing in
the research and live.

Files in model_dir (default artifacts/, written by the export cell of strategy.ipynb):
  move.ubj, direction.ubj                          the two XGBoost models; each file
                                                   carries its feature list and order
  ref_move.npy, ref_direction.npy, ref_combo.npy   the tune predictions (P_move,
                                                   P_dir, P_move * P_dir)
"""
from pathlib import Path

import numpy as np
import xgboost as xgb


def to_percentile(values, reference):
    """Share of reference values <= x, clipped to [1e-6, 1] (identical to the research code)."""
    return _percentile_sorted(values, np.sort(np.asarray(reference, dtype=np.float64)))


def _percentile_sorted(values, sorted_ref):
    """to_percentile for a reference that is already sorted (no re-sort every minute)."""
    pct = np.searchsorted(sorted_ref, np.asarray(values, dtype=np.float64), side='right') / len(sorted_ref)
    return np.clip(pct, 1e-6, 1.0)


def _load_model(path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (run the export cell of strategy.ipynb "
                                f"and copy its files into the model folder)")
    m = xgb.Booster()
    m.load_model(str(path))
    if not m.feature_names:
        raise ValueError(f"{path} has no feature names; the research models always carry them")
    return m


def _load_reference(path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (run the export cell of strategy.ipynb)")
    ref = np.sort(np.load(path).astype(np.float64))
    if ref.size == 0 or not np.isfinite(ref).all():
        raise ValueError(f"{path} is empty or contains non-finite values")
    return ref


def _iteration_range(model):
    """(0, best_iteration + 1) as in the research; all trees if no early stopping was recorded."""
    try:
        return (0, int(model.best_iteration) + 1)
    except (AttributeError, TypeError, ValueError):
        return (0, 0)


class LiveScorer:
    def __init__(self, model_dir):
        d = Path(model_dir)
        self.move = _load_model(d / "move.ubj")
        self.dir = _load_model(d / "direction.ubj")
        self.move_features = list(self.move.feature_names)
        self.dir_features = list(self.dir.feature_names)
        self._move_range = _iteration_range(self.move)
        self._dir_range = _iteration_range(self.dir)
        self.ref_move = _load_reference(d / "ref_move.npy")
        self.ref_dir = _load_reference(d / "ref_direction.npy")
        self.ref_combo = _load_reference(d / "ref_combo.npy")

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
        return {'raw_move': pm, 'raw_dir': pdir,
                'combo_pct': _percentile_sorted(pm * pdir, self.ref_combo),
                'dir_pct': _percentile_sorted(pdir, self.ref_dir),
                'move_pct': _percentile_sorted(pm, self.ref_move)}
