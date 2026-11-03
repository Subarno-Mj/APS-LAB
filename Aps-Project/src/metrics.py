"""Probability-quality metrics used across the project."""
import numpy as np


def expected_calibration_error(y_true_idx, proba, n_bins: int = 15) -> float:
    """ECE on top-label confidence (multi-class) or on P(class 1) (binary when proba is 1-D)."""
    proba = np.asarray(proba)
    y_true_idx = np.asarray(y_true_idx)
    if proba.ndim == 1:  # binary: reliability of P(y=1)
        conf, correct = proba, y_true_idx.astype(float)
    else:
        conf = proba.max(1)
        correct = (proba.argmax(1) == y_true_idx).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def multiclass_brier(y_true_idx, proba) -> float:
    proba = np.asarray(proba)
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(proba)), np.asarray(y_true_idx)] = 1
    return float(((proba - onehot) ** 2).sum(1).mean())


def normalized_entropy(proba) -> np.ndarray:
    """Predictive entropy scaled to [0, 1] (1 = uniform = maximally uncertain)."""
    proba = np.clip(np.asarray(proba), 1e-12, 1)
    k = proba.shape[1]
    return -(proba * np.log(proba)).sum(1) / np.log(k)


def reliability_curve(y, p, n_bins: int = 10):
    """Return (mean predicted, observed frequency, count) per bin for binary p."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    mp, of, cnt = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            mp.append(p[m].mean())
            of.append(np.asarray(y)[m].mean())
            cnt.append(int(m.sum()))
    return np.array(mp), np.array(of), np.array(cnt)
