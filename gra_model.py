"""Grey Relational Analysis (GRA) utilities.

This file includes direction transformation, min-max standardization,
entropy weights, grey relational coefficients/grades, and a heatmap helper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def positive_transform(x, types, targets=None, intervals=None):
    """Transform all criteria into benefit type: larger is better.

    Parameters
    ----------
    x : array-like, shape (m, n)
        Raw decision matrix. Rows are alternatives, columns are criteria.
    types : list[str]
        Each item is one of: 'max', 'min', 'target', 'interval'.
    targets : dict[int, float]
        Target values for target-type criteria.
    intervals : dict[int, tuple[float, float]]
        Best intervals [a, b] for interval-type criteria.
    """
    x = np.asarray(x, dtype=float)
    z = x.copy()
    targets = targets or {}
    intervals = intervals or {}

    for j, t in enumerate(types):
        col = x[:, j]
        if t == "max":
            z[:, j] = col
        elif t == "min":
            z[:, j] = np.max(col) - col
        elif t == "target":
            a = targets[j]
            d = np.abs(col - a)
            max_d = np.max(d)
            z[:, j] = 1.0 if max_d < EPS else 1.0 - d / max_d
        elif t == "interval":
            a, b = intervals[j]
            d = np.zeros_like(col)
            d[col < a] = a - col[col < a]
            d[col > b] = col[col > b] - b
            max_d = np.max(d)
            z[:, j] = 1.0 if max_d < EPS else 1.0 - d / max_d
        else:
            raise ValueError(f"Unknown criterion type: {t}")
    return z


def minmax_standardize(z):
    """Min-max standardization by column into [0, 1]."""
    z = np.asarray(z, dtype=float)
    z_min = np.min(z, axis=0)
    z_max = np.max(z, axis=0)
    denom = z_max - z_min
    r = np.zeros_like(z)
    valid = denom > EPS
    r[:, valid] = (z[:, valid] - z_min[valid]) / denom[valid]
    return r, valid


def entropy_weight(r):
    """Calculate entropy weights from a standardized matrix."""
    r = np.asarray(r, dtype=float)
    m, n = r.shape
    r_safe = r + EPS
    p = r_safe / np.sum(r_safe, axis=0, keepdims=True)
    e = -np.sum(p * np.log(p), axis=0) / np.log(m)
    g = 1.0 - e
    if np.sum(g) < EPS:
        w = np.ones(n) / n
    else:
        w = g / np.sum(g)
    return e, g, w


def grey_relational_analysis(r, weights=None, rho=0.5, reference="ideal"):
    """Run Grey Relational Analysis.

    Parameters
    ----------
    r : array-like, shape (m, n)
        Standardized matrix after direction transformation.
    weights : array-like, optional
        Criterion weights. If omitted, equal weights are used.
    rho : float
        Distinguishing coefficient in (0, 1]. Commonly set to 0.5.
    reference : 'ideal' or array-like
        If 'ideal', uses the best value in each criterion as reference.
        Otherwise, provide a custom reference sequence of length n.

    Returns
    -------
    dict with reference, delta, coefficients, grades, and ranks.
    """
    r = np.asarray(r, dtype=float)
    m, n = r.shape
    if not (0 < rho <= 1):
        raise ValueError("rho must be in (0, 1].")

    if reference == "ideal":
        x0 = np.max(r, axis=0)
    else:
        x0 = np.asarray(reference, dtype=float)
        if x0.shape != (n,):
            raise ValueError("reference must have length n.")

    delta = np.abs(x0 - r)
    delta_min = np.min(delta)
    delta_max = np.max(delta)
    coef = (delta_min + rho * delta_max) / (delta + rho * delta_max + EPS)

    if weights is None:
        w = np.ones(n) / n
    else:
        w = np.asarray(weights, dtype=float)
        if np.any(w < 0) or np.sum(w) <= 0:
            raise ValueError("weights must be non-negative with positive sum.")
        w = w / np.sum(w)

    grades = coef @ w
    ranks = pd.Series(grades).rank(ascending=False, method="min").astype(int).to_numpy()
    return {
        "reference": x0,
        "delta": delta,
        "coefficients": coef,
        "weights": w,
        "grades": grades,
        "ranks": ranks,
    }


def grade_label(score):
    """Example grade label for a grey relational grade in [0, 1]."""
    if score >= 0.80:
        return "优秀/强关联"
    if score >= 0.70:
        return "良好/较强关联"
    if score >= 0.60:
        return "中等/一般关联"
    if score >= 0.50:
        return "较弱关联"
    return "弱关联"


if __name__ == "__main__":
    alternatives = ["A方案", "B方案", "C方案", "D方案"]
    criteria = ["收益", "成本", "稳定性", "响应时间"]
    data = np.array([
        [80, 30, 0.70, 12],
        [75, 20, 0.90, 15],
        [90, 35, 0.60, 10],
        [70, 25, 0.85, 11],
    ])
    types = ["max", "min", "max", "min"]

    z = positive_transform(data, types)
    r, valid = minmax_standardize(z)
    e, g, w_entropy = entropy_weight(r)
    gra = grey_relational_analysis(r, weights=w_entropy, rho=0.5, reference="ideal")

    weight_table = pd.DataFrame({
        "指标": criteria,
        "熵值": e,
        "差异系数": g,
        "熵权": w_entropy,
    })
    result_table = pd.DataFrame({
        "方案": alternatives,
        "灰色关联度": gra["grades"],
        "排序": gra["ranks"],
        "等级": [grade_label(v) for v in gra["grades"]],
    }).sort_values("排序")

    print(weight_table.round(4))
    print(result_table.round(4))
