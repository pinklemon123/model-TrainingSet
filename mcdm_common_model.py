"""
Unified examples for TOPSIS, FCE and entropy weighting.
Save as mcdm_common_model.py and run with Python 3.
"""
import numpy as np

EPS = 1e-12


def positive_transform(X, types, targets=None, intervals=None):
    """Convert all criteria to benefit type: larger is better.

    types: list with values 'max', 'min', 'mid', 'interval'.
    targets: dict {j: target} for middle-type criteria.
    intervals: dict {j: (a, b)} for interval-type criteria.
    """
    X = np.asarray(X, dtype=float)
    Z = X.copy()
    targets = targets or {}
    intervals = intervals or {}
    for j, t in enumerate(types):
        col = X[:, j]
        if t == "max":
            Z[:, j] = col
        elif t == "min":
            Z[:, j] = np.max(col) - col
        elif t == "mid":
            a = targets[j]
            d = np.abs(col - a)
            Z[:, j] = 1.0 if np.max(d) < EPS else 1 - d / np.max(d)
        elif t == "interval":
            a, b = intervals[j]
            d = np.zeros_like(col)
            d[col < a] = a - col[col < a]
            d[col > b] = col[col > b] - b
            Z[:, j] = 1.0 if np.max(d) < EPS else 1 - d / np.max(d)
        else:
            raise ValueError(f"Unknown criterion type: {t}")
    return Z


def minmax_standardize(Z):
    """Scale each column into [0, 1]. This does not make column sums equal to 1."""
    Z = np.asarray(Z, dtype=float)
    zmin, zmax = np.min(Z, axis=0), np.max(Z, axis=0)
    denom = zmax - zmin
    R = np.zeros_like(Z)
    ok = denom > EPS
    R[:, ok] = (Z[:, ok] - zmin[ok]) / denom[ok]
    return R


def vector_normalize(Z):
    """Vector normalization, often used in TOPSIS."""
    Z = np.asarray(Z, dtype=float)
    norm = np.sqrt(np.sum(Z ** 2, axis=0))
    return Z / np.where(norm < EPS, 1.0, norm)


def entropy_weight(R):
    """Calculate entropy values and entropy weights from a non-negative matrix."""
    R = np.asarray(R, dtype=float)
    m, n = R.shape
    P = (R + EPS) / np.sum(R + EPS, axis=0, keepdims=True)
    e = -np.sum(P * np.log(P), axis=0) / np.log(m)
    g = 1 - e
    w = np.ones(n) / n if np.sum(g) < EPS else g / np.sum(g)
    return e, g, w


def topsis(X, types, weights=None):
    """Entropy-weighted TOPSIS by default. Returns closeness coefficients."""
    Z = positive_transform(X, types)
    R = vector_normalize(Z)
    if weights is None:
        _, _, weights = entropy_weight(minmax_standardize(Z))
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    V = R * weights
    ideal_best = np.max(V, axis=0)
    ideal_worst = np.min(V, axis=0)
    d_best = np.sqrt(np.sum((V - ideal_best) ** 2, axis=1))
    d_worst = np.sqrt(np.sum((V - ideal_worst) ** 2, axis=1))
    closeness = d_worst / (d_best + d_worst + EPS)
    return closeness, weights


def fce(weights, membership_matrix, grade_scores=None):
    """Single-level fuzzy comprehensive evaluation."""
    W = np.asarray(weights, dtype=float)
    W = W / W.sum()
    R = np.asarray(membership_matrix, dtype=float)
    B = W @ R
    B_norm = B / B.sum() if B.sum() > EPS else B
    result = {"B": B_norm, "grade_index": int(np.argmax(B_norm))}
    if grade_scores is not None:
        result["score"] = float(B_norm @ np.asarray(grade_scores, dtype=float))
    return result


if __name__ == "__main__":
    # Three alternatives, four criteria. Types: benefit, cost, benefit, benefit.
    X = np.array([
        [85,  12, 70, 0.80],
        [90,  16, 65, 0.75],
        [78,  10, 80, 0.88],
    ])
    types = ["max", "min", "max", "max"]
    c, w = topsis(X, types)
    print("Entropy weights:", np.round(w, 4))
    print("TOPSIS closeness:", np.round(c, 4))

    # FCE example: 4 criteria, 5 grades.
    R = np.array([
        [0.70, 0.20, 0.10, 0.00, 0.00],
        [0.30, 0.40, 0.20, 0.10, 0.00],
        [0.50, 0.30, 0.20, 0.00, 0.00],
        [0.20, 0.40, 0.30, 0.10, 0.00],
    ])
    res = fce([0.25, 0.25, 0.25, 0.25], R, grade_scores=[100, 80, 60, 40, 20])
    print("FCE vector:", np.round(res["B"], 4), "score:", round(res["score"], 2))
