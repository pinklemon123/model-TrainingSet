"""TOPSIS, Entropy Weight TOPSIS, AHP weights, and Entropy-FCE utilities.
Run: python topsis_model.py
"""
import numpy as np
import pandas as pd

EPS = 1e-12

def positive_transform(X, types, targets=None, intervals=None):
    """Transform criteria to benefit type: larger means better."""
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
            md = np.max(d)
            Z[:, j] = 1.0 if md < EPS else 1.0 - d / md
        elif t == "interval":
            a, b = intervals[j]
            d = np.zeros_like(col)
            d[col < a] = a - col[col < a]
            d[col > b] = col[col > b] - b
            md = np.max(d)
            Z[:, j] = 1.0 if md < EPS else 1.0 - d / md
        else:
            raise ValueError(f"Unknown criterion type: {t}")
    return Z

def minmax_standardize(Z):
    """Column-wise min-max standardization into [0, 1]."""
    Z = np.asarray(Z, dtype=float)
    mn = Z.min(axis=0)
    mx = Z.max(axis=0)
    den = mx - mn
    R = np.zeros_like(Z)
    ok = den > EPS
    R[:, ok] = (Z[:, ok] - mn[ok]) / den[ok]
    return R

def vector_standardize(Z):
    """Column-wise vector normalization used by classical TOPSIS."""
    Z = np.asarray(Z, dtype=float)
    den = np.sqrt((Z ** 2).sum(axis=0))
    R = np.zeros_like(Z)
    ok = den > EPS
    R[:, ok] = Z[:, ok] / den[ok]
    return R

def entropy_weights(R):
    """Objective entropy weights. R should be non-negative."""
    R = np.asarray(R, dtype=float)
    m, n = R.shape
    P = (R + EPS) / (R + EPS).sum(axis=0, keepdims=True)
    e = -(P * np.log(P)).sum(axis=0) / np.log(m)
    g = 1.0 - e
    if g.sum() < EPS:
        w = np.ones(n) / n
    else:
        w = g / g.sum()
    return e, g, w

def ahp_weights(A):
    """Eigenvector AHP weights and Saaty consistency ratio."""
    A = np.asarray(A, dtype=float)
    vals, vecs = np.linalg.eig(A)
    k = np.argmax(vals.real)
    lam_max = vals[k].real
    w = np.abs(vecs[:, k].real)
    w = w / w.sum()
    n = A.shape[0]
    RI = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24,
          7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    CI = (lam_max - n) / (n - 1) if n > 1 else 0.0
    CR = 0.0 if RI.get(n, 1.49) == 0 else CI / RI.get(n, 1.49)
    return w, lam_max, CI, CR

def topsis(Z, weights, standardize="vector"):
    """TOPSIS on benefit-type matrix Z."""
    Z = np.asarray(Z, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    R = vector_standardize(Z) if standardize == "vector" else minmax_standardize(Z)
    V = R * w
    pos = V.max(axis=0)
    neg = V.min(axis=0)
    D_pos = np.sqrt(((V - pos) ** 2).sum(axis=1))
    D_neg = np.sqrt(((V - neg) ** 2).sum(axis=1))
    C = D_neg / (D_pos + D_neg + EPS)
    return C, D_pos, D_neg, R, V

def entropy_topsis(X, types):
    """Positive transform -> min-max for entropy -> vector TOPSIS."""
    Z = positive_transform(X, types)
    R_entropy = minmax_standardize(Z)
    e, g, w = entropy_weights(R_entropy)
    C, Dp, Dm, R, V = topsis(Z, w, standardize="vector")
    return e, g, w, C, Dp, Dm

def fce_entropy(weights, membership, labels=None, scores=None):
    """Fuzzy comprehensive evaluation with entropy/AHP/other weights."""
    W = np.asarray(weights, dtype=float)
    W = W / W.sum()
    R = np.asarray(membership, dtype=float)
    # Optional row normalization if memberships are from overlapping functions.
    row_sum = R.sum(axis=1, keepdims=True)
    R = np.divide(R, row_sum, out=np.zeros_like(R), where=row_sum > EPS)
    B = W @ R
    B_norm = B / B.sum() if B.sum() > EPS else B
    idx = int(np.argmax(B_norm))
    out = {"B": B_norm, "index": idx}
    if labels is not None:
        out["label"] = labels[idx]
    if scores is not None:
        out["score"] = float(B_norm @ np.asarray(scores, dtype=float))
    return out

if __name__ == "__main__":
    alternatives = ["A1", "A2", "A3", "A4", "A5"]
    criteria = ["benefit", "cost", "risk", "satisfaction", "potential"]
    X = np.array([
        [82, 68, 3.2, 78, 70],
        [76, 55, 4.1, 85, 82],
        [90, 72, 2.8, 74, 76],
        [68, 48, 3.5, 80, 88],
        [84, 60, 2.9, 90, 79],
    ], dtype=float)
    types = ["max", "min", "min", "max", "max"]
    e, g, w, C, Dp, Dm = entropy_topsis(X, types)
    weight_table = pd.DataFrame({
        "criterion": criteria, "entropy": e, "diversity": g, "weight": w
    })
    result = pd.DataFrame({
        "alternative": alternatives, "D+": Dp, "D-": Dm, "C": C
    })
    result["rank"] = result["C"].rank(ascending=False, method="min").astype(int)
    print(weight_table.round(4))
    print(result.sort_values("rank").round(4))
