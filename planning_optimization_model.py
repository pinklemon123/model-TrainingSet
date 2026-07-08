
"""Common optimization model templates.
Run sections independently depending on the model type.
"""
import numpy as np
from scipy.optimize import linprog, minimize, differential_evolution


def solve_lp():
    """Linear programming example."""
    c = np.array([3, 5])
    A_ub = np.array([[2, 1], [1, 2]])
    b_ub = np.array([100, 80])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(0, None), (0, None)], method="highs")
    return res


def solve_nlp():
    """Constrained nonlinear programming example."""
    def f(v):
        x, y = v
        return (x - 1) ** 2 + (y - 2) ** 2
    cons = [
        {"type": "ineq", "fun": lambda v: v[0] + v[1] - 2},
        {"type": "ineq", "fun": lambda v: 10 - v[0] ** 2 - v[1] ** 2},
    ]
    return minimize(f, x0=np.array([0.0, 0.0]), constraints=cons, method="SLSQP")


def monte_carlo_inventory(seed=0):
    """Monte Carlo simulation for inventory decision."""
    rng = np.random.default_rng(seed)
    demand = np.maximum(0, rng.normal(100, 20, size=20000))
    price, cost, salvage = 12, 7, 2

    def profit(Q):
        sales = np.minimum(Q, demand)
        leftover = np.maximum(Q - demand, 0)
        p = price * sales + salvage * leftover - cost * Q
        return p.mean(), np.percentile(p, 5)

    qs = np.arange(50, 151)
    records = [(Q, *profit(Q)) for Q in qs]
    return max(records, key=lambda r: r[1])


def global_optimize():
    """Global optimization by differential evolution."""
    def rastrigin(x):
        x = np.asarray(x)
        return 20 + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
    bounds = [(-5.12, 5.12), (-5.12, 5.12)]
    return differential_evolution(rastrigin, bounds, seed=0)


def weighted_goal_programming():
    """Weighted goal programming with deviation variables."""
    # Variables: x1, x2, d1_minus, d1_plus, d2_minus, d2_plus
    c = np.array([0, 0, 5, 0, 0, 2])
    A_eq = np.array([
        [6, 8, 1, -1, 0, 0],
        [2, 3, 0, 0, 1, -1],
    ])
    b_eq = np.array([100, 40])
    return linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * 6, method="highs")


if __name__ == "__main__":
    lp = solve_lp()
    print("LP:", lp.success, lp.fun, lp.x)
    nlp = solve_nlp()
    print("NLP:", nlp.success, nlp.fun, nlp.x)
    print("Monte Carlo:", monte_carlo_inventory())
    de = global_optimize()
    print("Global:", de.fun, de.x)
    gp = weighted_goal_programming()
    print("Goal programming:", gp.success, gp.fun, gp.x)
