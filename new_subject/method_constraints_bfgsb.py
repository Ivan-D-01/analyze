import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def calculate_y(x: np.ndarray) -> np.ndarray:
    x1, x2 = x
    return np.array((x1, 1 + x2 - np.sqrt(x1) * np.sqrt(1 + x2)))

def objective_with_penalty(x: np.ndarray, eps: float, penalty_weight: float = 10_000.0) -> float:
    y1, y2 = calculate_y(x)
    violation = max(0.0, y2 - eps)
    return float(y1 + penalty_weight * violation**2)


def solve_by_constraints_bfgsb(eps_values: np.ndarray, x0: np.ndarray | None = None):
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    start_point = np.array([0.5, 0.5]) if x0 is None else np.array(x0, dtype=float)
    solutions = []

    for eps in eps_values:
        result = minimize(
            objective_with_penalty,
            x0=start_point,
            args=(eps,),
            method="L-BFGS-B",
            bounds=bounds,
        )

        if not result.success:
            continue

        x_best = result.x
        y_best = calculate_y(x_best)

        if y_best[1] <= eps + 1e-6:
            solutions.append((eps, x_best, y_best))
            start_point = x_best

    return solutions

eps_values = np.linspace(0.0, 1.0, 20)
solutions = solve_by_constraints_bfgsb(eps_values)

for eps, x, y in solutions:
    print(f"epsilon: {eps:.4f}; x = ({x[0]:.4f}, {x[1]:.4f}); y = ({y[0]:.4f}, {y[1]:.4f})")

curve_x1 = np.linspace(0.0, 1.0, 400)
curve_x2 = np.linspace(0.0, 1.0, 400)
curve_grid = np.array([(a, b) for a in curve_x1 for b in curve_x2])
curve_y = np.array([calculate_y(x) for x in curve_grid])

solution_y = np.array([y for _, _, y in solutions])

plt.scatter(curve_y[:, 0], curve_y[:, 1], s=10, alpha=0.2, label="All solutions")
if len(solution_y) > 0:
    plt.scatter(solution_y[:, 0], solution_y[:, 1], color="red", label="Constraint method + L-BFGS-B")
plt.xlabel("y1")
plt.ylabel("y2")
plt.legend()
plt.show()
