import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def calculate_y(x: np.ndarray) -> np.ndarray:
    x1, x2 = x
    return np.array((
        x1,
        1 + x2 - np.sqrt(x1) * np.sqrt(1 + x2)
    ))


def weighted_objective(x: np.ndarray, weights: np.ndarray) -> float:
    y = calculate_y(x)
    return float(np.dot(weights, y))


def box_barrier(x: np.ndarray, mu: float) -> float:
    if np.any(x <= 0) or np.any(x >= 1):
        return np.inf

    return -mu * np.sum(np.log(x) + np.log(1 - x))


def barrier_objective(x: np.ndarray, weights: np.ndarray, mu: float) -> float:
    return weighted_objective(x, weights) + box_barrier(x, mu)


alphas = np.linspace(0, 1, 101)

solutions_x = []
solutions_y = []
solutions_w = []

mu0 = 1e-1
mu_factor = 0.3
n_barrier_steps = 8

for alpha in alphas:
    weights = np.array([alpha, 1 - alpha])

    x = np.array([0.5, 0.5])

    success = True

    for k in range(n_barrier_steps):
        mu = mu0 * (mu_factor ** k)

        res = minimize(
            fun=lambda z: barrier_objective(z, weights, mu),
            x0=x,
            method="L-BFGS-B",
            bounds=[(1e-12, 1 - 1e-12), (1e-12, 1 - 1e-12)]
        )

        if not res.success:
            success = False
            break

        x = res.x

    if success:
        x_star = x
        y_star = calculate_y(x_star)

        solutions_x.append(x_star)
        solutions_y.append(y_star)
        solutions_w.append(weights)


solutions_x = np.array(solutions_x)
solutions_y = np.array(solutions_y)
solutions_w = np.array(solutions_w)

print(solutions_y)
plt.scatter(solutions_y[:, 0], solutions_y[:, 1])
plt.xlabel("y1")
plt.ylabel("y2")
plt.grid()
plt.show()