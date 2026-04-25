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


bounds = [(0, 1), (0, 1)]

# набор весов для двух критериев: w1 + w2 = 1
alphas = np.linspace(0, 1, 101)

solutions_x = []
solutions_y = []
solutions_w = []

for alpha in alphas:
    weights = np.array([alpha, 1 - alpha])

    res = minimize(
        fun=lambda x: weighted_objective(x, weights),
        x0=np.array([0.5, 0.5]),
        method="L-BFGS-B",
        bounds=bounds
    )

    if res.success:
        x_star = res.x
        y_star = calculate_y(x_star)

        solutions_x.append(x_star)
        solutions_y.append(y_star)
        solutions_w.append(weights)

solutions_x = np.array(solutions_x)
solutions_y = np.array(solutions_y)
solutions_w = np.array(solutions_w)


plt.scatter(solutions_y[:, 0], solutions_y[:, 1])
plt.xlabel("y1")
plt.ylabel("y2")
plt.grid()
plt.show()