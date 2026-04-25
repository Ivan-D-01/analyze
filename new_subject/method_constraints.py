import numpy as np
import matplotlib.pyplot as plt


def calculate_y(x1, x2) -> np.ndarray:
    return np.array((x1, 1 + x2 - np.sqrt(x1) * np.sqrt(1 + x2)))


def solve_by_constraints(grid: np.ndarray, y_values: np.ndarray, eps_values: np.ndarray):
    solutions = []

    for eps in eps_values:
        feasible_mask = y_values[:, 1] <= eps
        feasible_x = grid[feasible_mask]
        feasible_y = y_values[feasible_mask]

        if len(feasible_y) == 0:
            continue

        best_idx = np.argmin(feasible_y[:, 0])
        solutions.append((eps, feasible_x[best_idx], feasible_y[best_idx]))

    return solutions


N = 100
x1 = np.linspace(0, 1, N)
x2 = np.linspace(0, 1, N)

grid = np.array([(a, b) for a in x1 for b in x2])
Y = np.array([calculate_y(x1, x2) for x1, x2 in grid])

eps_values = np.linspace(np.min(Y[:, 1]), np.max(Y[:, 1]), 20)
solutions = solve_by_constraints(grid, Y, eps_values)

for eps, x, y in solutions:
    print(f"epsilon: {eps:.4f}; x = ({x[0]:.4f}, {x[1]:.4f}); y = ({y[0]:.4f}, {y[1]:.4f})")

solution_y = np.array([y for _, _, y in solutions])

plt.scatter(Y[:, 0], Y[:, 1], s=10, alpha=0.25, label="All solutions")
plt.scatter(solution_y[:, 0], solution_y[:, 1], color="red", label="Constraint method")
plt.xlabel("y1")
plt.ylabel("y2")
plt.legend()
plt.show()
