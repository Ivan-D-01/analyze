import numpy as np
import matplotlib.pyplot as plt

def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)

def calculate_y(x1, x2) -> np.ndarray:
    return np.array((x1, 1+x2 - np.sqrt(x1) * np.sqrt(1+x2)))

bounds = [(0,1),(0,1)]

N = 100
x1 = np.linspace(0, 1, N)
x2 = np.linspace(0, 1, N)

grid = [(a, b) for a in x1 for b in x2]
Y = np.array([calculate_y(x1, x2) for (x1, x2) in grid])

for y, (x1, x2) in zip(Y, grid):
    print(f'x1: {x1}', f'x2: {x2}', f'y: {y}')


pareto_mask = np.ones(len(Y), dtype=bool)

for i in range(len(Y)):
    for j in range(len(Y)):
        if i != j and dominates(Y[j], Y[i]):
            pareto_mask[i] = False
            break

pareto_X = np.array(grid)[pareto_mask]
pareto_Y = Y[pareto_mask]

plt.scatter(pareto_Y[:,0], pareto_Y[:,1])