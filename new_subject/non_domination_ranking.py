import numpy as np


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """
    True, если a доминирует b.
    Предполагается минимизация всех критериев.
    """
    return np.all(a <= b) and np.any(a < b)


def non_dominated_ranking(Y: np.ndarray):
    """
    Non-dominated sorting.

    Parameters
    ----------
    Y : np.ndarray, shape (n_points, n_objectives)
        Значения критериев.

    Returns
    -------
    fronts : list[list[int]]
        fronts[0] = индексы F1,
        fronts[1] = индексы F2, ...
    
    ranks : np.ndarray
        ranks[i] = номер фронта точки i, начиная с 1.
    """
    n = len(Y)

    dominates_list = [[] for _ in range(n)]
    domination_count = np.zeros(n, dtype=int)

    fronts = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            if dominates(Y[i], Y[j]):
                dominates_list[i].append(j)

            elif dominates(Y[j], Y[i]):
                domination_count[i] += 1

        if domination_count[i] == 0:
            fronts[0].append(i)

    ranks = np.zeros(n, dtype=int)

    current_rank = 1
    k = 0

    while k < len(fronts) and len(fronts[k]) > 0:
        next_front = []

        for i in fronts[k]:
            ranks[i] = current_rank

            for j in dominates_list[i]:
                domination_count[j] -= 1

                if domination_count[j] == 0:
                    next_front.append(j)

        current_rank += 1
        k += 1

        if next_front:
            fronts.append(next_front)

    return fronts, ranks

Y = np.array([
    [1, 5],
    [2, 3],
    [3, 2],
    [2, 4],
    [4, 4],
])

fronts, ranks = non_dominated_ranking(Y)

print("Fronts:", fronts)
print("Ranks:", ranks)