import numpy as np

def sphere(x):
    return np.sum(x ** 2)

def grasshopper_optimization(obj_func, dim, n_grasshoppers, max_iter, lb, ub):
    # Инициализация
    grasshoppers = np.random.uniform(lb, ub, (n_grasshoppers, dim))
    fitness = np.array([obj_func(g) for g in grasshoppers])
    best_idx = np.argmin(fitness)
    target = grasshoppers[best_idx].copy()
    leader_fitness = fitness[best_idx]

    convergence_curve = np.zeros(max_iter)
    for t in range(max_iter):
        c = 0.00001 + (1 - 0.00001) * (max_iter - t) / max_iter
        
        for i in range(n_grasshoppers):
            social_effect = np.zeros(dim)
            neighbors = grasshoppers[np.random.choice(n_grasshoppers, size=5, replace=False)]
            
            for j in range(len(neighbors)):
                if i != j:
                    r = np.linalg.norm(neighbors[j] - grasshoppers[i])
                    s = 0.5 * np.exp(-r / 1.5) - np.exp(-r)  # Функция s(r)
                    social_effect += s * (neighbors[j] - grasshoppers[i]) / (r + 1e-10)
            
            grasshoppers[i] = c * social_effect + target
            grasshoppers[i] = np.clip(grasshoppers[i], lb, ub)
        
        fitness = np.array([obj_func(g) for g in grasshoppers])
        new_best_idx = np.argmin(fitness)
        if fitness[new_best_idx] < leader_fitness:
            target = grasshoppers[new_best_idx].copy()
            leader_fitness = fitness[new_best_idx]

        convergence_curve[t] = leader_fitness
        if t % 50 == 0:
            print(f"Iteration {t}: Best Fitness = {leader_fitness}")
            #print(f"Iteration {t}: Curve = {convergence_curve}")
            
    return target, leader_fitness, convergence_curve

if __name__ == "__main__":
    
    dim = 10      
    n_grasshoppers = 10  
    max_iter = 500
    lb = -10      
    ub = 10       

    best_solution, best_fitness, _ = grasshopper_optimization(
        sphere, dim, n_grasshoppers, max_iter, lb, ub
    )

    print("\n--- Results ---")
    print(f"Best solution: {best_solution}")
    print(f"Best function value: {best_fitness}")