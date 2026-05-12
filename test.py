import numpy as np
def objective_raw(x: np.ndarray) -> float:
    x1, x2 = float(x[0]), float(x[1])
    return (x1**2 + x2 - 5) ** 2 + (x1 + x2**2 - 5) ** 2


#  - Adam: (1.791286863304, 1.791286863304)
#  - PSO: (1.791292173972, 1.791285138457)
#
#-2.791280311193, -2.791283640997

print(objective_raw(x=np.array([1.791292173972, 1.791285138457])))
print(objective_raw(x=np.array([-2.791280311193, -2.791283640997])))