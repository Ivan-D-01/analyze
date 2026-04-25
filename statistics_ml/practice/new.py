# применить гауссовский процесс
# K = {k(xi,xj)} i,j = [1;n]
# k* = {k(x,xi)} i = [1;n]


import numpy as np

np.random.seed(42)

N = 4
X = np.random.uniform(0, 4, N)
noise = np.random.normal(0, 0.1, N)
y = X * np.log(X) - 0.4 * X**2 + noise


def _pairwise_distances(x1, x2):
    return np.abs(x1[:, None] - x2[None, :])


def rbf_kernel(x1, x2, t):
    distances = _pairwise_distances(x1, x2)
    return np.exp(-(distances**2) / (2 * t**2))


def exponential_kernel(x1, x2, t):
    distances = _pairwise_distances(x1, x2)
    return np.exp(-distances / t)


def linear_kernel(x1, x2):
    return x1[:, None] * x2[None, :]


def linear_bias_kernel(x1, x2, d):
    return 1.0 + linear_kernel(x1, x2) ** d


if __name__ == "__main__":
    t = 1.0 # tau
    b = 1.3 # bias
    np.set_printoptions(precision=3, suppress=True)
    
    print(_pairwise_distances(x1 = np.array([1,0,5]), x2 = np.array([2,3,4])))

    print('data')
    print(f'{X}\n')
    rbf_covariance = rbf_kernel(X, X, t)
    exponential_covariance = exponential_kernel(X, X, t)
    linear_covariance = linear_kernel(X, X)
    linear_bias_covariance = linear_bias_kernel(X, X, b)

    print("RBF covariance matrix:")
    print(rbf_covariance)
    print()

    print("Exponential covariance matrix:")
    print(exponential_covariance)
    print()

    print("Linear covariance matrix:")
    print(linear_covariance)
    print()

    print("Linear bias covariance matrix:")
    print(linear_bias_covariance)
