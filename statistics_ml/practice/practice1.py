import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

N = 50
X = np.random.uniform(0, 4, N)
noise = np.random.normal(0, 0.1, N)
y = X * np.log(X) - 0.4 * X**2  + noise


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


def gaussian_process_predict(X_train, y_train, X_test, kernel, noise_variance=1e-6, **kernel_params):
    K = kernel(X_train, X_train, **kernel_params)   + noise_variance * np.eye(len(X_train))
    K_s = kernel(X_train, X_test, **kernel_params)
    K_ss = kernel(X_test, X_test, **kernel_params) # + 1e-8 * np.eye(len(X_test))

    alpha = np.linalg.solve(K, y_train)
    mu = K_s.T @ alpha
    Sigma = K_ss - K_s.T @ np.linalg.solve(K, K_s)

    return mu, Sigma


if __name__ == "__main__":
    t = 1.0  # tau
    b = 1.3  # bias
    noise_variance = 0.01
    np.set_printoptions(precision=3, suppress=True)

    print(_pairwise_distances(x1=np.array([1, 0, 5]), x2=np.array([2, 3, 4])))

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

    X_train = X
    y_train = y
    X_test = np.linspace(0.0, 4.0, 1000)
    y_true = X_test * np.log(np.clip(X_test, 1e-8, None)) - 0.4 * X_test**2

    kernels = [
        ("RBF kernel", rbf_kernel, {"t": t}),
        ("Exponential kernel", exponential_kernel, {"t": t}),
        ("Linear kernel", linear_kernel, {}),
        ("Linear bias kernel", linear_bias_kernel, {"d": b}),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)

    for ax, (title, kernel, kernel_params) in zip(axes.flat, kernels):
        mu, Sigma = gaussian_process_predict(
            X_train,
            y_train,
            X_test,
            kernel,
            noise_variance=noise_variance,
            **kernel_params,
        )
        std = np.sqrt(np.maximum(np.diag(Sigma), 0.0))

        ax.scatter(X_train, y_train, color="black", s=20, label="Train data", zorder=3)
        ax.plot(X_test, y_true, color="tab:green", linestyle="--", label="True function")
        ax.plot(X_test, mu, color="tab:blue", label="GP mean")
        ax.fill_between(
            X_test,
            mu - 1.96 * std,
            mu + 1.96 * std,
            color="tab:blue",
            alpha=0.2,
            label="95% confidence interval",
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
