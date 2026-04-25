from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class ExpantionsResolver:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.plot_path = self.base_dir / "polynomial_ridge_plot.png"
        self.domain = (-2.0, 4.0)
        self.sigma = 0.2
        self.n_samples = 50
        self.random_state = 42
        self.result = None
        np.random.seed(self.random_state)

    @staticmethod
    def f(x):
        return np.exp(-(x**2)) - np.exp(-((x - 2) ** 2) / 8)

    def generate_sample(self):
        x = np.sort(np.random.uniform(self.domain[0], self.domain[1], self.n_samples))
        y_true = self.f(x)
        y = y_true + np.random.normal(0.0, self.sigma, self.n_samples)
        return x.reshape(-1, 1), y

    def cv_mse(self, degree, alpha, x, y):
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scores = []
        for train_idx, test_idx in kf.split(x):
            model = Pipeline(
                [
                    ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                    #("scale", StandardScaler()),
                    ("ridge", Ridge(alpha=alpha)),
                ]
            )
            model.fit(x[train_idx], y[train_idx])
            pred = model.predict(x[test_idx])
            scores.append(mean_squared_error(y[test_idx], pred))
        return float(np.mean(scores))

    def fit_polynomial_ridge(self, x, y):
        best = None
        for degree in range(1, 13):
            for alpha in np.logspace(-6, 3, 25):
                score = self.cv_mse(degree, alpha, x, y)
                if best is None or score < best["cv_mse"]:
                    best = {"degree": degree, "alpha": alpha, "cv_mse": score}

        model = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=best["degree"], include_bias=False)),
                #("scale", StandardScaler()),
                ("ridge", Ridge(alpha=best["alpha"])),
            ]
        )
        model.fit(x, y)
        self.result = best
        return model

    def make_plot(self, x, y, model):
        grid_x = np.linspace(self.domain[0], self.domain[1], 400).reshape(-1, 1)
        plt.figure(figsize=(10, 6))
        plt.scatter(x[:, 0], y, color="black", s=25, label="sample")
        plt.plot(grid_x[:, 0], self.f(grid_x[:, 0]), linewidth=2, label="true f(x)")
        plt.plot(grid_x[:, 0], model.predict(grid_x), linewidth=2, label="polynomial ridge")
        plt.title("Polynomial Ridge Regression")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=150)
        plt.close()

    def resolve(self):
        x, y = self.generate_sample()
        model = self.fit_polynomial_ridge(x, y)
        self.make_plot(x, y, model)
        print("Task 4. Point 1")
        print(f"Domain: {self.domain}")
        print(f"Sigma: {self.sigma}")
        print(f"N: {self.n_samples}")
        print(f"Plot saved to: {self.plot_path}")
        print("Best polynomial ridge model")
        print(f"Degree: {self.result['degree']}")
        print(f"Alpha: {self.result['alpha']}")
        print(f"CV MSE: {self.result['cv_mse']}")
        print(f"Degrees of freedom (approx): {self.result['degree'] + 1}")


if __name__ == "__main__":
    ExpantionsResolver().resolve()
