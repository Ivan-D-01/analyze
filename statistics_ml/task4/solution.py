from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler


class ExpantionsResolver:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.plot_path = self.base_dir / "polynomial_ridge_plot.png"
        self.domain = (-2.0, 4.0)
        self.sigma = 0.2
        self.n_samples = 50
        self.random_state = 42
        self.polynomial_result = None
        self.spline_result = None
        self.smoothed_spline_result = None
        self.fixed_knots = np.array([-2.0, -0.5, 1.0, 2.5, 4.0]).reshape(-1, 1)
        np.random.seed(self.random_state)

    @staticmethod
    def f(x):
        return np.exp(-(x**2)) - np.exp(-((x - 2) ** 2) / 8)

    def generate_sample(self):
        x = np.sort(np.random.uniform(self.domain[0], self.domain[1], self.n_samples))
        y_true = self.f(x)
        y = y_true + np.random.normal(0.0, self.sigma, self.n_samples)
        return x.reshape(-1, 1), y

    def fit_polynomial_ridge(self, x, y):
        pipeline = Pipeline(
            [
                ("poly", PolynomialFeatures(include_bias=False)),
                ("scale", StandardScaler()),
                ("ridge", Ridge()),
            ]
        )
        param_grid = {
            "poly__degree": list(range(1, 13)),
            "ridge__alpha": np.logspace(-6, 3, 25),
        }
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=5,
            n_jobs=1,
        )
        search.fit(x, y)

        self.polynomial_result = {
            "degree": search.best_params_["poly__degree"],
            "alpha": search.best_params_["ridge__alpha"],
            "cv_mse": -search.best_score_,
        }
        return search.best_estimator_

    def fit_fixed_knot_cubic_spline(self, x, y):
        model = Pipeline(
            [
                (
                    "spline",
                    SplineTransformer(
                        degree=3,
                        knots=self.fixed_knots,
                        include_bias=False,
                        extrapolation="continue",
                    ),
                ),
                ("ridge", Ridge(alpha=0.0))
            ]
        )
        model.fit(x, y)
        pred = model.predict(x)
        self.spline_result = {
            "knots": self.fixed_knots.ravel().tolist(),
            "train_mse": float(np.mean((y - pred) ** 2)),
            "degrees_of_freedom": int(model.named_steps["spline"].n_features_out_ + 1),
        }
        return model

    def fit_smoothed_fixed_knot_cubic_spline(self, x, y):
        pipeline = Pipeline(
            [
                (
                    "spline",
                    SplineTransformer(
                        degree=3,
                        knots=self.fixed_knots,
                        include_bias=False,
                        extrapolation="continue",
                    ),
                ),
                ("ridge", Ridge()),
            ]
        )
        param_grid = {"ridge__alpha": np.logspace(-6, 3, 25)}
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=5,
            n_jobs=1,
        )
        search.fit(x, y)

        best_model = search.best_estimator_
        pred = best_model.predict(x)
        self.smoothed_spline_result = {
            "knots": self.fixed_knots.ravel().tolist(),
            "alpha": search.best_params_["ridge__alpha"],
            "cv_mse": -search.best_score_,
            "train_mse": float(np.mean((y - pred) ** 2)),
            "degrees_of_freedom": int(best_model.named_steps["spline"].n_features_out_ + 1),
        }
        return best_model

    def make_plot(self, x, y, polynomial_model, spline_model, smoothed_spline_model):
        grid_x = np.linspace(self.domain[0], self.domain[1], 400).reshape(-1, 1)
        plt.figure(figsize=(10, 6))
        plt.scatter(x[:, 0], y, color="black", s=25, label="sample")
        plt.scatter(
            self.fixed_knots[:, 0],
            self.f(self.fixed_knots[:, 0]),
            color="red",
            marker="x",
            s=80,
            linewidths=2,
            label="spline knots",
        )
        plt.plot(grid_x[:, 0], self.f(grid_x[:, 0]), linewidth=2, label="true f(x)")
        plt.plot(
            grid_x[:, 0],
            polynomial_model.predict(grid_x),
            linewidth=2,
            color="green",
            label="polynomial ridge",
        )
        plt.plot(
            grid_x[:, 0],
            spline_model.predict(grid_x),
            linewidth=2,
            color="orange",
            label="fixed-knot cubic spline",
        )
        plt.plot(
            grid_x[:, 0],
            smoothed_spline_model.predict(grid_x),
            linewidth=2,
            color="red",
            label="smoothed fixed-knot cubic spline",
        )
        plt.title("Polynomial Ridge And Fixed-Knot Cubic Spline")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=150)
        plt.close()

    def resolve(self):
        x, y = self.generate_sample()
        polynomial_model = self.fit_polynomial_ridge(x, y)
        spline_model = self.fit_fixed_knot_cubic_spline(x, y)
        smoothed_spline_model = self.fit_smoothed_fixed_knot_cubic_spline(x, y)
        self.make_plot(x, y, polynomial_model, spline_model, smoothed_spline_model)

        print("\n#########################\n")
        print("Task 4. Point 1")
        print(f"Domain: {self.domain}")
        print(f"Sigma: {self.sigma}")
        print(f"N: {self.n_samples}")
        print(f"Plot saved to: {self.plot_path}")
        print("Best polynomial ridge model")
        print(f"Degree: {self.polynomial_result['degree']}")
        print(f"Alpha: {self.polynomial_result['alpha']}")
        print(f"CV MSE: {self.polynomial_result['cv_mse']}")
        print(f"Degrees of freedom (approx): {self.polynomial_result['degree'] + 1}")

        print("\n#########################\n")
        print("Task 4. Point 2")
        print("Fixed-knot cubic spline without smoothing")
        print(f"Knots: {self.spline_result['knots']}")
        print(f"Train MSE: {self.spline_result['train_mse']}")
        print(f"Degrees of freedom (approx): {self.spline_result['degrees_of_freedom']}")

        print("\n#########################\n")
        print("Task 4. Point 3")
        print("Fixed-knot cubic spline with smoothing")
        print(f"Knots: {self.smoothed_spline_result['knots']}")
        print(f"Alpha: {self.smoothed_spline_result['alpha']}")
        print(f"CV MSE: {self.smoothed_spline_result['cv_mse']}")
        print(f"Train MSE: {self.smoothed_spline_result['train_mse']}")
        print(
            "Degrees of freedom (approx): "
            f"{self.smoothed_spline_result['degrees_of_freedom']}"
        )


if __name__ == "__main__":
    ExpantionsResolver().resolve()
