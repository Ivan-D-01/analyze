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
        self.basis_plot_path = self.base_dir / "cubic_spline_basis_plot.png"
        self.domain = (-2.0, 4.0)
        self.sigma = 0.2
        self.n_samples = 50
        self.random_state = 42
        self.polynomial_result = None
        self.spline_result = None
        self.smoothed_spline_result = None
        self.smoothed_x_knot_spline_result = None
        self.fixed_knots = np.array([-2.0, -0.5, 1.0, 2.5, 4.0]).reshape(-1, 1)
        self.percent_of_x_knots = 0.5
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
                (
                    "scale",
                    StandardScaler(),
                ),  # Добавлено из-за LinAlgWarning: ill-conditioned matrix.
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
                    ),
                ),
                ("ridge", Ridge(alpha=0.0)),
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

    def fit_smoothed_x_knot_cubic_spline(self, x: np.ndarray, y: np.ndarray):
        x_knots = np.unique(
            np.sort(
                np.random.choice(x[:, 0], size=int(len(x) * self.percent_of_x_knots), replace=False)
            )
        ).reshape(-1, 1)

        pipeline = Pipeline(
            [
                (
                    "spline",
                    SplineTransformer(
                        degree=3,
                        knots=x_knots,
                        include_bias=False,
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
        self.smoothed_x_knot_spline_result = {
            "knots": x_knots.ravel().tolist(),
            "alpha": search.best_params_["ridge__alpha"],
            "cv_mse": -search.best_score_,
            "train_mse": float(np.mean((y - pred) ** 2)),
            "degrees_of_freedom": int(best_model.named_steps["spline"].n_features_out_ + 1),
        }
        return best_model

    def make_plot(
        self,
        x,
        y,
        polynomial_model,
        spline_model,
        smoothed_spline_model,
        smoothed_x_knots_spline_model,
    ):
        grid_x = np.linspace(self.domain[0], self.domain[1], 400).reshape(-1, 1)
        plt.figure(figsize=(10, 6))
        plt.scatter(x[:, 0], y, color="black", s=25, label="выборка")
        plt.scatter(
            self.fixed_knots[:, 0],
            self.f(self.fixed_knots[:, 0]),
            color="red",
            marker="x",
            s=80,
            linewidths=2,
            label="узлы сплайна",
        )
        plt.plot(grid_x[:, 0], self.f(grid_x[:, 0]), linewidth=2, label="истинная f(x)")
        plt.plot(
            grid_x[:, 0],
            polynomial_model.predict(grid_x),
            linewidth=2,
            color="green",
            label="полиномиальный ридж",
        )
        plt.plot(
            grid_x[:, 0],
            spline_model.predict(grid_x),
            linewidth=2,
            color="orange",
            label="кубический сплайн с фиксированными узлами",
        )
        plt.plot(
            grid_x[:, 0],
            smoothed_spline_model.predict(grid_x),
            linewidth=2,
            color="red",
            label="сглаживающий кубический сплайн с фиксированными узлами",
        )
        plt.plot(
            grid_x[:, 0],
            smoothed_x_knots_spline_model.predict(grid_x),
            linewidth=2,
            color="purple",
            label="сглаживающий кубический сплайн по наблюдаемым x",
        )
        plt.title("Полиномиальный ридж и кубические сплайны")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=150)
        plt.close()

    def make_basis_plot(self, spline_model):
        grid_x = np.linspace(self.domain[0], self.domain[1], 400).reshape(-1, 1)
        basis = spline_model.named_steps["spline"].transform(grid_x)

        plt.figure(figsize=(10, 6))
        for idx in range(basis.shape[1]):
            plt.plot(grid_x[:, 0], basis[:, idx], linewidth=2, label=f"базис {idx + 1}")
        for knot in self.fixed_knots[:, 0]:
            plt.axvline(knot, color="gray", linestyle="--", linewidth=1)
        plt.scatter(
            self.fixed_knots[:, 0],
            np.zeros_like(self.fixed_knots[:, 0]),
            color="black",
            s=30,
            label="узлы",
        )
        plt.title("Базисные функции кубического сплайна с фиксированными узлами")
        plt.xlabel("x")
        plt.ylabel("значение базисной функции")
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(self.basis_plot_path, dpi=150)
        plt.close()

    def resolve(self):
        x, y = self.generate_sample()
        polynomial_model = self.fit_polynomial_ridge(x, y)
        spline_model = self.fit_fixed_knot_cubic_spline(x, y)
        smoothed_spline_model = self.fit_smoothed_fixed_knot_cubic_spline(x, y)
        smoothed_x_knots_spline_model = self.fit_smoothed_x_knot_cubic_spline(x, y)
        self.make_plot(
            x,
            y,
            polynomial_model,
            spline_model,
            smoothed_spline_model,
            smoothed_x_knots_spline_model,
        )
        self.make_basis_plot(spline_model)

        print("\n#########################\n")
        print("Задание 4. Пункт 1")
        print(f"Область определения: {self.domain}")
        print(f"Сигма: {self.sigma}")
        print(f"Размер выборки: {self.n_samples}")
        print(f"График сохранён в: {self.plot_path}")
        print(f"График базисных функций сохранён в: {self.basis_plot_path}")
        print("Лучшая полиномиальная ридж-регрессия")
        print(f"Степень: {self.polynomial_result['degree']}")
        print(f"Параметр alpha: {self.polynomial_result['alpha']}")
        print(f"Средняя квадратичная ошибка CV: {self.polynomial_result['cv_mse']}")
        print(
            "Число степеней свободы (приблизительно): "
            f"{self.polynomial_result['degree'] + 1}"
        )

        print("\n#########################\n")
        print("Задание 4. Пункт 2")
        print("Кубический сплайн с фиксированными узлами без сглаживания")
        print(f"Узлы: {self.spline_result['knots']}")
        print(f"Средняя квадратичная ошибка на обучении: {self.spline_result['train_mse']}")
        print(
            "Число степеней свободы (приблизительно): "
            f"{self.spline_result['degrees_of_freedom']}"
        )

        print("\n#########################\n")
        print("Задание 4. Пункт 3")
        print("Кубический сплайн с фиксированными узлами и сглаживанием")
        print(f"Узлы: {self.smoothed_spline_result['knots']}")
        print(f"Параметр alpha: {self.smoothed_spline_result['alpha']}")
        print(f"Средняя квадратичная ошибка CV: {self.smoothed_spline_result['cv_mse']}")
        print(
            "Средняя квадратичная ошибка на обучении: "
            f"{self.smoothed_spline_result['train_mse']}"
        )
        print(
            "Число степеней свободы (приблизительно): "
            f"{self.smoothed_spline_result['degrees_of_freedom']}"
        )

        print("\n#########################\n")
        print("Задание 4. Пункт 4")
        print("Построены базисные функции для кубического сплайна с фиксированными узлами")
        print(f"Узлы: {self.fixed_knots.ravel().tolist()}")
        print(f"График базисных функций сохранён в: {self.basis_plot_path}")

        print("\n#########################\n")
        print("Задание 4. Пункт 5")
        print("Сглаживающий кубический сплайн с узлами, равными наблюдаемым x")
        print(f"Узлы: {self.smoothed_x_knot_spline_result['knots']}")
        print(f"Параметр alpha: {self.smoothed_x_knot_spline_result['alpha']}")
        print(f"Средняя квадратичная ошибка CV: {self.smoothed_x_knot_spline_result['cv_mse']}")
        print(
            "Средняя квадратичная ошибка на обучении: "
            f"{self.smoothed_x_knot_spline_result['train_mse']}"
        )
        print(
            "Число степеней свободы (приблизительно): "
            f"{self.smoothed_x_knot_spline_result['degrees_of_freedom']}"
        )


if __name__ == "__main__":
    ExpantionsResolver().resolve()
