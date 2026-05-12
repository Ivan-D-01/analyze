from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score
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
        self.comparison_result = None
        self.fixed_knots = np.array([-2.0, -0.5, 1.0, 2.5, 4.0]).reshape(-1, 1)
        self.percent_of_x_knots = 0.5
        np.random.seed(self.random_state)

    @staticmethod
    def f(x):
        return np.exp(-(x**2)) - np.exp(-((x - 2) ** 2) / 8)

    @staticmethod
    def format_number(value, digits=4):
        if isinstance(value, (int, np.integer)):
            return str(value)
        return f"{float(value):.{digits}f}"

    def format_sequence(self, values, digits=4):
        return "[" + ", ".join(self.format_number(value, digits) for value in values) + "]"

    @staticmethod
    def mse(y_true, y_pred):
        return float(np.mean((y_true - y_pred) ** 2))

    def generate_sample(self):
        x = np.sort(np.random.uniform(self.domain[0], self.domain[1], self.n_samples))
        y_true = self.f(x)
        y = y_true + np.random.normal(0.0, self.sigma, self.n_samples)
        return x.reshape(-1, 1), y

    def compute_cv_mse(self, model, x, y):
        scores = cross_val_score(
            clone(model),
            x,
            y,
            scoring="neg_mean_squared_error",
            cv=5,
            n_jobs=1,
        )
        return float(-scores.mean())

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
            "cv_mse": float(-search.best_score_),
            "train_mse": self.mse(y, search.best_estimator_.predict(x)),
            "degrees_of_freedom": search.best_params_["poly__degree"] + 1,
        }
        return search.best_estimator_

    def fit_fixed_knot_cubic_spline(self, x, y):
        model = Pipeline(
            [
                (
                    "spline",
                    SplineTransformer( # превращает исходный x в набор сплайновых базисных функций
                        degree=3,
                        knots=self.fixed_knots,
                        include_bias=False,
                    ),
                ),
                ("ridge", Ridge(alpha=0.0)), #  подбирает коэффициенты при этих базисных функциях
            ] 
        )
        model.fit(x, y)
        pred = model.predict(x)
        self.spline_result = {
            "knots": self.fixed_knots.ravel().tolist(),
            "cv_mse": self.compute_cv_mse(model, x, y), # просто для сравнения добавлена для 6 пункта
            "train_mse": self.mse(y, pred),
            "degrees_of_freedom": int(model.named_steps["spline"].n_features_out_ + 1),
            # + 1 так как в Ridge(alpha=0.0) fit_intercept: bool = True,
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
            "cv_mse": float(-search.best_score_),
            "train_mse": self.mse(y, pred),
            "degrees_of_freedom": int(best_model.named_steps["spline"].n_features_out_ + 1),
        }
        return best_model

    def fit_smoothed_x_knot_cubic_spline(self, x: np.ndarray, y: np.ndarray):
        x_knots = np.unique(
            np.sort(
                np.random.choice(
                    x[:, 0],
                    size=int(len(x) * self.percent_of_x_knots),
                    replace=False,
                )
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
            "cv_mse": float(-search.best_score_),
            "train_mse": self.mse(y, pred),
            "degrees_of_freedom": int(best_model.named_steps["spline"].n_features_out_ + 1),
        }
        return best_model

    def build_comparison(self):
        rows = [
            {
                "name": "Полиномиальный Ridge",
                "cv_mse": self.polynomial_result["cv_mse"],
                "train_mse": self.polynomial_result["train_mse"],
                "degrees_of_freedom": self.polynomial_result["degrees_of_freedom"],
            },
            {
                "name": "Фиксированный кубический сплайн без сглаживания",
                "cv_mse": self.spline_result["cv_mse"],
                "train_mse": self.spline_result["train_mse"],
                "degrees_of_freedom": self.spline_result["degrees_of_freedom"],
            },
            {
                "name": "Фиксированный кубический сплайн со сглаживанием",
                "cv_mse": self.smoothed_spline_result["cv_mse"],
                "train_mse": self.smoothed_spline_result["train_mse"],
                "degrees_of_freedom": self.smoothed_spline_result["degrees_of_freedom"],
            },
            {
                "name": "Сглаживающий сплайн по наблюдаемым x",
                "cv_mse": self.smoothed_x_knot_spline_result["cv_mse"],
                "train_mse": self.smoothed_x_knot_spline_result["train_mse"],
                "degrees_of_freedom": self.smoothed_x_knot_spline_result["degrees_of_freedom"],
            },
        ]
        best_by_cv = min(rows, key=lambda row: row["cv_mse"])
        min_df = min(rows, key=lambda row: row["degrees_of_freedom"])
        self.comparison_result = {
            "rows": rows,
            "best_by_cv": best_by_cv,
            "min_df": min_df,
        }

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
            label="узлы фиксированного сплайна",
        )
        plt.plot(grid_x[:, 0], self.f(grid_x[:, 0]), linewidth=2, label="истинная f(x)")
        plt.plot(
            grid_x[:, 0],
            polynomial_model.predict(grid_x),
            linewidth=2,
            color="green",
            label="полиномиальный Ridge",
        )
        plt.plot(
            grid_x[:, 0],
            spline_model.predict(grid_x),
            linewidth=2,
            color="orange",
            label="фиксированный кубический сплайн",
        )
        plt.plot(
            grid_x[:, 0],
            smoothed_spline_model.predict(grid_x),
            linewidth=2,
            color="red",
            label="сглаженный фиксированный кубический сплайн",
        )
        plt.plot(
            grid_x[:, 0],
            smoothed_x_knots_spline_model.predict(grid_x),
            linewidth=2,
            color="purple",
            label="сглаженный сплайн по наблюдаемым x",
        )
        plt.title("Сравнение аппроксимаций")
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
        self.build_comparison()
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
        print(
            "Область определения: "
            f"({self.format_number(self.domain[0])}, {self.format_number(self.domain[1])})"
        )
        print(f"Сигма: {self.format_number(self.sigma)}")
        print(f"Размер выборки: {self.n_samples}")
        print(f"График сохранён в: {self.plot_path}")
        print(f"График базисных функций сохранён в: {self.basis_plot_path}")
        print("Лучшая полиномиальная Ridge-регрессия")
        print(f"Степень: {self.polynomial_result['degree']}")
        print(f"Параметр alpha: {self.format_number(self.polynomial_result['alpha'])}")
        print(
            "Средняя квадратичная ошибка CV: "
            f"{self.format_number(self.polynomial_result['cv_mse'])}"
        )
        print(
            "Средняя квадратичная ошибка на обучении: "
            f"{self.format_number(self.polynomial_result['train_mse'])}"
        )
        print(
            "Число степеней свободы: "
            f"{self.polynomial_result['degrees_of_freedom']}"
        )

        print("\n#########################\n")
        print("Задание 4. Пункт 2")
        print("Кубический сплайн с фиксированными узлами без сглаживания")
        print(f"Узлы: {self.format_sequence(self.spline_result['knots'])}")
        print(
            "Средняя квадратичная ошибка CV: "
            f"{self.format_number(self.spline_result['cv_mse'])}"
        )
        print(
            "Средняя квадратичная ошибка на обучении: "
            f"{self.format_number(self.spline_result['train_mse'])}"
        )
        print(
            "Число степеней свободы: "
            f"{self.spline_result['degrees_of_freedom']}"
        )

        print("\n#########################\n")
        print("Задание 4. Пункт 3")
        print("Кубический сплайн с фиксированными узлами и сглаживанием")
        print(f"Узлы: {self.format_sequence(self.smoothed_spline_result['knots'])}")
        print(f"Параметр alpha: {self.format_number(self.smoothed_spline_result['alpha'])}")
        print(
            "Средняя квадратичная ошибка CV: "
            f"{self.format_number(self.smoothed_spline_result['cv_mse'])}"
        )
        print(
            "Средняя квадратичная ошибка на обучении: "
            f"{self.format_number(self.smoothed_spline_result['train_mse'])}"
        )
        print(
            "Число степеней свободы: "
            f"{self.smoothed_spline_result['degrees_of_freedom']}"
        )

        print("\n#########################\n")
        print("Задание 4. Пункт 4")
        print("Построены базисные функции для кубического сплайна с фиксированными узлами")
        print(f"Узлы: {self.format_sequence(self.fixed_knots.ravel().tolist())}")
        print(f"График базисных функций сохранён в: {self.basis_plot_path}")

        print("\n#########################\n")
        print("Задание 4. Пункт 5")
        print("Сглаживающий кубический сплайн с узлами, равными наблюдаемым x")
        print(f"Узлы: {self.format_sequence(self.smoothed_x_knot_spline_result['knots'])}")
        print(
            "Параметр alpha: "
            f"{self.format_number(self.smoothed_x_knot_spline_result['alpha'])}"
        )
        print(
            "Средняя квадратичная ошибка CV: "
            f"{self.format_number(self.smoothed_x_knot_spline_result['cv_mse'])}"
        )
        print(
            "Средняя квадратичная ошибка на обучении: "
            f"{self.format_number(self.smoothed_x_knot_spline_result['train_mse'])}"
        )
        print(
            "Число степеней свободы: "
            f"{self.smoothed_x_knot_spline_result['degrees_of_freedom']}"
        )

        print("\n#########################\n")
        print("Задание 4. Пункт 6")
        print("Сравнение моделей по числу степеней свободы и точности")
        for row in self.comparison_result["rows"]:
            print(
                f"{row['name']}: CV MSE = {self.format_number(row['cv_mse'])}, "
                f"train MSE = {self.format_number(row['train_mse'])}, "
                f"степени свободы = {row['degrees_of_freedom']}"
            )
        print(
            "Лучшая модель по CV: "
            f"{self.comparison_result['best_by_cv']['name']} "
            f"(CV MSE = {self.format_number(self.comparison_result['best_by_cv']['cv_mse'])})"
        )
        print(
            "Модель с минимальным числом степеней свободы: "
            f"{self.comparison_result['min_df']['name']} "
            f"(степени свободы = {self.comparison_result['min_df']['degrees_of_freedom']})"
        )


if __name__ == "__main__":
    ExpantionsResolver().resolve()
