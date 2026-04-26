from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.nonparametric.kernel_regression import KernelReg


class KernelSmoothingResolver:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.plot_path = self.base_dir / "kernel_smoothing_plot.png"
        self.domain = (-2.0, 4.0)
        self.sigma = 0.2
        self.n_samples = 50
        self.random_state = 42
        self.result = None
        self.results = []
        np.random.seed(self.random_state)

    @staticmethod
    def f(x):
        return np.exp(-(x**2)) - np.exp(-((x - 2) ** 2) / 8)

    @staticmethod
    def format_number(value, digits=4):
        if isinstance(value, (int, np.integer)):
            return str(value)
        return f"{float(value):.{digits}f}"

    def generate_sample(self):
        x = np.sort(np.random.uniform(self.domain[0], self.domain[1], self.n_samples))
        y_true = self.f(x)
        y = y_true + np.random.normal(0.0, self.sigma, self.n_samples)
        return x, y

    def build_model(self, x, y, kernel_name, reg_type):
        return KernelReg(
            endog=y,
            exog=[x],
            var_type="c",  # continuous predictor x
            reg_type=reg_type,
            bw="cv_ls",  # least-squares cross-validation for bandwidth selection
            ckertype=kernel_name,  # kernel type for continuous predictors
        )

    def fit_kernel_smoother(self, x, y):
        # kernels задаёт форму весов в ядерном сглаживании.
        # reg_types задаёт тип локальной аппроксимации:
        # lc = local constant, ll = local linear.
        kernels = ["gaussian", "tricube"]
        reg_types = ["lc", "ll"]

        best = None
        self.results = []
        for kernel_name in kernels:
            for reg_type in reg_types:
                model = self.build_model(x, y, kernel_name, reg_type)
                estimator = (
                    model._est_loc_constant if reg_type == "lc" else model._est_loc_linear
                )
                cv_mse = float(np.asarray(model.cv_loo(model.bw, estimator)).item())
                train_pred, _ = model.fit([x])

                candidate = {
                    "kernel": kernel_name,
                    "reg_type": reg_type,
                    "bandwidth": float(model.bw[0]),
                    "cv_mse": cv_mse,
                    "train_mse": float(np.mean((y - train_pred) ** 2)),
                    "model": model,
                }
                self.results.append(candidate)
                if best is None or candidate["cv_mse"] < best["cv_mse"]:
                    best = candidate

        self.result = best

    def make_plot(self, x, y):
        grid_x = np.linspace(self.domain[0], self.domain[1], 400)
        reg_type_label = {"lc": "local_constant", "ll": "local_linear"}
        colors = {
            ("gaussian", "lc"): "darkorange",
            ("gaussian", "ll"): "red",
            ("tricube", "lc"): "green",
            ("tricube", "ll"): "purple",
        }

        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, color="black", s=25, label="выборка")
        plt.plot(grid_x, self.f(grid_x), linewidth=2, label="истинная f(x)")
        for candidate in self.results:
            grid_pred, _ = candidate["model"].fit([grid_x])
            plt.plot(
                grid_x,
                grid_pred,
                linewidth=2,
                color=colors[(candidate["kernel"], candidate["reg_type"])],
                label=(
                    f"{candidate['kernel']}, "
                    f"{reg_type_label[candidate['reg_type']]}, "
                    f"h={self.format_number(candidate['bandwidth'])}"
                ),
            )

        plt.title("Сравнение конфигураций ядерного сглаживания")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=150)
        plt.close()

    def resolve(self):
        x, y = self.generate_sample()
        self.fit_kernel_smoother(x, y)
        self.make_plot(x, y)

        reg_type_label = {"lc": "local_constant", "ll": "local_linear"}

        print("\n#########################\n")
        print("Задание 5. Пункт 1")
        print("Оценка функции методом ядерного сглаживания")
        print(
            "Область определения: "
            f"({self.format_number(self.domain[0])}, {self.format_number(self.domain[1])})"
        )
        print(f"Сигма: {self.format_number(self.sigma)}")
        print(f"Размер выборки: {self.n_samples}")

        print("\nВсе конфигурации")
        for candidate in self.results:
            print(
                f"{candidate['kernel']}, {reg_type_label[candidate['reg_type']]}: "
                f"bandwidth = {self.format_number(candidate['bandwidth'])}, "
                f"LOOCV MSE = {self.format_number(candidate['cv_mse'])}, "
                f"train MSE = {self.format_number(candidate['train_mse'])}"
            )

        print("\nЛучшая конфигурация")
        print(f"Ядро: {self.result['kernel']}")
        print(f"Локальная аппроксимация: {reg_type_label[self.result['reg_type']]}")
        print(f"Ширина окна: {self.format_number(self.result['bandwidth'])}")
        print(
            "Средняя квадратичная ошибка Leave-One-Out Cross-Validation: "
            f"{self.format_number(self.result['cv_mse'])}"
        )
        print(
            "Средняя квадратичная ошибка на обучении: "
            f"{self.format_number(self.result['train_mse'])}"
        )
        print(f"График сохранён в: {self.plot_path}")


if __name__ == "__main__":
    KernelSmoothingResolver().resolve()
