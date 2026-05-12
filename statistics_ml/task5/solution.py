from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, Matern, RBF, WhiteKernel
from sklearn.model_selection import KFold, cross_val_score
from statsmodels.nonparametric.kernel_regression import KernelReg


class Task5Resolver:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.kernel_plot_path = self.base_dir / "kernel_smoothing_plot.png"
        self.gp_plot_path = self.base_dir / "gaussian_process_plot.png"
        self.domain = (-2.0, 4.0)
        self.sigma = 0.2
        self.n_samples = 50
        self.random_state = 42
        self.kernel_result = None
        self.kernel_results = []
        self.gp_result = None
        self.gp_results = []
        self.gp_top_results = {}
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

    def build_kernel_model(self, x, y, kernel_name, reg_type):
        return KernelReg(
            endog=y,
            exog=[x],
            var_type="c",
            reg_type=reg_type,
            # bw="cv_ls" означает автоматический подбор ширины окна:
            # statsmodels перебирает кандидаты и выбирает bandwidth,
            # минимизирующий leave-one-out CV критерий наименьших квадратов.
            bw="cv_ls",
            ckertype=kernel_name,
        )

    def fit_kernel_smoother(self, x, y):
        # gaussian использует веса, плавно убывающие с расстоянием от точки x:
        # дальние наблюдения влияют мало, но не обнуляются полностью.
        # tricube тоже сильнее взвешивает близкие точки, но делает это локальнее:
        # вне окна вклад наблюдений становится нулевым.

        # Для gaussian ядра:

        # K(u) = (1 / sqrt(2*pi)) * exp(-u^2 / 2)

        # Для tricube ядра:

        # K(u) = (1 - |u|^3)^3,  при |u| <= 1
        # K(u) = 0,               при |u| > 1

        kernels = ["gaussian", "tricube"]
        reg_types = ["lc", "ll"]

        best = None
        self.kernel_results = []
        for kernel_name in kernels:
            for reg_type in reg_types:
                model = self.build_kernel_model(x, y, kernel_name, reg_type)
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
                self.kernel_results.append(candidate)
                if best is None or candidate["cv_mse"] < best["cv_mse"]:
                    best = candidate

        self.kernel_result = best

    def make_kernel_plot(self, x, y):
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
        for candidate in self.kernel_results:
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
        plt.savefig(self.kernel_plot_path, dpi=150)
        plt.close()

    def fit_gaussian_process(self, x, y):
        # GaussianProcessRegressor ожидает матрицу признаков формы (n_samples, n_features),
        # поэтому одномерный вектор x разворачиваем в столбец.
        x_2d = x.reshape(-1, 1)
        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        kernel_candidates = [
            (
                "linear",
                [
                    # DotProduct задаёт линейные функции вида a + b * x.
                    # WhiteKernel добавляет независимый шум наблюдений.
                    # kDotProduct(x, x') = sigma_0 + x·x'
                    DotProduct(sigma_0=sigma_0) + WhiteKernel(noise_level=noise_level)
                    for sigma_0 in [1e-6, 0.5, 1.0]
                    for noise_level in [0.01, 0.04, 0.09]
                ],
            ),
            (
                "gaussian",
                [
                    # RBF задаёт очень гладкие функции:
                    # близкие x и x' дают близкие значения f(x) и f(x').
                    # Чем больше length_scale, тем плавнее итоговая кривая.
                    
                    # kRBF(x, x') = exp(-(x - x')**2 / (2 * length_scale**2))
                    RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
                    for length_scale in [0.2, 0.4, 0.8, 1.2]
                    for noise_level in [0.01, 0.04, 0.09]
                ],
            ),
            (
                "exponential",
                [
                    # Matern с nu=0.5 соответствует экспоненциальному ядру
                    # k(x, x') = exp(-|x - x'| / length_scale): функции менее гладкие,
                    # допускают более резкие локальные изменения, чем RBF.
                    Matern(length_scale=length_scale, nu=0.5)
                    + WhiteKernel(noise_level=noise_level)
                    for length_scale in [0.2, 0.4, 0.8, 1.2]
                    for noise_level in [0.01, 0.04, 0.09]
                ],
            ),
        ]

        best = None
        self.gp_results = []
        for kernel_name, kernels in kernel_candidates:
            for kernel in kernels:
                model = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=0.0, # шум к диагонали ковариацонной матрицы
                    normalize_y=True,
                    optimizer=None,
                    random_state=self.random_state,
                )
                cv_scores = cross_val_score(
                    model,
                    x_2d,
                    y,
                    scoring="neg_mean_squared_error",
                    cv=cv,
                    n_jobs=1,
                )
                cv_mse = float(-cv_scores.mean())
                model.fit(x_2d, y)
                train_pred, train_std = model.predict(x_2d, return_std=True)

                candidate = {
                    "kernel_name": kernel_name,
                    "kernel_repr": str(kernel),
                    "cv_mse": cv_mse,
                    "train_mse": float(np.mean((y - train_pred) ** 2)),
                    "train_std_mean": float(np.mean(train_std)),
                    "model": model,
                }
                self.gp_results.append(candidate)
                if best is None or candidate["cv_mse"] < best["cv_mse"]:
                    best = candidate

        self.gp_result = best
        self.gp_top_results = {}
        for kernel_name, _ in kernel_candidates:
            kernel_runs = [item for item in self.gp_results if item["kernel_name"] == kernel_name]
            kernel_runs.sort(key=lambda item: item["cv_mse"])
            self.gp_top_results[kernel_name] = kernel_runs[:3]

    def make_gp_plot(self, x, y):
        grid_x = np.linspace(self.domain[0], self.domain[1], 400).reshape(-1, 1)

        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, color="black", s=25, label="выборка")
        plt.plot(grid_x[:, 0], self.f(grid_x[:, 0]), linewidth=2, label="истинная f(x)")

        color_groups = {
            "linear": ["#2ca02c", "#66c266", "#a3d9a5"],
            "gaussian": ["#1f77b4", "#5aa0d6", "#9cc9eb"],
            "exponential": ["#9467bd", "#b494d3", "#d1c0e8"],
        }
        for kernel_name, top_candidates in self.gp_top_results.items():
            for idx, candidate in enumerate(top_candidates):
                mean_pred = candidate["model"].predict(grid_x)
                plt.plot(
                    grid_x[:, 0],
                    mean_pred,
                    linewidth=1.8,
                    alpha=0.95,
                    color=color_groups[kernel_name][idx],
                    label=(
                        f"{kernel_name} top-{idx + 1}, "
                        f"CV={self.format_number(candidate['cv_mse'])}"
                    ),
                )

        best_mean_pred, std_pred = self.gp_result["model"].predict(grid_x, return_std=True)
        plt.plot(
            grid_x[:, 0],
            best_mean_pred,
            linewidth=3,
            color="black",
            label="лучшая GP-модель",
        )
        plt.fill_between(
            grid_x[:, 0],
            best_mean_pred - 1.96 * std_pred,
            best_mean_pred + 1.96 * std_pred,
            color="skyblue",
            alpha=0.3,
            label="95% доверительный интервал",
        )
        plt.title("Топ-3 конфигурации гауссовского процесса для каждого ядра")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(self.gp_plot_path, dpi=150)
        plt.close()

    def resolve(self):
        x, y = self.generate_sample()

        self.fit_kernel_smoother(x, y)
        self.make_kernel_plot(x, y)

        self.fit_gaussian_process(x, y)
        self.make_gp_plot(x, y)

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
        for candidate in self.kernel_results:
            print(
                f"{candidate['kernel']}, {reg_type_label[candidate['reg_type']]}: "
                f"bandwidth = {self.format_number(candidate['bandwidth'])}, "
                f"LOOCV MSE = {self.format_number(candidate['cv_mse'])}, "
                f"train MSE = {self.format_number(candidate['train_mse'])}"
            )

        print("\nЛучшая конфигурация")
        print(f"Ядро: {self.kernel_result['kernel']}")
        print(
            "Локальная аппроксимация: "
            f"{reg_type_label[self.kernel_result['reg_type']]}"
        )
        print(f"Ширина окна: {self.format_number(self.kernel_result['bandwidth'])}")
        print(
            "Средняя квадратичная ошибка Leave-One-Out Cross-Validation: "
            f"{self.format_number(self.kernel_result['cv_mse'])}"
        )
        print(
            "Средняя квадратичная ошибка на обучении: "
            f"{self.format_number(self.kernel_result['train_mse'])}"
        )
        print(f"График сохранён в: {self.kernel_plot_path}")

        print("\n#########################\n")
        print("Задание 5. Пункт 2")
        print("Оценка функции с помощью регрессии гауссовским процессом")
        print("\nТоп-3 конфигурации для каждого ядра")
        for kernel_name, top_candidates in self.gp_top_results.items():
            print(f"\n{kernel_name}")
            for idx, candidate in enumerate(top_candidates, start=1):
                print(
                    f"top-{idx}: kernel = {candidate['kernel_repr']}, "
                    f"CV MSE = {self.format_number(candidate['cv_mse'])}, "
                    f"train MSE = {self.format_number(candidate['train_mse'])}, "
                    f"mean train std = {self.format_number(candidate['train_std_mean'])}"
                )

        print("\nЛучшая конфигурация")
        print(f"Ядро: {self.gp_result['kernel_name']}")
        print(f"Параметры: {self.gp_result['kernel_repr']}")
        print(f"Средняя квадратичная ошибка CV: {self.format_number(self.gp_result['cv_mse'])}")
        print(
            "Средняя квадратичная ошибка на обучении: "
            f"{self.format_number(self.gp_result['train_mse'])}"
        )
        print(
            "Среднее стандартное отклонение на обучении: "
            f"{self.format_number(self.gp_result['train_std_mean'])}"
        )
        print(f"График сохранён в: {self.gp_plot_path}")


if __name__ == "__main__":
    Task5Resolver().resolve()
