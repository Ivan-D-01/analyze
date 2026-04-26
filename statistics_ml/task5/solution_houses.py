from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, Matern, RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.kernel_regression import KernelReg


class HousesTask5Resolver:
    def __init__(self, df, target_col, feature_cols):
        self.df = df
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.base_dir = Path(__file__).resolve().parent
        self.plot_path = self.base_dir / "house_prices_methods_plot.png"
        self.random_state = 42
        self.kernel_result = None
        self.kernel_results = []
        self.gp_result = None
        self.gp_results = []
        self.validation_size = 0.2
        self.kernel_bandwidth_grid = [0.3, 0.6, 0.9, 1.2, 1.8]
        self.gp_tuning_sample_size = 250

    @staticmethod
    def format_number(value, digits=4):
        if isinstance(value, (int, np.integer)):
            return str(value)
        return f"{float(value):.{digits}f}"

    @staticmethod
    def format_sequence(values, digits=4):
        return "[" + ", ".join(f"{float(value):.{digits}f}" for value in values) + "]"

    def prepare_data(self):
        data = self.df[self.feature_cols + [self.target_col]].copy()
        X = data[self.feature_cols]
        y = data[self.target_col].astype(float)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_state)

    def prepare_validation_split(self, X_train, y_train):
        return train_test_split(
            X_train,
            y_train,
            test_size=self.validation_size,
            random_state=self.random_state,
        )

    def fit_kernel_smoother(self, X_train, X_test, y_train, y_test):
        X_subtrain, X_val, y_subtrain, y_val = self.prepare_validation_split(X_train, y_train)
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_subtrain_arr = scaler.fit_transform(imputer.fit_transform(X_subtrain))
        X_val_arr = scaler.transform(imputer.transform(X_val))
        X_train_arr = scaler.transform(imputer.transform(X_train))
        X_test_arr = scaler.transform(imputer.transform(X_test))
        y_subtrain_arr = y_subtrain.to_numpy()
        y_val_arr = y_val.to_numpy()
        y_train_arr = y_train.to_numpy()
        y_test_arr = y_test.to_numpy()

        kernels = ["gaussian", "tricube"]
        reg_types = ["lc", "ll"]
        var_type = "c" * X_train_arr.shape[1]

        best = None
        self.kernel_results = []
        for kernel_name in kernels:
            for reg_type in reg_types:
                for bandwidth in self.kernel_bandwidth_grid:
                    bandwidth_vector = np.full(X_train_arr.shape[1], bandwidth, dtype=float)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        model = KernelReg(
                            endog=y_subtrain_arr,
                            exog=X_subtrain_arr,
                            var_type=var_type,
                            reg_type=reg_type,
                            bw=bandwidth_vector,
                            ckertype=kernel_name,
                        )
                        val_pred, _ = model.fit(X_val_arr)
                        train_pred, _ = model.fit(X_train_arr)
                        test_pred, _ = model.fit(X_test_arr)

                    candidate = {
                        "kernel": kernel_name,
                        "reg_type": reg_type,
                        "bandwidth": bandwidth_vector.copy(),
                        "cv_mse": float(mean_squared_error(y_val_arr, val_pred)),
                        "train_mse": float(mean_squared_error(y_train_arr, train_pred)),
                        "test_mse": float(mean_squared_error(y_test_arr, test_pred)),
                        "train_r2": float(r2_score(y_train_arr, train_pred)),
                        "test_r2": float(r2_score(y_test_arr, test_pred)),
                        "test_pred": test_pred,
                    }
                    self.kernel_results.append(candidate)
                    if best is None or candidate["cv_mse"] < best["cv_mse"]:
                        best = candidate

        self.kernel_result = best

    def fit_gaussian_process(self, X_train, X_test, y_train, y_test):
        X_subtrain, X_val, y_subtrain, y_val = self.prepare_validation_split(X_train, y_train)
        if len(X_subtrain) > self.gp_tuning_sample_size:
            tune_indices = X_subtrain.sample(
                n=self.gp_tuning_sample_size,
                random_state=self.random_state,
            ).index
            X_subtrain_tune = X_subtrain.loc[tune_indices]
            y_subtrain_tune = y_subtrain.loc[tune_indices]
        else:
            X_subtrain_tune = X_subtrain
            y_subtrain_tune = y_subtrain

        kernel_candidates = [
            (
                "linear",
                [
                    DotProduct(sigma_0=sigma_0) + WhiteKernel(noise_level=noise_level)
                    for sigma_0 in [0.5, 1.0]
                    for noise_level in [0.01, 0.05]
                ],
            ),
            (
                "gaussian",
                [
                    RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
                    for length_scale in [0.4, 0.8, 1.2]
                    for noise_level in [0.01, 0.05]
                ],
            ),
            (
                "exponential",
                [
                    Matern(length_scale=length_scale, nu=0.5)
                    + WhiteKernel(noise_level=noise_level)
                    for length_scale in [0.4, 0.8, 1.2]
                    for noise_level in [0.01, 0.05]
                ],
            ),
        ]

        best = None
        self.gp_results = []
        for kernel_name, kernels in kernel_candidates:
            for kernel in kernels:
                pipeline = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                        (
                            "gp",
                            GaussianProcessRegressor(
                                kernel=kernel,
                                alpha=0.0,
                                normalize_y=True,
                                optimizer=None,
                                random_state=self.random_state,
                            ),
                        ),
                    ]
                )
                pipeline.fit(X_subtrain_tune, y_subtrain_tune)
                val_pred = pipeline.predict(X_val)

                final_pipeline = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                        (
                            "gp",
                            GaussianProcessRegressor(
                                kernel=kernel,
                                alpha=0.0,
                                normalize_y=True,
                                optimizer=None,
                                random_state=self.random_state,
                            ),
                        ),
                    ]
                )
                final_pipeline.fit(X_train, y_train)
                train_pred, train_std = final_pipeline.predict(X_train, return_std=True)
                test_pred, test_std = final_pipeline.predict(X_test, return_std=True)

                candidate = {
                    "kernel_name": kernel_name,
                    "kernel_repr": str(kernel),
                    "cv_mse": float(mean_squared_error(y_val, val_pred)),
                    "train_mse": float(mean_squared_error(y_train, train_pred)),
                    "test_mse": float(mean_squared_error(y_test, test_pred)),
                    "train_r2": float(r2_score(y_train, train_pred)),
                    "test_r2": float(r2_score(y_test, test_pred)),
                    "train_std_mean": float(np.mean(train_std)),
                    "test_std_mean": float(np.mean(test_std)),
                    "test_pred": test_pred,
                }
                self.gp_results.append(candidate)
                if best is None or candidate["cv_mse"] < best["cv_mse"]:
                    best = candidate

        self.gp_result = best

    def make_plot(self, y_test):
        order = np.argsort(y_test.to_numpy())
        sorted_true = y_test.to_numpy()[order]
        sorted_kernel_pred = self.kernel_result["test_pred"][order]
        sorted_gp_pred = self.gp_result["test_pred"][order]

        plt.figure(figsize=(10, 6))
        plt.plot(sorted_true, linewidth=2, color="black", label="истинная цена")
        plt.plot(
            sorted_kernel_pred,
            linewidth=2,
            color="darkorange",
            label="лучший ядерный сглаживатель",
        )
        plt.plot(
            sorted_gp_pred,
            linewidth=2,
            color="navy",
            label="лучший гауссовский процесс",
        )
        plt.title("Сравнение лучших моделей для house_prices.csv")
        plt.xlabel("объект тестовой выборки")
        plt.ylabel("SalePrice")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=150)
        plt.close()

    def resolve(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.fit_kernel_smoother(X_train, X_test, y_train, y_test)
        self.fit_gaussian_process(X_train, X_test, y_train, y_test)
        self.make_plot(y_test)

        reg_type_label = {"lc": "local_constant", "ll": "local_linear"}

        print("\n#########################\n")
        print("Задание 5. Данные house_prices.csv")
        print("Прогноз непрерывной переменной SalePrice")
        print(f"Целевая переменная: {self.target_col}")
        print(f"Используемые признаки: {', '.join(self.feature_cols)}")
        print("Примечание: для устойчивого непараметрического сглаживания на реальных данных используется сокращённый набор непрерывных признаков.")
        print(f"Размер обучающей выборки: {len(X_train)}")
        print(f"Размер тестовой выборки: {len(X_test)}")

        print("\nЯдерное сглаживание: все конфигурации")
        for candidate in self.kernel_results:
            print(
                f"{candidate['kernel']}, {reg_type_label[candidate['reg_type']]}: "
                f"bandwidth = {self.format_sequence(candidate['bandwidth'])}, "
                f"CV MSE = {self.format_number(candidate['cv_mse'])}, "
                f"test MSE = {self.format_number(candidate['test_mse'])}, "
                f"test R^2 = {self.format_number(candidate['test_r2'])}"
            )

        print("\nЛучшая модель ядерного сглаживания")
        print(f"Ядро: {self.kernel_result['kernel']}")
        print(
            "Локальная аппроксимация: "
            f"{reg_type_label[self.kernel_result['reg_type']]}"
        )
        print(f"Ширины окна: {self.format_sequence(self.kernel_result['bandwidth'])}")
        print(f"CV MSE: {self.format_number(self.kernel_result['cv_mse'])}")
        print(f"Train MSE: {self.format_number(self.kernel_result['train_mse'])}")
        print(f"Test MSE: {self.format_number(self.kernel_result['test_mse'])}")
        print(f"Train R^2: {self.format_number(self.kernel_result['train_r2'])}")
        print(f"Test R^2: {self.format_number(self.kernel_result['test_r2'])}")

        print("\nГауссовский процесс: все конфигурации")
        for candidate in self.gp_results:
            print(
                f"{candidate['kernel_name']}, kernel = {candidate['kernel_repr']}: "
                f"CV MSE = {self.format_number(candidate['cv_mse'])}, "
                f"test MSE = {self.format_number(candidate['test_mse'])}, "
                f"test R^2 = {self.format_number(candidate['test_r2'])}"
            )

        print("\nЛучшая модель гауссовского процесса")
        print(f"Ядро: {self.gp_result['kernel_name']}")
        print(f"Параметры: {self.gp_result['kernel_repr']}")
        print(f"CV MSE: {self.format_number(self.gp_result['cv_mse'])}")
        print(f"Train MSE: {self.format_number(self.gp_result['train_mse'])}")
        print(f"Test MSE: {self.format_number(self.gp_result['test_mse'])}")
        print(f"Train R^2: {self.format_number(self.gp_result['train_r2'])}")
        print(f"Test R^2: {self.format_number(self.gp_result['test_r2'])}")
        print(
            "Среднее стандартное отклонение на тесте: "
            f"{self.format_number(self.gp_result['test_std_mean'])}"
        )

        print("\nСравнение лучших моделей")
        best_method = min(
            [
                ("kernel_smoothing", self.kernel_result["cv_mse"]),
                ("gaussian_process", self.gp_result["cv_mse"]),
            ],
            key=lambda item: item[1],
        )
        print(
            "Лучшая модель по CV: "
            f"{best_method[0]} (CV MSE = {self.format_number(best_method[1])})"
        )
        print(f"График сохранён в: {self.plot_path}")


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent / "task1" / "data" / "house_prices.csv"
    df = pd.read_csv(data_path)

    HousesTask5Resolver(
        df=df,
        target_col="SalePrice",
        feature_cols=[
            "GrLivArea",
            "GarageArea",
        ],
    ).resolve()
