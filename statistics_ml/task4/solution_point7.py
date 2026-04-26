from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PolynomialFeatures,
    SplineTransformer,
    StandardScaler,
)


class HousesAdditiveModelResolver:
    def __init__(self, df, target_col, continuous_cols, categorical_cols):
        self.df = df
        self.target_col = target_col
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        self.base_dir = Path(__file__).resolve().parent
        self.plot_path = self.base_dir / "additive_model_plot.png"
        self.results = {}

    @staticmethod
    def format_number(value, digits=4):
        if isinstance(value, (int, np.integer)):
            return str(value)
        return f"{float(value):.{digits}f}"

    def prepare_data(self):
        data = self.df[self.continuous_cols + self.categorical_cols + [self.target_col]].copy()
        y = data[self.target_col].astype(float)
        X = data.drop(columns=[self.target_col])
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def build_spline_pipeline(self):
        continuous_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "basis",
                    SplineTransformer(
                        degree=3,
                        include_bias=False,
                        knots="quantile",
                    ),
                ),
                ("scale", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        preprocessor = ColumnTransformer(
            [
                ("continuous", continuous_pipeline, self.continuous_cols),
                ("categorical", categorical_pipeline, self.categorical_cols),
            ]
        )
        return Pipeline([("preprocessor", preprocessor), ("ridge", Ridge())])

    def build_polynomial_pipeline(self):
        continuous_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "basis",
                    PolynomialFeatures(include_bias=False),
                ),
                ("scale", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        preprocessor = ColumnTransformer(
            [
                ("continuous", continuous_pipeline, self.continuous_cols),
                ("categorical", categorical_pipeline, self.categorical_cols),
            ]
        )
        return Pipeline([("preprocessor", preprocessor), ("ridge", Ridge())])

    def fit_model(self, name, pipeline, param_grid, X_train, X_test, y_train, y_test):
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=5,
            n_jobs=1,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        train_pred = best_model.predict(X_train)
        test_pred = best_model.predict(X_test)

        result = {
            "model_name": name,
            "params": search.best_params_,
            "cv_mse": float(-search.best_score_),
            "train_mse": float(mean_squared_error(y_train, train_pred)),
            "test_mse": float(mean_squared_error(y_test, test_pred)),
            "train_r2": float(r2_score(y_train, train_pred)),
            "test_r2": float(r2_score(y_test, test_pred)),
            "test_pred": test_pred,
        }
        self.results[name] = result
        return result

    def fit_models(self):
        X_train, X_test, y_train, y_test = self.prepare_data()

        spline_result = self.fit_model(
            name="Сплайновая аддитивная модель",
            pipeline=self.build_spline_pipeline(),
            param_grid={
                "preprocessor__continuous__basis__n_knots": list(range(4, 12)),
                "ridge__alpha": np.logspace(-4, 3, 20),
            },
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
        polynomial_result = self.fit_model(
            name="Полиномиальная аддитивная модель",
            pipeline=self.build_polynomial_pipeline(),
            param_grid={
                "preprocessor__continuous__basis__degree": [1, 2, 3, 4],
                "ridge__alpha": np.logspace(-4, 3, 20),
            },
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        return X_test, y_test

    def make_plot(self, y_test):
        order = np.argsort(y_test.to_numpy())
        sorted_true = y_test.to_numpy()[order]
        sorted_spline_pred = self.results["Сплайновая аддитивная модель"]["test_pred"][order]
        sorted_polynomial_pred = self.results["Полиномиальная аддитивная модель"][
            "test_pred"
        ][order]

        plt.figure(figsize=(10, 6))
        plt.plot(sorted_true, linewidth=2, label="истинная цена")
        plt.plot(
            sorted_spline_pred,
            linewidth=2,
            color="purple",
            label="предсказание: сплайновая модель",
        )
        plt.plot(
            sorted_polynomial_pred,
            linewidth=2,
            color="green",
            label="предсказание: полиномиальная модель",
        )
        plt.title("Сравнение аддитивных моделей для SalePrice")
        plt.xlabel("объект тестовой выборки")
        plt.ylabel("SalePrice")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=150)
        plt.close()

    def print_result(self, result):
        print(f"\n{result['model_name']}")
        if "n_knots" in result["params"]:
            print(f"Число узлов сплайна: {result['params']['preprocessor__continuous__basis__n_knots']}")
        if "preprocessor__continuous__basis__degree" in result["params"]:
            print(
                "Степень полинома: "
                f"{result['params']['preprocessor__continuous__basis__degree']}"
            )
        print(f"Параметр alpha: {self.format_number(result['params']['ridge__alpha'])}")
        print(f"Средняя квадратичная ошибка CV: {self.format_number(result['cv_mse'])}")
        print(
            "Средняя квадратичная ошибка на обучении: "
            f"{self.format_number(result['train_mse'])}"
        )
        print(
            "Средняя квадратичная ошибка на тесте: "
            f"{self.format_number(result['test_mse'])}"
        )
        print(f"R^2 на обучении: {self.format_number(result['train_r2'])}")
        print(f"R^2 на тесте: {self.format_number(result['test_r2'])}")

    def resolve(self):
        X_test, y_test = self.fit_models()
        self.make_plot(y_test)

        print("\n#########################\n")
        print("Задание 4. Пункт 7")
        print("Аддитивные модели для прогноза SalePrice по house_prices.csv")
        print(f"Целевая переменная: {self.target_col}")
        print(f"Числовые признаки: {', '.join(self.continuous_cols)}")
        print(f"Категориальные признаки: {', '.join(self.categorical_cols)}")
        print(f"Размер тестовой выборки: {len(X_test)}")

        self.print_result(self.results["Сплайновая аддитивная модель"])
        self.print_result(self.results["Полиномиальная аддитивная модель"])

        best_result = min(self.results.values(), key=lambda item: item["cv_mse"])
        print("\nСравнение")
        print(
            "Лучшая модель по CV: "
            f"{best_result['model_name']} "
            f"(CV MSE = {self.format_number(best_result['cv_mse'])})"
        )
        print(f"График сохранён в: {self.plot_path}")


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent / "task1" / "data" / "house_prices.csv"
    df = pd.read_csv(data_path)

    HousesAdditiveModelResolver(
        df=df,
        target_col="SalePrice",
        continuous_cols=[
            "LotFrontage",
            "LotArea",
            "YearBuilt",
            "YearRemodAdd",
            "1stFlrSF",
            "2ndFlrSF",
            "GrLivArea",
            "TotalBsmtSF",
            "GarageArea",
        ],
        categorical_cols=[
            "MSZoning",
            "Street",
            "Alley",
            "LotShape",
            "LandContour",
            "Utilities",
            "LotConfig",
            "LandSlope",
            "Neighborhood",
        ],
    ).resolve()
