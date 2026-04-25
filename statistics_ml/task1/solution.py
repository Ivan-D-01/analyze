import pandas as pd
import numpy as np
import typing as t
from pathlib import Path


class HousePricesResolver:
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: t.List[str],
        continious_cols: t.List[str],
        categorical_cols: t.List[str],
    ):
        self.df = df
        self.target_col = target_col
        self.continious_cols = continious_cols
        self.categorical_cols = categorical_cols

    def one_hot_encode(self, df, column_name):
        """
        Функция для one-hot encoding категориальной переменной
        :param df: исходный DataFrame
        :param column_name: имя категориальной колонки
        :return: DataFrame с one-hot encoded колонками и исходный DataFrame без исходной колонки
        """
        dummies = pd.get_dummies(
            df[column_name], prefix=column_name, dtype="int8", drop_first=True
        )
        df = pd.concat([df.drop(column_name, axis=1), dummies], axis=1)
        return df

    def prepare_data(self, df):
        """
        Подготовка данных: обработка пропусков, one-hot encoding категориальных переменных,
        выбор непрерывных переменных для регрессии
        :param df: исходный DataFrame
        :return: подготовленный DataFrame и целевая переменная SalePrice
        """
        # print("prepare START")
        df = df.drop("Id", axis=1)

        df = df[self.categorical_cols + self.continious_cols + self.target_col]
        # print(df.dtypes)
        for col in self.continious_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        for col in self.categorical_cols:
            if col in df.columns:
                df = self.one_hot_encode(df, col)
        # print(df.dtypes)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if "SalePrice" not in numeric_cols:
            raise ValueError("Target variable 'SalePrice' not found in data")

        df["intercept"] = 1

        feature_cols = [col for col in numeric_cols if col != "SalePrice"]
        feature_cols.append("intercept")
        # print(df[feature_cols].dtypes)
        # print("prepare END")
        return df[feature_cols], df["SalePrice"]

    def linear_regression_analytical(self, X, y):
        """
        Аналитическое решение линейной регрессии (метод наименьших квадратов)
        :param X: матрица признаков (n_samples, n_features)
        :param y: вектор целевой переменной (n_samples,)
        :return: вектор коэффициентов (n_features,)
        """
        X = np.array(X)

        y = np.array(y)

        XTX = np.dot(X.T, X)
        rank = np.linalg.matrix_rank(XTX)

        if rank < XTX.shape[0]:
            print(f"Матрица вырождена: ранг {rank} < {XTX.shape[0]}")
        else:
            print("Матрица невырождена")

        XTX_inv = np.linalg.pinv(XTX)
        XTy = np.dot(X.T, y)
        coefficients = np.dot(XTX_inv, XTy)

        return coefficients

    def predict(self, X, coefficients):
        """
        Предсказание значений по модели
        :param X: матрица признаков
        :param coefficients: вектор коэффициентов
        :return: вектор предсказаний
        """
        return np.dot(X, coefficients)

    def r2_score(self, y_true, y_pred):
        """
        Расчет R^2 (коэффициента детерминации)
        :param y_true: истинные значения
        :param y_pred: предсказанные значения
        :return: R^2 score
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)

        return 1 - (ss_residual / ss_total)

    def add_interactions(self, df):
        """
        Добавление взаимодействий между OverallQual и one-hot колонками Neighborhood
        :param df: DataFrame с признаками
        :return: DataFrame с добавленными взаимодействиями
        """
        df_with_interactions = df.copy()

        interaction_cols = [
            col
            for col in df.columns
            if col.startswith("Neighborhood_")
        ]
        print(f"interaction columns: {len(interaction_cols)}")

        base_feature = "OverallQual"
        if base_feature not in df.columns:
            return df_with_interactions

        for interaction_col in interaction_cols:
            new_col_name = f"{base_feature}_x_{interaction_col}"
            df_with_interactions[new_col_name] = (
                df_with_interactions[base_feature] * df_with_interactions[interaction_col]
            )

        return df_with_interactions

    def train_test_split(self, X, y, test_size=0.2, random_state=None):
        """
        Разделение данных на обучающую и тестовую выборки
        :param X: признаки
        :param y: целевая переменная
        :param test_size: размер тестовой выборки (0-1)
        :param random_state: seed для воспроизводимости
        :return: X_train, X_test, y_train, y_test
        """
        if random_state is not None:
            np.random.seed(random_state)

        n_samples = X.shape[0]
        n_test = int(n_samples * test_size)

        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]

        return X_train, X_test, y_train, y_test

    def resolve(self):

        X, y = self.prepare_data(self.df)
        # print(X.dtypes)
        X_train, X_test, y_train, y_test = self.train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        coefficients = self.linear_regression_analytical(X_train, y_train)

        y_train_pred = self.predict(X_train, coefficients)

        train_r2 = self.r2_score(y_train, y_train_pred)
        print(f"R2 на обучающей выборке: {train_r2}")

        y_test_pred = self.predict(X_test, coefficients)
        test_r2 = self.r2_score(y_test, y_test_pred)
        print(f"R2 на тестовой выборке: {test_r2:.4f}")

        print("\nИнтерпретация коэффициентов (первые 10):")
        for feature, coef in zip(X.columns[:10], coefficients[:10]):
            print(f"{feature}: {coef:.2f}")

        X_with_interactions = self.add_interactions(X)

        X_train_int, X_test_int, y_train_int, y_test_int = self.train_test_split(
            X_with_interactions, y, test_size=0.2, random_state=42)

        coefficients_int = self.linear_regression_analytical(X_train_int, y_train_int)

        y_train_pred_int = self.predict(X_train_int, coefficients_int)
        train_r2_int = self.r2_score(y_train_int, y_train_pred_int)
        print(f"\nR2 с взаимодействиями на обучающей выборке: {train_r2_int:.4f}")

        y_test_pred_int = self.predict(X_test_int, coefficients_int)
        test_r2_int = self.r2_score(y_test_int, y_test_pred_int)
        print(f"R2 с взаимодействиями на тестовой выборке: {test_r2_int:.4f}")


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent / "data" / "house_prices.csv"
    df = pd.read_csv(data_path)

    house_prices_resolver = HousePricesResolver(
        df=df,
        target_col=["SalePrice"],
        continious_cols=[
            "LotFrontage",
            "LotArea",
            "OverallQual",
            "OverallCond",
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
            "LotShape",
            "LandContour",
            "Utilities",
            "LotConfig",
            "LandSlope",
            "Neighborhood",
        ],
    )
    house_prices_resolver.resolve()
