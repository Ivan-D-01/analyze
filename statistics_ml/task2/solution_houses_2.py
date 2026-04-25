from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class HousesGLMResolver:
    def __init__(
        self,
        df,
        target_categorical_col,
        target_gamma_gaussian_continious_col,
        continious_cols,
        categorical_cols,
    ):
        self.df = df
        self.target_categorical_col = target_categorical_col
        self.target_gamma_gaussian_continious_col = target_gamma_gaussian_continious_col
        self.continious_cols = continious_cols
        self.categorical_cols = categorical_cols
        self.results_summary = []

    def prepare_data(self):
        data = self.df[
            self.continious_cols + self.categorical_cols + [self.target_categorical_col]
        ].copy()
        data[self.continious_cols] = SimpleImputer(strategy="median").fit_transform(
            data[self.continious_cols]
        )
        data[self.categorical_cols] = SimpleImputer(strategy="most_frequent").fit_transform(
            data[self.categorical_cols]
        )
        data = pd.get_dummies(data, columns=self.categorical_cols, drop_first=True)

        X = data.drop(columns=[self.target_categorical_col])
        X[self.continious_cols] = StandardScaler().fit_transform(X[self.continious_cols])
        X = sm.add_constant(X, has_constant="add").astype(float)
        y = data[self.target_categorical_col]
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def print_target_distribution(self):
        counts = self.df[self.target_categorical_col].value_counts(dropna=False)
        shares = self.df[self.target_categorical_col].value_counts(
            normalize=True, dropna=False
        )
        print(f"Target distribution: {self.target_categorical_col}")
        for level in counts.index:
            print(f"{level}: count={counts[level]}, share={shares[level]:.4f}")

    def print_split_distribution(self, y_train, y_test):
        full_counts = self.df[self.target_categorical_col].value_counts(dropna=False)
        train_counts = y_train.value_counts(dropna=False)
        test_counts = y_test.value_counts(dropna=False)

        print("\nTarget split distribution")
        for level in full_counts.index:
            full_count = int(full_counts.get(level, 0))
            train_count = int(train_counts.get(level, 0))
            test_count = int(test_counts.get(level, 0))
            full_share = full_count / len(self.df)
            train_share = train_count / len(y_train)
            test_share = test_count / len(y_test)
            print(
                f"{level}: "
                f"full={full_count} ({full_share:.4f}), "
                f"train={train_count} ({train_share:.4f}), "
                f"test={test_count} ({test_share:.4f})"
            )

    def fit_binary_models(self, name, link, model_name):
        X_train, X_test, y_train, y_test = self.prepare_data()
        levels = sorted(y_train.unique())

        print(f"\n{name}")
        self.print_split_distribution(y_train, y_test)
        for level in levels:
            y_train_bin = (y_train == level).astype(int)
            y_test_bin = (y_test == level).astype(int)

            model = sm.GLM(
                y_train_bin,
                X_train,
                family=sm.families.Binomial(link=link),
            )
            result = model.fit()

            y_prob = result.predict(X_test)
            y_prob = np.clip(y_prob, 1e-12, 1 - 1e-12)
            y_pred = (y_prob >= 0.5).astype(int)
            coefficients = result.params.sort_values(key=lambda s: s.abs(), ascending=False)
            top_coefficients = coefficients.head(3).round(4).to_dict()
            low_coefficients = coefficients.tail(3).sort_values(key=lambda s: s.abs()).round(4).to_dict()
            log_likelihood = np.sum(
                y_test_bin * np.log(y_prob) + (1 - y_test_bin) * np.log(1 - y_prob)
            )

            print(f"\nLevel {level}")
            print(f"Accuracy: {accuracy_score(y_test_bin, y_pred):.4f}")
            print(f"Precision: {precision_score(y_test_bin, y_pred, zero_division=0):.4f}")
            print(f"Recall: {recall_score(y_test_bin, y_pred, zero_division=0):.4f}")
            print(f"F1: {f1_score(y_test_bin, y_pred, zero_division=0):.4f}")
            print(f"Test log-likelihood: {log_likelihood:.4f}")
            print(f"Top 3 coefficients: {top_coefficients}")
            print(f"Bottom 3 coefficients: {low_coefficients}")

            self.results_summary.append(
                {
                    "model": model_name,
                    "level": level,
                    "log_likelihood": float(log_likelihood),
                }
            )

    def print_loglik_summary(self):
        if not self.results_summary:
            return

        summary_df = pd.DataFrame(self.results_summary)
        pivot = summary_df.pivot_table(
            index="level",
            columns="model",
            values="log_likelihood",
            aggfunc="first",
        )
        pivot["winner"] = np.where(
            pivot["logit"] > pivot["probit"], "logit", "probit"
        )

        print("\n#########################\n\n3. Log-likelihood summary")
        print(pivot.round(4).to_string())

    def fit_multinomial_logistic_regression(self):
        X_train, X_test, y_train, y_test = self.prepare_data()

        model = LogisticRegression(
            penalty=None,
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=2000,
            fit_intercept=False,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("\n#########################\n\n4. Multinomial logistic regression")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Macro F1: {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")

        coef_df = pd.DataFrame(model.coef_, index=model.classes_, columns=X_train.columns)
        for level in model.classes_:
            top_series = coef_df.loc[level].reindex(
                coef_df.loc[level].abs().sort_values(ascending=False).head(3).index
            )
            print(f"{level}: top 3 coefficients = {top_series.round(4).to_dict()}")

    def fit_gamma_vs_gaussian_regression(self):
        feature_categorical_cols = [
            col for col in self.categorical_cols if col != self.target_categorical_col
        ]
        feature_continious_cols = [
            col
            for col in self.continious_cols
            if col != self.target_gamma_gaussian_continious_col
        ]

        data = self.df[
            feature_continious_cols
            + feature_categorical_cols
            + [self.target_gamma_gaussian_continious_col]
        ].copy()
        data = data[data[self.target_gamma_gaussian_continious_col] > 0].copy()
        data[feature_continious_cols] = SimpleImputer(strategy="median").fit_transform(
            data[feature_continious_cols]
        )
        data[feature_categorical_cols] = SimpleImputer(strategy="most_frequent").fit_transform(
            data[feature_categorical_cols]
        )
        data = pd.get_dummies(data, columns=feature_categorical_cols, drop_first=True)

        X = data.drop(columns=[self.target_gamma_gaussian_continious_col])
        X[feature_continious_cols] = StandardScaler().fit_transform(X[feature_continious_cols])
        X = sm.add_constant(X, has_constant="add").astype(float)
        y = data[self.target_gamma_gaussian_continious_col].astype(float)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        gamma_result = sm.GLM(
            y_train,
            X_train,
            family=sm.families.Gamma(link=sm.families.links.Log()),
        ).fit()
        gaussian_result = sm.GLM(
            y_train,
            X_train,
            family=sm.families.Gaussian(),
        ).fit()

        gamma_pred = gamma_result.predict(X_test)
        gaussian_pred = gaussian_result.predict(X_test)
        gamma_loglik = np.sum(
            gamma_result.family.loglike_obs(y_test, gamma_pred, scale=gamma_result.scale)
        )
        gaussian_loglik = np.sum(
            gaussian_result.family.loglike_obs(
                y_test, gaussian_pred, scale=gaussian_result.scale
            )
        )

        print("\n#########################\n\n5. Gamma vs Gaussian regression")
        print(f"Gamma test log-likelihood: {gamma_loglik:.4f}")
        print(f"Gaussian test log-likelihood: {gaussian_loglik:.4f}")

        if gamma_loglik > gaussian_loglik:
            print("Gamma regression is better than Gaussian on the test set by log-likelihood.")
        else:
            print("Gaussian regression is better than Gamma on the test set by log-likelihood.")

    def resolve(self):
        self.print_target_distribution()
        self.fit_binary_models(
            "\n#########################\n\n1. Binary logistic regression", sm.families.links.Logit(), "logit"
        )
        self.fit_binary_models(
            "\n#########################\n\n2. Binary probit regression", sm.families.links.Probit(), "probit"
        )
        self.print_loglik_summary()
        self.fit_multinomial_logistic_regression()
        self.fit_gamma_vs_gaussian_regression()


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent / "data_houses" / "house_prices.csv"
    df = pd.read_csv(data_path)

    HousesGLMResolver(
        df=df,
        target_categorical_col="MSZoning",
        target_gamma_gaussian_continious_col="SalePrice",
        continious_cols=[
            "LotFrontage",
            "LotArea",
            "YearBuilt",
            "YearRemodAdd",
            "1stFlrSF",
            "2ndFlrSF",
            "GrLivArea",
            "TotalBsmtSF",
            "GarageArea",
            "SalePrice",
        ],
        categorical_cols=[
            #"MSZoning",
            "Street",
            "Alley",
            "LotShape",
            "LandContour",
            #"OverallQual",
            "Utilities",
            "LotConfig",
            "LandSlope",
            "Neighborhood"
        ],
    ).resolve()
