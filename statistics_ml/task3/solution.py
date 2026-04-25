import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlxtend.feature_selection import SequentialFeatureSelector

def compare_simple_ridge_and_lasso_1(data, categorical_cols, continious_cols, target_col):
    X = data[continious_cols + categorical_cols]
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_imputer = SimpleImputer(strategy="median")
    X_train_num = num_imputer.fit_transform(X_train[continious_cols])
    X_test_num = num_imputer.transform(X_test[continious_cols])

    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_num)
    X_test_num = scaler.transform(X_test_num)

    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_train_cat = cat_imputer.fit_transform(X_train[categorical_cols])
    X_test_cat = cat_imputer.transform(X_test[categorical_cols])

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_cat = encoder.fit_transform(X_train_cat)
    X_test_cat = encoder.transform(X_test_cat)

    print("Тип и форма X_train_num:", type(X_train_num), X_train_num.shape)
    print("Тип и форма X_train_cat:", type(X_train_cat), X_train_cat.shape)

    X_train_processed = np.hstack([X_train_num, X_train_cat])
    X_test_processed = np.hstack([X_test_num, X_test_cat])

    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(max_iter=10000),
        "Lasso": Lasso(max_iter=10000)
    }

    params = {
        "Ridge": {"alpha": np.logspace(0.5, 1.5, 10), "solver": ['svd', 'cholesky', 'lsqr']},
        "Lasso": {"alpha": np.logspace(0.5, 1.5, 10)}
    }

    results = {}
    for name, model in models.items():
        if name in params:
            
            grid = GridSearchCV(model, param_grid=params[name], cv=5)
            grid.fit(X_train_processed, y_train)
            best_model = grid.best_estimator_
            best_alpha = grid.best_params_["alpha"]
            print(model)
            if name == "Ridge":
                best_solver = grid.best_params_["solver"]
                print(f"{name} best solber: {best_solver}")    
            print(f"{name} best alpha: {best_alpha}")
        else:
            best_model = model
            best_model.fit(X_train_processed, y_train)
        
        y_pred = best_model.predict(X_test_processed)
        
        results[name] = {
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MAE": mean_absolute_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred)
        }

    print("\nРезультаты:")
    for model, metrics in results.items():
        print(f"\n{model}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

def generate_data():
    N = 100  
    M = 10  
    true_coefs = np.zeros(M)
    true_coefs[0] = 1    
    true_coefs[1] = -2    

    np.random.seed(42)
    X = np.random.normal(0, 1, size=(N, M)) 
    epsilon = np.random.normal(0, 1, size=N)  
    y = X[:, 0] - 2*X[:, 1] + epsilon
    
    return X, y, N, M, true_coefs

def generate_and_compare_with_lasso_2(X, y, N, M, true_coefs):

    train_size = int(0.8 * N)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    models = {
        "LinearRegression": LinearRegression(),
        "Lasso": Lasso(max_iter=100000)
    }

    params = {
        "Lasso": {"alpha": np.logspace(-10, 10, 100)}
    }

    results = {}
    for name, model in models.items():
        if name in params:
            grid = GridSearchCV(model, params[name], cv=5, scoring='neg_mean_squared_error')
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            best_alpha = grid.best_params_['alpha']
            print(f"{name} best alpha: {best_alpha:.4f}")
        else:
            best_model = model
            best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        results[name] = {
            "MSE": mse,
            "coefs": best_model.coef_ if hasattr(best_model, 'coef_') else None,
            "intercept": best_model.intercept_ if hasattr(best_model, 'intercept_') else None
        }


    print("\nИстинные коэффициенты:")
    print(true_coefs)

    print("\nСравнение моделей:")
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"MSE: {res['MSE']:.4f}")
        print("Коэффициенты:")
        print(res['coefs'])
        
        if name == "Lasso":
            n_nonzero = np.sum(res['coefs'] != 0)
            print(f"Ненулевых коэффициентов: {n_nonzero}/{M}")

def stepwise_regression(X, y, N):
    lr_model = LinearRegression()
    train_size = int(0.8 * N)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    sfs_forward = SequentialFeatureSelector(lr_model,
                                        k_features=(1,10),
                                        forward=True,
                                        floating=False,
                                        scoring='neg_mean_squared_error',
                                        cv=5)

    sfs_forward.fit(X_train, y_train)
    selected_features = sfs_forward.k_feature_idx_
    
    X_selected = X_train[:, list(selected_features)]
    lr_model.fit(X_selected, y_train)
    
    print("\n\nSequentialFeatureSelector")
    print("Выбранные признаки:", selected_features) 
    print("Выбранные признаки (индексы):", sfs_forward.k_feature_idx_)
    print("Выбранные признаки (номера):", sfs_forward.k_feature_idx_)
    print("Коэффициенты модели:", lr_model.coef_)
    
    y_pred = lr_model.predict(X_test[:, list(selected_features)])
    mse = mean_squared_error(y_test, y_pred)
    print("MSE:", mse)


if __name__ == "__main__":

    data = pd.read_csv("analyze/statistics_ml/task2/data_houses/house_prices.csv")

    target_col = ["SalePrice"]
    continious_cols = [
        "LotFrontage", "LotArea", "YearBuilt", "YearRemodAdd", 
        "1stFlrSF", "2ndFlrSF", "GrLivArea", "TotalBsmtSF", "GarageArea",
        "LotFrontage", "MasVnrArea"
    ]
    categorical_cols = [
        "MSZoning", "Street", "Alley", "LotShape", "LandContour", 
        "OverallQual", "OverallCond", "Utilities", "LotConfig", 
        "LandSlope", "Neighborhood", "MSSubClass", "Utilities", 
        "Condition1", "Condition2", "BldgType", "HouseStyle",
        "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd",
        "MasVnrType", "ExterQual", "ExterCond", "Foundation"
    ]

    compare_simple_ridge_and_lasso_1(data, categorical_cols, continious_cols, target_col)
    
    X, y, N, M, true_coefs = generate_data()
    
    generate_and_compare_with_lasso_2(X, y, N, M, true_coefs)
    
    stepwise_regression(X, y, N)