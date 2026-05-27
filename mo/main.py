from eda import EDA
from data_processor import DataProcessor
import pandas as pd
from autoeda import AutoEDA
from ml_learner import MLLearner
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer

NEED_TO_AUTOEDA = False


if __name__ == "__main__":
    df_train = pd.read_csv('mo/data_input/train.csv')
    df_test = pd.read_csv('mo/data_input/test.csv')
    
    random_state = 42
    
    target_col = ["SalePrice"]
    continuous_cols = [
        "LotFrontage", "LotArea", "YearBuilt", "YearRemodAdd",
        "1stFlrSF", "2ndFlrSF", "GrLivArea", "TotalBsmtSF", "GarageArea",
        "MasVnrArea"
    ]
    
    if NEED_TO_AUTOEDA:
        AutoEDA(df_train, 'mo/autoviz_plots').create_auto_eda()

    EDA('house_prices_analysis.pdf').create_eda_pdf_report(
        df_train, continuous_cols, target_col)

    fill_na_rules_train = {
        'PoolQC': df_train['PoolQC'].mode()[0], 
        'MiscFeature':  df_train['MiscFeature'].mode()[0],  
        'Alley':  df_train['Alley'].mode()[0],  
        'Fence':  df_train['Fence'].mode()[0],  
        'FireplaceQu': df_train['FireplaceQu'].mode()[0],  
        'LotFrontage': df_train['LotFrontage'].median(),
        'GarageType': df_train['GarageType'].mode()[0], 
        'GarageYrBlt': df_train['GarageYrBlt'].mode()[0], 
        'GarageFinish': df_train['GarageFinish'].mode()[0],
        'GarageQual': df_train['GarageQual'].mode()[0], 
        'GarageCond':  df_train['GarageCond'].mode()[0], 
        'BsmtExposure':  df_train['BsmtExposure'].mode()[0],
        'BsmtFinType2':  df_train['BsmtFinType2'].mode()[0],
        'BsmtFinType1':  df_train['BsmtFinType1'].mode()[0],
        'BsmtCond':  df_train['BsmtCond'].mode()[0],
        'BsmtQual':  df_train['BsmtQual'].mode()[0],
        'KitchenQual':  df_train['KitchenQual'].mode()[0],
        'MasVnrArea': df_train['MasVnrArea'].mode()[0], 
        'MasVnrType':  df_train['MasVnrType'].mode()[0],
        'Electrical': df_train['Electrical'].mode()[0], 
        'Functional': df_train['Functional'].mode()[0]
    }

    fill_na_rules_test = {
        'PoolQC': df_test['PoolQC'].mode()[0], 
        'MiscFeature':  df_test['MiscFeature'].mode()[0],  
        'Alley':  df_test['Alley'].mode()[0],  
        'Fence':  df_test['Fence'].mode()[0],  
        'FireplaceQu': df_test['FireplaceQu'].mode()[0],
        'LotFrontage': df_test['LotFrontage'].median(),
        'GarageType': df_test['GarageType'].mode()[0], 
        'GarageYrBlt': df_test['GarageYrBlt'].mode()[0],  
        'GarageFinish': df_test['GarageFinish'].mode()[0], 
        'GarageQual': df_test['GarageQual'].mode()[0],  
        'GarageCond':  df_test['GarageCond'].mode()[0],  
        'BsmtExposure':  df_test['BsmtExposure'].mode()[0],
        'BsmtFinType2':  df_test['BsmtFinType2'].mode()[0],
        'BsmtFinType1':  df_test['BsmtFinType1'].mode()[0],
        'BsmtCond':  df_test['BsmtCond'].mode()[0],  
        'BsmtQual':  df_test['BsmtQual'].mode()[0],  
        'KitchenQual':  df_test['KitchenQual'].mode()[0],
        'MasVnrArea': df_test['MasVnrArea'].mode()[0], 
        'MasVnrType':  df_test['MasVnrType'].mode()[0], 
        'Electrical': df_test['Electrical'].mode()[0],
        'Functional': df_test['Functional'].mode()[0]
    }

    ordinal_mappings = {
        'PoolQC': {'Ex': 4,'Gd': 3,'TA': 2,'Fa': 1,'NA': 0},
        'LotShape': {'Reg': 3, 'IR1': 2, 'IR2': 1, 'IR3': 0},
        'Utilities': {'AllPub': 3, 'NoSewr': 2, 'NoSeWa': 1, 'ELO': 0},
        'LandSlope': {'Gtl': 2, 'Mod': 1, 'Sev': 0},
        'ExterQual': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
        'ExterCond': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
        'BsmtQual': {'Ex': 6, 'Gd': 5, 'TA': 4, 'Fa': 3, 'Po': 2, 'No': 1, 'NA': 0},
        'BsmtCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
        'BsmtExposure': {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0},
        'BsmtFinType1': {'GLQ': 7, 'ALQ': 6, 'BLQ': 5, 'Rec': 4, 'LwQ': 3, 'Unf': 2, 'None': 1, 'NA': 0},
        'BsmtFinType2': {'GLQ': 7, 'ALQ': 6, 'BLQ': 5, 'Rec': 4, 'LwQ': 3, 'Unf': 2, 'None': 1, 'NA': 0},
        'Functional': {'Sal': 0, 'Sev': 1, 'Maj2': 2,'Maj1': 3, 'Mod': 4,  'Min2': 5,  'Min1': 6, 'Typ': 7},
        'GarageCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
        'Electrical': {'SBrkr': 4, 'FuseA': 3, 'FuseF': 2, 'FuseP': 1, 'Mix': 0},
        'HeatingQC': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
        'KitchenQual': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
        'FireplaceQu': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
        'GarageFinish': {'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0},
        'GarageQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
        'PavedDrive': {'Y': 2, 'P': 1, 'N': 0},
        'Fence': {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'NA': 0}
    }

    nominal_features = [
        'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LandContour',
        'LotConfig', 'Neighborhood', 'Condition1', 'Condition2',
        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir',
        'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition', 'Electrical'
    ]
    
    models_params = {
            'Linear Regression (Baseline)': {
                'model': LinearRegression(),
                'params': {
                    'fit_intercept': [True, False],
                #     'copy_X': [True, False],
                #     'positive': [True, False]
                }
            },
           'Ridge Regression': {
               'model': Ridge(),
               'params': {
                   'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                #    'fit_intercept': [True, False],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
               }
           },
           'Random Forest': {
               'model': RandomForestRegressor(random_state=random_state),
               'params': {
                   'n_estimators': [50, 100, 200],
                #    'max_depth': [None, 5, 10, 15],
                #    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                #    'max_features': ['sqrt', 'log2']
               }
           },
           'Gradient Boosting': {
                 'model': GradientBoostingRegressor(random_state=random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                #    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 4, 5, 6],
                #    'min_samples_split': [2, 5, 10],
                #    'min_samples_leaf': [1, 2, 4]
                }
            },
            'CatBoost': {
                'model': CatBoostRegressor(random_state=random_state, verbose=0),
                'params': {
                    'iterations': [500, 1000],
                #    'depth': [4, 6, 8],
                #    'learning_rate': [0.01, 0.05, 0.1],
                    'l2_leaf_reg': [1, 3, 5]
                }
            }
    }

    df_train_processed, label_encoders, train_columns, iqr_bounds, columns_for_iqr =  DataProcessor.data_process(
                            df=df_train,
                            fill_na_rules=fill_na_rules_train,
                            output_path='mo/data_processed/train.csv',
                            ordinal_mappings=ordinal_mappings,
                            nominal_features=nominal_features,
                            )

    df_test_processed, _, _, _, _ = DataProcessor.data_process(
                            df=df_test,
                            fill_na_rules=fill_na_rules_test,
                            output_path='mo/data_processed/test.csv',
                            ordinal_mappings=ordinal_mappings,
                            nominal_features=nominal_features,
                            label_encoders=label_encoders,
                            train_columns=train_columns,
                            iqr_bounds=iqr_bounds,
                            columns_for_iqr=columns_for_iqr
                            )

    imputer = SimpleImputer(strategy='most_frequent')
    train_sale_price = df_train_processed['SalePrice']
    df_train_processed = pd.DataFrame(imputer.fit_transform(df_train_processed.drop('SalePrice',axis=1)), columns=df_train_processed.drop('SalePrice',axis=1).columns)
    df_train_processed['SalePrice'] = train_sale_price

    df_test_processed = pd.DataFrame(imputer.transform(df_test_processed), columns=df_test_processed.columns)
    
    ml_learner = MLLearner()
    _, best_model = ml_learner.train_and_evaluate_models(df_train_processed.drop('Id',axis=1), models_params=models_params)
    
    ml_learner.get_results_on_test(df_test_processed.drop('Id',axis=1),
                                   best_model,
                                   df_test_processed['Id'])    
    
