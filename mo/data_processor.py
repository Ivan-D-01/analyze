import pandas as pd
import numpy as np
import typing as t
from pathlib import Path
from sklearn.impute import SimpleImputer

class DataProcessor:
    @staticmethod
    def get_numeric_columns_for_iqr(
        df: pd.DataFrame,
        min_unique: int = 10,
        max_skewness: float = 5.0,
        exclude_columns: t.Optional[t.List[str]] = None
    ) -> t.List[str]:
        """
        Находит числовые столбцы, подходящие для обработки IQR.
        
        :param df: DataFrame для анализа
        :param min_unique: Минимальное количество уникальных значений
        :param max_skewness: Максимальное допустимое значение асимметрии
        :param exclude_columns: Столбцы для исключения
        :return: Список подходящих столбцов
        """
        if exclude_columns is None:
            exclude_columns = []
            
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in exclude_columns]
        
        suitable_cols = []
        for col in numeric_cols:
            unique_count = df[col].nunique()
            skewness = abs(df[col].skew())
            
            if (unique_count >= min_unique and 
                skewness <= max_skewness and 
                df[col].std() != 0):
                suitable_cols.append(col)
                
        return suitable_cols

    @staticmethod
    def replace_outliers_iqr(
        df: pd.DataFrame,
        columns_for_iqr: t.List[str],
        threshold: float = 1.5,
        iqr_bounds: t.Optional[dict] = None
    ) -> t.Union[pd.DataFrame, tuple[pd.DataFrame, dict]]:
        """
        Заменяет выбросы на границы IQR. Может работать в двух режимах:
        1) Если iqr_bounds=None - вычисляет границы по переданному df (для train)
        2) Если iqr_bounds задан - использует готовые границы (для test)
        
        :param df: Обрабатываемый DataFrame
        :param columns: Столбцы для обработки
        :param threshold: Множитель IQR
        :param iqr_bounds: Предвычисленные границы (для test данных)
        :return: Обработанный DataFrame + границы (если iqr_bounds=None)
        """
        clean_df = df.copy()
        computed_bounds = {}
        
        for col in columns_for_iqr:
            if iqr_bounds is None:
                Q1 = clean_df[col].quantile(0.25)
                Q3 = clean_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                computed_bounds[col] = {'lower': lower, 'upper': upper}
            else:
                lower = iqr_bounds[col]['lower']
                upper = iqr_bounds[col]['upper']
            
            clean_df[col] = clean_df[col].clip(lower=lower, upper=upper)
        
        return (clean_df, computed_bounds) if iqr_bounds is None else (clean_df, None)

    @staticmethod
    def save_to_csv(
            df: pd.DataFrame,
            file_path: str,
            index: bool = False,
            encoding: str = 'utf-8',
            sep: str = ',',
            decimal: str = '.',
            float_format: t.Optional[str] = None,
            date_format: str = '%Y-%m-%d',
            compression: t.Optional[str] = None) -> None:
        try:
            path = Path(file_path)
            
            path.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_csv(
                path,
                index=index,
                encoding=encoding,
                sep=sep,
                decimal=decimal,
                float_format=float_format,
                date_format=date_format,
                compression=compression
            )
            
            print(f"Данные успешно сохранены в {path}")
        
        except PermissionError:
            raise PermissionError(f"Нет прав на запись в {path}")
        except Exception as e:
            raise Exception(f"Ошибка при сохранении файла: {str(e)}")

    def smart_encode_dataframe_v2(df, ordinal_mappings, nominal_features, label_encoders=None, train_columns=None):
        """
        Применяет кодирование к DataFrame используя:
        - Порядковое кодирование (ordinal) для признаков с естественной градацией
        - One-hot кодирование (get_dummies) для номинальных признаков
        
        Параметры:
        - df: DataFrame для кодирования
        - train_columns: список колонок после get_dummies (для согласования колонок)
        
        Возвращает:
        - преобразованный DataFrame
        """
        df = df.copy()
        
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
        
        for col, mapping in ordinal_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0).astype(int) 
        
        nominal_features = [
            'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LandContour',
            'LotConfig', 'Neighborhood', 'Condition1', 'Condition2',
            'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
            'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir',
            'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition', 'Electrical'
        ]
        
        nominal_features = [col for col in nominal_features if col in df.columns]
        
        df = pd.get_dummies(df, columns=nominal_features)
        
        if train_columns is not None:
            for col in train_columns:
                if col not in df.columns:
                    df[col] = False
            df = df[train_columns]
        
        return df, None

    @staticmethod
    def data_process(
        df: pd.DataFrame,
        fill_na_rules: dict,
        output_path: str,
        ordinal_mappings: dict,
        nominal_features: list,
        label_encoders: t.Optional[dict] = None,
        train_columns: t.Optional[list] = None,
        iqr_bounds: t.Optional[dict] = None,
        columns_for_iqr: t.Optional[list] = None
    ) -> tuple:
        id_col = df['Id'] if 'Id' in df.columns else None
        sale_price_col = df['SalePrice'] if 'SalePrice' in df.columns else None
        
        is_train = train_columns is None
        
        if is_train:
            numeric_cols = DataProcessor.get_numeric_columns_for_iqr(df)
            columns_for_iqr = [col for col in numeric_cols if (col != 'Id' and col!= 'SalePrice')]
            
            df, iqr_bounds = DataProcessor.replace_outliers_iqr(
                df, columns_for_iqr=columns_for_iqr, iqr_bounds=None)
        else:
            columns_for_iqr = [col for col in columns_for_iqr if (col != 'Id' and col!= 'SalePrice')] if columns_for_iqr else None
            df, _ = DataProcessor.replace_outliers_iqr(
                df, columns_for_iqr=columns_for_iqr, iqr_bounds=iqr_bounds)

        
        cols_to_encode = [col for col in df.columns if (col != 'Id' and col!= 'SalePrice')]
        if is_train:
            df_encoded, label_encoders = DataProcessor.smart_encode_dataframe_v2(
                df[cols_to_encode], ordinal_mappings=ordinal_mappings, nominal_features=nominal_features, label_encoders=None, train_columns=None)
            train_columns = df_encoded.columns.tolist()
        else:
            df_encoded, _ = DataProcessor.smart_encode_dataframe_v2(
                df[cols_to_encode], ordinal_mappings=ordinal_mappings, nominal_features=nominal_features, label_encoders=label_encoders, train_columns=train_columns)
        
        if id_col is not None:
            df_encoded['Id'] = id_col
        if sale_price_col is not None:
            df_encoded['SalePrice'] = sale_price_col
            
        feature_cols = [col for col in df_encoded.columns if (col != 'Id' and col!= 'SalePrice')]
        df_features = df_encoded[feature_cols].copy()
        
        df_features['TotalSF'] = df_features['TotalBsmtSF'] + df_features['1stFlrSF'] + df_features['2ndFlrSF']
        df_features['TotalBath'] = df_features['FullBath'] + 0.5 * df_features['HalfBath']
        df_features['Age'] = df_features['YrSold'] - df_features['YearBuilt']
        df_features['RemodAge'] = df_features['YrSold'] - df_features['YearRemodAdd']
        
        for col in ['LotArea', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF']:
            if col in df_features.columns:
                df_features[col+'_log'] = np.log1p(df_features[col])
        
        if id_col is not None:
            df_features['Id'] = id_col
        if sale_price_col is not None:
            df_features['SalePrice'] = sale_price_col

        DataProcessor.save_to_csv(df_features, output_path)
        
        if is_train:
            return df_features, label_encoders, train_columns, iqr_bounds, columns_for_iqr
        else:
            return df_features, None, None, None, None

if __name__ == '__main__':
    df_train = pd.read_csv('mo/data_input/train.csv')
    df_test = pd.read_csv('mo/data_input/test.csv')
    # df_test_y = pd.read_csv('mo/data_input/sample_submission.csv')
    # df_test = pd.merge(df_test_x, df_test_y, on = 'Id')
    print(df_train.count())
    print(df_test.count())


    target_col = ["SalePrice"]
    continuous_cols = [
        "LotFrontage", "LotArea", "YearBuilt", "YearRemodAdd",
        "1stFlrSF", "2ndFlrSF", "GrLivArea", "TotalBsmtSF", "GarageArea",
        "MasVnrArea"
    ]
    categorical_cols = [
        "MSZoning", "Street", "Alley", "LotShape", "LandContour",
        "OverallQual", "OverallCond", "Utilities", "LotConfig",
        "LandSlope", "Neighborhood", "MSSubClass", "Utilities",
        "Condition1", "Condition2", "BldgType", "HouseStyle",
        "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd",
        "MasVnrType", "ExterQual", "ExterCond", "Foundation"
    ]

    fill_na_rule_train = {
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
    _, label_encoders, train_columns, iqr_bounds, columns_for_iqr =  DataProcessor.data_process(df_train,
                               fill_na_rules=fill_na_rule_train,
                               output_path='mo/data_processed/train.csv'
                            )
    DataProcessor.data_process(df_test,
                               fill_na_rules=fill_na_rules_test,
                               output_path='mo/data_processed/test.csv',
                               label_encoders=label_encoders,
                               train_columns=train_columns,
                               iqr_bounds=iqr_bounds,
                               columns_for_iqr=columns_for_iqr
                               )
    
    df1 = pd.read_csv('mo/data_processed/train.csv')
    df2 = pd.read_csv('mo/data_processed/test.csv')
    el_df1 = list(df1.columns)
    el_df2 = list(df2.columns)
    print(el_df1 ==  el_df2)
    for el in el_df1:
        if el not in el_df2:
            print(el)
    print('aaa')
    for el in el_df2:
        if el not in el_df1:
            print(el)

    print(df1.count())
    print(df2.count())