import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings('ignore')  

class MLLearner:
    def __init__(self):
        print('Initializing MLLearner')
        
    def get_results_on_test(self, test_df_processed, best_model, ids):
        y_pred = best_model.predict(test_df_processed)
        results_df = pd.DataFrame({
            'Id': pd.to_numeric(ids).astype(int),
            'SalePrice': y_pred,
        })
        results_df.to_csv('mo/my_submission.csv', index=False)
        print(f"Результаты предсказаний сохранены в файл: my_submission.csv")



    def train_and_evaluate_models(self, df, models_params: dict, target_col='SalePrice', test_size=0.2, random_state=42,):
        
        results = {
            'models': {},
            'metrics': pd.DataFrame(columns=['Model', 'RMSE', 'R2', 'CV R2', 'Best Params']),
            'feature_importance': {}
        }
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        for name, mp in models_params.items():
            try:
                print(f"\nОбучение модели с поиском по сетке: {name}")
                
                grid_search = GridSearchCV(
                    mp['model'],
                    mp['params'],
                    cv=3,
                    scoring='r2',
                    n_jobs=-1,
                    verbose=1
                )
                

                grid_search.fit(X_train, y_train)
                
                best_model = grid_search.best_estimator_
                

                y_pred = best_model.predict(X_test)
                
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                cv_score = cross_val_score(best_model, X_train, y_train, 
                                        cv=5, scoring='r2').mean()
                
                results['models'][name] = best_model
                results['metrics'].loc[len(results['metrics'])] = {
                    'Model': name,
                    'RMSE': rmse,
                    'R2': r2,
                    'CV R2': cv_score,
                    'Best Params': str(grid_search.best_params_)
                }
                
                # Важность признаков
                if hasattr(best_model, 'feature_importances_'):
                    results['feature_importance'][name] = pd.Series(
                        best_model.feature_importances_,
                        index=X_train.columns  
                    ).sort_values(ascending=False)
                elif hasattr(best_model, 'coef_'):
                    results['feature_importance'][name] = pd.Series(
                        best_model.coef_
                    ).sort_values(key=abs, ascending=False)
                    
                print(f"{name} обучена. Лучшие параметры: {grid_search.best_params_}")
                print(f"R2: {r2:.4f}, RMSE: {rmse:.4f}, CV R2: {cv_score:.4f}")
                
            except Exception as e:
                print(f"Ошибка при обучении модели {name}: {str(e)}")
                continue
        
        best_model_name = results['metrics'].loc[results['metrics']['R2'].idxmax(), 'Model']
        best_model = results['models'][best_model_name]
        
        print(f"\nЛучшая модель: {best_model_name}")
        print(f"Параметры лучшей модели: {results['metrics'].loc[results['metrics']['Model'] == best_model_name, 'Best Params'].values[0]}")
        
        results['metrics'] = results['metrics'].sort_values('R2', ascending=False)
        print("\nСравнение моделей:")
        print(results['metrics'].to_markdown(index=False))
        
        if best_model_name in results['feature_importance']:
            print("\nТоп-10 важных признаков для лучшей модели:")
            print(results['feature_importance'][best_model_name].head(10))
        
        return results, best_model


if __name__ == "__main__":
    df = pd.read_csv("mo/data_processed/train.csv")
    test_df = pd.read_csv('mo/data_processed/test.csv')

    imputer = SimpleImputer(strategy='most_frequent')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    test_df = pd.DataFrame(imputer.transform(test_df), columns=test_df.columns)

    ml_learner = MLLearner()
    _, best_model = ml_learner.train_and_evaluate_models(df.drop('Id',axis=1))
    
    ml_learner.get_results_on_test(test_df.drop('Id',axis=1),
                                   best_model,
                                   test_df['Id'])
    