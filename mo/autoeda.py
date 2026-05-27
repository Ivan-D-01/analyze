from autoviz.AutoViz_Class import AutoViz_Class
import pandas as pd
import numpy as np

class AutoEDA:
    def __init__(self, df: pd.DataFrame, output_direstory: str):
        self.df = df
        self.output_directory = output_direstory
    
    def _preprocess_data_for_autoeda(self, df):
        df.fillna('MISSING', inplace=True)
        
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'year' in col.lower()]
        for col in date_cols:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
        
        cols_to_drop = [col for col in df.columns if df[col].nunique() == len(df)]
        df.drop(cols_to_drop, axis=1, inplace=True)
        
        return df

    def create_auto_eda(self):
        
        df = self._preprocess_data_for_autoeda(self.df)
        AV = AutoViz_Class()

        try:
            AV.AutoViz(
                filename='',  
                sep=",",
                depVar="SalePrice",  
                dfte=df, 
                header=0,
                verbose=2,  
                lowess=False,
                chart_format="html",  
                max_rows_analyzed=150000,
                max_cols_analyzed=30,
                save_plot_dir=self.output_directory
            )
        except Exception as e:
            print(f"Произошла ошибка: {str(e)}")
            print("Попробуем альтернативный метод с ограниченным числом признаков...")
            
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()[:10]  # берем только 10 категориальных
            reduced_df = df[num_cols + cat_cols + ['SalePrice']]
            
            AV.AutoViz(
                filename='',
                sep=",",
                depVar="SalePrice",
                dfte=reduced_df,
                verbose=2,
                chart_format="html",
                save_plot_dir=self.output_directory
            )

        print("\nАнализ завершен. Основные результаты:")
        print(f"- Проанализировано строк: {len(df)}")
        print(f"- Проанализировано столбцов: {len(df.columns)}")
        print(f"- Типы данных:\n{df.dtypes.value_counts()}")

if __name__ == "__main__":
    df = pd.read_csv('mo/data_processed/test.csv')
    AutoEDA(df, 'test').create_auto_eda()
    df = pd.read_csv('mo/data_processed/train.csv')
    AutoEDA(df, 'train').create_auto_eda()