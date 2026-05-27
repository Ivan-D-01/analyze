import seaborn as sns
import pandas as pd

df = pd.read_csv('mo/data_processed/test.csv')

columns_with_nan = df.columns[df.isna().any()].tolist()
print("Колонки с NaN (вариант 1):", columns_with_nan)

# Вариант 2: С подсчетом количества NaN в каждой колонке
nan_counts = df.isna().sum()
print("\nКоличество NaN по колонкам (вариант 2):")
print(nan_counts[nan_counts > 0])

# Вариант 3: Полная информация о NaN (процентное соотношение)
nan_info = df.isna().mean().mul(100).round(2)
print("\nПроцент NaN значений по колонкам (вариант 3):")
print(nan_info[nan_info > 0])