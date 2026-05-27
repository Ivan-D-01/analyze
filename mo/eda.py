import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages
from io import StringIO
import sys


class EDA:
    def __init__(self, filename):
        self.pdf = PdfPages(filename)
        self.buffer = StringIO()

    def add_text_page(self, title, text):
        fig = plt.figure(figsize=(11, 8))
        plt.text(0.1, 0.9, title, fontsize=16, fontweight='bold')
        plt.text(0.1, 0.8, text, fontsize=12,
                 verticalalignment='top', wrap=True)
        plt.axis('off')
        self.pdf.savefig(fig)
        plt.close()

    def add_plot(self, fig):
        self.pdf.savefig(fig)
        plt.close()

    def capture_print(self, func, *args, **kwargs):
        old_stdout = sys.stdout
        sys.stdout = self.buffer
        func(*args, **kwargs)
        sys.stdout = old_stdout
        text_output = self.buffer.getvalue()
        self.buffer.truncate(0)
        self.buffer.seek(0)
        return text_output

    def close(self):
        self.pdf.close()

    def print_corr_info(self, corr_with_target):
        print("=== КОРРЕЛЯЦИОННЫЙ АНАЛИЗ ===")
        print("\nТоп-10 признаков по корреляции с SalePrice:")
        print(corr_with_target.head(10).to_string())

    def print_missing_info(self, missing):
        print("=== АНАЛИЗ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ ===")
        print("\nКоличество пропусков по признакам:")
        print(missing.to_string())

    def create_eda_pdf_report(self, df, continuous_cols, target_col):
        # 1. Анализ целевой переменной
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        sns.histplot(df[target_col[0]], kde=True, ax=ax1)
        ax1.set_title('Распределение SalePrice')
        stats.probplot(df[target_col[0]], plot=ax2)
        ax2.set_title('QQ-plot для SalePrice')
        plt.suptitle('Анализ целевой переменной', y=1.02)
        plt.tight_layout()
        self.add_plot(fig)

        # 2. Анализ пропущенных значений
        missing = df.isnull().sum().sort_values(ascending=False)
        missing = missing[missing > 0]

        fig = plt.figure(figsize=(10, 6))
        missing.plot.bar()
        plt.title('Количество пропущенных значений')
        plt.ylabel('Количество пропусков')
        plt.xticks(rotation=45)
        plt.tight_layout()
        self.add_plot(fig)

        text_output = self.capture_print(self.print_missing_info, missing)
        self.add_text_page("Анализ пропущенных значений", text_output)

        # 3. Анализ непрерывных переменных
        for i, part in enumerate(np.array_split(continuous_cols, 2)):
            fig = plt.figure(figsize=(15, 10))
            for j, col in enumerate(part):
                plt.subplot(3, 2, j+1)
                sns.histplot(df[col], kde=True)
                plt.title(f'Распределение {col}')
            plt.suptitle(
                f'Анализ непрерывных переменных (часть {i+1})', y=1.02)
            plt.tight_layout()
            self.add_plot(fig)

        # 4. Анализ выбросов
        fig = plt.figure(figsize=(15, 8))
        sns.boxplot(data=df[continuous_cols])
        plt.title('Распределение непрерывных признаков (выбросы)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        self.add_plot(fig)

        # 5. Корреляционный анализ
        corr_matrix = df[continuous_cols + target_col].corr()
        corr_with_target = corr_matrix[target_col[0]].sort_values(
            ascending=False)

        fig = plt.figure(figsize=(15, 12))
        sns.heatmap(corr_matrix, annot=True,
                    cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Матрица корреляций между числовыми признаками')
        self.add_plot(fig)

        fig = plt.figure(figsize=(10, 8))
        sns.barplot(
            x=corr_with_target.values[1:11], y=corr_with_target.index[1:11])
        plt.title('Топ-10 признаков с наибольшей корреляцией к SalePrice')
        plt.xlabel('Коэффициент корреляции')
        plt.tight_layout()
        self.add_plot(fig)

        text_output = self.capture_print(self.print_corr_info, corr_with_target)
        self.add_text_page("Корреляционный анализ", text_output)


def main():
    path = 'mo/data_input/'
    train_name = 'train.csv'
    df = pd.read_csv(f'{path}{train_name}')

    target_col = ["SalePrice"]
    continuous_cols = [
        "LotFrontage", "LotArea", "YearBuilt", "YearRemodAdd",
        "1stFlrSF", "2ndFlrSF", "GrLivArea", "TotalBsmtSF", "GarageArea",
        "MasVnrArea"
    ]

    report = EDA(f'{path}house_prices_analysis.pdf')
    report.create_eda_pdf_report(df, continuous_cols, target_col)
    report.close()

    print(f"PDF-отчет успешно сгенерирован: {path}house_prices_analysis.pdf")


if __name__ == "__main__":
    main()