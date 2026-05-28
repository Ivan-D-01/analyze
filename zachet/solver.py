from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols


def print_header(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def save_boxplots(dataframe: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    plot_columns = ["F_1", "F_2", "F_3"]

    for axis, factor_name in zip(axes, plot_columns, strict=True):
        dataframe.boxplot(column="response_C", by=factor_name, ax=axis)
        axis.set_title(factor_name)
        axis.set_xlabel(factor_name)
        axis.set_ylabel("C")

    fig.suptitle("response_C boxplots by factor")
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore", module="statsmodels")

    data_path = Path(__file__).with_name("data.csv")
    df = pd.read_csv(data_path)
    work_df = df.rename(columns={"C": "response_C"}).copy()

    required_columns = {"F_1", "F_2", "F_3", "C"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing columns in {data_path}: {', '.join(sorted(missing_columns))}"
        )

    print(f"Data file: {data_path}")
    print(f"Shape: {df.shape}")

    print_header("1. One-way ANOVA: C ~ F_1")
    one_way_model = ols("response_C ~ C(F_1)", data=work_df).fit()
    one_way_anova = sm.stats.anova_lm(one_way_model, typ=2)
    print(one_way_anova)

    print_header("2. Two-way ANOVA: C ~ F_1 * F_2")
    two_way_model = ols("response_C ~ C(F_1) * C(F_2)", data=work_df).fit()
    two_way_anova = sm.stats.anova_lm(two_way_model, typ=2)
    print(two_way_anova)

    print_header("3. Three-way ANOVA: C ~ F_1 * F_2 * F_3")
    three_way_model = ols("response_C ~ C(F_1) * C(F_2) * C(F_3)", data=work_df).fit()
    three_way_anova = sm.stats.anova_lm(three_way_model, typ=2)
    print(three_way_anova)

    boxplot_path = Path(__file__).with_name("boxplots.pdf")
    save_boxplots(work_df, boxplot_path)
    print(f"Saved boxplots to {boxplot_path}")

if __name__ == "__main__":
    main()
