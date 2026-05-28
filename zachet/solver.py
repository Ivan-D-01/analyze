from pathlib import Path
import warnings

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA


def print_header(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def main() -> None:
    warnings.filterwarnings("ignore", module="statsmodels")

    data_path = Path(__file__).with_name("data.csv")
    df = pd.read_csv(data_path)
    work_df = df.rename(columns={"C": "response_C"}).copy()

    required_columns = {"F_1", "F_2", "C"}
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

if __name__ == "__main__":
    main()
