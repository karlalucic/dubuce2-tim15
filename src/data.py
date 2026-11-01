"""Loading the processed returns and building walk-forward train/test splits."""

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

PROCESSED_CSV = (
    Path(__file__).resolve().parents[1]
    / "data" / "processed" / "industry10_daily_2000_2024.csv"
)


def load_returns(path: Path = PROCESSED_CSV) -> pd.DataFrame:
    """Load daily decimal returns of the 10 industry portfolios.

    Returns a DataFrame indexed by date (DatetimeIndex) with one column per
    industry: NoDur, Durbl, Manuf, Enrgy, HiTec, Telcm, Shops, Hlth, Utils, Other.
    """
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df.sort_index()


def walk_forward_splits(
    df: pd.DataFrame,
    train_years: int = 5,
    first_test_year: int = 2005,
    last_test_year: int = 2024,
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Yield (test_year, train_df, test_df) for each test year.

    train_df holds the `train_years` calendar years before test_year and
    test_df holds test_year only, so train always ends before test starts.
    """
    years = df.index.year
    for test_year in range(first_test_year, last_test_year + 1):
        train_df = df[(years >= test_year - train_years) & (years < test_year)]
        test_df = df[years == test_year]
        assert train_df.index.max() < test_df.index.min(), "Train/test leakage"
        yield test_year, train_df, test_df
