import pandas as pd
import pytest

from src.data import load_returns, walk_forward_splits

EXPECTED_COLUMNS = ["NoDur", "Durbl", "Manuf", "Enrgy", "HiTec",
                    "Telcm", "Shops", "Hlth", "Utils", "Other"]


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return load_returns()


def test_returns_load_with_expected_shape(returns):
    assert isinstance(returns.index, pd.DatetimeIndex)
    assert list(returns.columns) == EXPECTED_COLUMNS
    assert 6000 <= len(returns) <= 6500


def test_returns_have_no_missing_values(returns):
    assert not returns.isna().any().any()


def test_returns_are_decimal_not_percent(returns):
    # Daily returns of industry portfolios stay well below 100% in absolute value.
    assert returns.abs().max().max() < 0.5


def test_date_range(returns):
    assert returns.index.min() >= pd.Timestamp("2000-01-01")
    assert returns.index.max() <= pd.Timestamp("2024-12-31")
    assert returns.index.min().year == 2000
    assert returns.index.max().year == 2024
    assert returns.index.is_monotonic_increasing
    assert returns.index.is_unique


def test_walk_forward_covers_all_test_years(returns):
    years = [year for year, _, _ in walk_forward_splits(returns)]
    assert years == list(range(2005, 2025))


def test_walk_forward_train_and_test_do_not_overlap(returns):
    for test_year, train, test in walk_forward_splits(returns):
        assert train.index.intersection(test.index).empty
        assert train.index.max() < test.index.min()
        assert set(test.index.year) == {test_year}
        assert set(train.index.year) == set(range(test_year - 5, test_year))
