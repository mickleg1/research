import pandas as pd
import pytest

from analyze import (
    assert_grouping_respects_volume_basis,
    assert_operator_normalized,
    assert_single_volume_basis,
)


def test_grouping_guard_rejects_mixed_basis_without_basis_group() -> None:
    df = pd.DataFrame(
        {
            "volume_basis": ["24h", "total"],
            "volume": [10.0, 20.0],
        }
    )
    with pytest.raises(AssertionError):
        assert_grouping_respects_volume_basis(df, ["metric_month"], "unit-test")


def test_single_basis_assertion_accepts_uniform_basis() -> None:
    df = pd.DataFrame({"volume_basis": ["monthly_handle", "monthly_handle"]})
    basis = assert_single_volume_basis(df, "unit-test")
    assert basis == "monthly_handle"


def test_operator_normalization_rejects_alias_like_values() -> None:
    with pytest.raises(AssertionError):
        assert_operator_normalized(pd.Series(["DraftKings Sportsbook", "fanduel"]))
