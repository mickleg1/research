import pandas as pd

from collect_prediction_markets import (
    CORE_COLUMNS,
    EXPECTED_COLUMNS,
    assert_valid_prediction_schema,
    build_prediction_dataframe,
)


def test_expected_columns_present() -> None:
    for col in ["source_url", "collected_at_utc", "collector_version"]:
        assert col in EXPECTED_COLUMNS


def test_core_columns_lead_schema() -> None:
    assert EXPECTED_COLUMNS[: len(CORE_COLUMNS)] == CORE_COLUMNS


def test_schema_validation_rejects_missing_provenance() -> None:
    df = pd.DataFrame(
        [
            {
                "platform": "kalshi",
                "platform_type": "prediction_market",
                "metric_date": "2026-06-01",
                "geography": "US",
                "volume": 100.0,
                "volume_basis": "24h",
                "revenue": None,
                "liquidity_or_oi": 10.0,
                "market_id": "ABC",
                "market_title": "Example",
                "category": "sports",
                "implied_probability": 0.55,
                "raw_price": 55.0,
                "volume_24h": 100.0,
                "volume_total": 4000.0,
                "open_interest": 10.0,
                "status": "open",
                "resolution": None,
                "source_url": "",
                "collected_at_utc": "",
                "collector_version": "",
                "volume_is_onchain_derived": False,
            }
        ]
    )

    try:
        assert_valid_prediction_schema(df)
        assert False, "Expected schema validation to fail for empty provenance fields"
    except AssertionError:
        assert True


def test_build_prediction_dataframe_orders_columns() -> None:
    row = {
        "platform": "kalshi",
        "platform_type": "prediction_market",
        "metric_date": "2026-06-01",
        "geography": "US",
        "volume": 100.0,
        "volume_basis": "24h",
        "revenue": None,
        "liquidity_or_oi": 10.0,
        "market_id": "ABC",
        "market_title": "Example",
        "category": "sports",
        "implied_probability": 0.55,
        "raw_price": 55.0,
        "volume_24h": 100.0,
        "volume_total": 4000.0,
        "open_interest": 10.0,
        "status": "open",
        "resolution": None,
        "source_url": "https://example.com",
        "collected_at_utc": "2026-06-01T00:00:00+00:00",
        "collector_version": "abc1234",
        "volume_is_onchain_derived": False,
    }
    out = build_prediction_dataframe([row])
    assert list(out.columns) == EXPECTED_COLUMNS
    assert len(out) == 1
