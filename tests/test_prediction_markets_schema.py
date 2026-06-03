from datetime import datetime, timezone

import pandas as pd
import pytest

from collect_prediction_markets import (
    MONTHLY_COLUMNS,
    SNAPSHOT_COLUMNS,
    build_monthly_dataframe,
    build_snapshot_dataframe,
    check_page_cap,
    resolve_category_group,
)


def test_check_page_cap_raises_when_exceeded() -> None:
    with pytest.raises(RuntimeError):
        check_page_cap(page_num=3, max_pages=2, context="unit-test")


def test_resolve_category_group_defaults_to_other() -> None:
    mapping = {"sports": "sports_event"}
    assert resolve_category_group("Sports", mapping) == "sports_event"
    assert resolve_category_group("Politics", mapping) == "other"


def test_build_snapshot_dataframe_requires_provenance() -> None:
    row = {
        "platform": "kalshi",
        "platform_type": "prediction_market",
        "market_id": "ABC",
        "title": "Example",
        "category": "sports",
        "category_group": "sports_event",
        "implied_probability": 0.51,
        "raw_price": 51.0,
        "volume_24h": 100.0,
        "volume_total": 1000.0,
        "liquidity_or_oi": 40.0,
        "status": "open",
        "source_url": "",
        "collected_at_utc": "",
        "collector_version": "",
    }

    with pytest.raises(AssertionError):
        build_snapshot_dataframe([row])


def test_build_monthly_dataframe_outputs_expected_columns() -> None:
    rows = [
        {
            "platform": "kalshi",
            "platform_type": "prediction_market",
            "market_id": "A",
            "title": "A",
            "category": "sports",
            "category_group": "sports_event",
            "implied_probability": 0.6,
            "raw_price": 0.6,
            "volume_24h": 10.0,
            "volume_total": 40.0,
            "liquidity_or_oi": 5.0,
            "status": "open",
            "source_url": "https://example.com/a",
            "collected_at_utc": "2026-01-01T00:00:00+00:00",
            "collector_version": "abc123",
            "__market_ts": datetime(2026, 1, 15, tzinfo=timezone.utc),
        },
        {
            "platform": "polymarket",
            "platform_type": "prediction_market",
            "market_id": "B",
            "title": "B",
            "category": "politics",
            "category_group": "other",
            "implied_probability": 0.4,
            "raw_price": 0.4,
            "volume_24h": 12.0,
            "volume_total": 80.0,
            "liquidity_or_oi": 7.0,
            "status": "settled",
            "source_url": "https://example.com/b",
            "collected_at_utc": "2026-01-01T00:00:00+00:00",
            "collector_version": "abc123",
            "__market_ts": datetime(2026, 1, 20, tzinfo=timezone.utc),
        },
    ]

    out = build_monthly_dataframe(
        rows,
        start_dt=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end_dt=datetime(2026, 12, 31, tzinfo=timezone.utc),
    )

    assert list(out.columns) == MONTHLY_COLUMNS
    assert len(out) == 1
    assert out.iloc[0]["category_group"] == "sports_event"


def test_snapshot_columns_constant() -> None:
    assert SNAPSHOT_COLUMNS[:4] == ["platform", "platform_type", "market_id", "title"]
