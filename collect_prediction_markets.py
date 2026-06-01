from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from collector_common import (
    PoliteClient,
    add_provenance,
    configure_logger,
    ensure_project_dirs,
    load_config,
    save_raw_json,
)


OUTPUT_PATH = "processed/prediction_markets.csv"

EXPECTED_COLUMNS = [
    "platform",
    "platform_type",
    "metric_date",
    "geography",
    "volume",
    "volume_basis",
    "revenue",
    "liquidity_or_oi",
    "market_id",
    "market_title",
    "category",
    "implied_probability",
    "raw_price",
    "volume_24h",
    "volume_total",
    "open_interest",
    "status",
    "resolution",
    "source_url",
    "collected_at_utc",
    "collector_version",
    "volume_is_onchain_derived",
]


def first_of(mapping: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def parse_outcome_prices(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, list):
        out = [to_float(v) for v in value]
        return [v for v in out if v is not None] or None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                out = [to_float(v) for v in parsed]
                return [v for v in out if v is not None] or None
        except json.JSONDecodeError:
            return None
    return None


def normalize_probability(raw_price: Optional[float], explicit_prob: Optional[float]) -> Optional[float]:
    if explicit_prob is not None:
        if explicit_prob > 1:
            return explicit_prob / 100.0
        return explicit_prob
    if raw_price is None:
        return None
    if raw_price > 1:
        return raw_price / 100.0
    return raw_price


def normalize_kalshi_market(market: Dict[str, Any], source_url: str) -> Dict[str, Any]:
    raw_price = to_float(
        first_of(
            market,
            [
                "yes_price",
                "yes_bid",
                "yes_ask",
                "last_price",
                "last_trade_price",
            ],
        )
    )
    implied = normalize_probability(
        raw_price=raw_price,
        explicit_prob=to_float(first_of(market, ["implied_probability"])),
    )
    volume_24h = to_float(first_of(market, ["volume_24h", "volume24h"]))
    volume_total = to_float(first_of(market, ["volume", "total_volume", "volume_total"]))
    open_interest = to_float(first_of(market, ["open_interest", "liquidity"]))

    volume = volume_24h if volume_24h is not None else volume_total
    volume_basis = "24h" if volume_24h is not None else ("total" if volume_total is not None else "unknown")

    row = {
        "platform": "kalshi",
        "platform_type": "prediction_market",
        "metric_date": datetime.now(timezone.utc).date().isoformat(),
        "geography": "US",
        "volume": volume,
        "volume_basis": volume_basis,
        "revenue": None,
        "liquidity_or_oi": open_interest,
        "market_id": first_of(market, ["ticker", "id"]),
        "market_title": first_of(market, ["title", "subtitle"]),
        "category": first_of(market, ["category", "series_ticker", "event_ticker"]),
        "implied_probability": implied,
        "raw_price": raw_price,
        "volume_24h": volume_24h,
        "volume_total": volume_total,
        "open_interest": open_interest,
        "status": first_of(market, ["status"]),
        "resolution": first_of(market, ["result", "settlement_result", "resolution"]),
        "volume_is_onchain_derived": False,
    }
    return add_provenance(row, source_url=source_url)


def fetch_kalshi_markets(client: PoliteClient, cfg: Dict[str, Any], logger) -> List[Dict[str, Any]]:
    base = cfg["base_url"].rstrip("/")
    endpoint = cfg["markets_endpoint"]
    params = dict(cfg.get("default_params", {}))
    cursor: Optional[str] = None
    page_num = 1
    rows: List[Dict[str, Any]] = []

    while True:
        page_params = dict(params)
        if cursor:
            page_params["cursor"] = cursor
        response = client.get(url=f"{base}{endpoint}", params=page_params, use_cache=True, robots_required=False)
        payload = json.loads(response.body.decode("utf-8"))
        save_raw_json(source=f"kalshi_markets_p{page_num}", payload=payload)

        markets = payload.get("markets", []) if isinstance(payload, dict) else []
        logger.info("kalshi page=%s rows=%s url=%s", page_num, len(markets), response.url)
        for market in markets:
            rows.append(normalize_kalshi_market(market, source_url=response.url))

        cursor = payload.get("cursor")
        page_num += 1
        if not cursor:
            break

    return rows


def _extract_polymarket_items(payload: Any) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if isinstance(payload, list):
        return payload, None
    if isinstance(payload, dict):
        if isinstance(payload.get("markets"), list):
            return payload.get("markets", []), payload.get("next_cursor")
        if isinstance(payload.get("data"), list):
            return payload.get("data", []), payload.get("next_cursor")
    return [], None


def fetch_polymarket_open_interest(
    client: PoliteClient,
    data_api_base: str,
    condition_ids: List[str],
    logger,
) -> Dict[str, float]:
    if not condition_ids:
        return {}

    out: Dict[str, float] = {}
    batch_size = 100
    for idx in range(0, len(condition_ids), batch_size):
        chunk = condition_ids[idx : idx + batch_size]
        params = {"market": ",".join(chunk)}
        response = client.get(
            url=f"{data_api_base.rstrip('/')}/oi",
            params=params,
            use_cache=True,
            robots_required=False,
        )
        try:
            payload = json.loads(response.body.decode("utf-8"))
        except Exception:
            logger.warning("Failed to decode Polymarket OI response for chunk starting at %s", idx)
            continue
        save_raw_json(source=f"polymarket_oi_{idx}", payload=payload)
        if isinstance(payload, list):
            for item in payload:
                market = item.get("market")
                value = to_float(item.get("value"))
                if market and value is not None:
                    out[market] = value
    return out


def normalize_polymarket_market(
    market: Dict[str, Any],
    source_url: str,
    oi_lookup: Dict[str, float],
) -> Dict[str, Any]:
    outcome_prices = parse_outcome_prices(first_of(market, ["outcomePrices", "outcome_prices"]))
    raw_price = to_float(first_of(market, ["lastTradePrice", "last_price"]))
    if raw_price is None and outcome_prices:
        raw_price = outcome_prices[0]

    implied = normalize_probability(
        raw_price=raw_price,
        explicit_prob=to_float(first_of(market, ["probability", "implied_probability"])),
    )

    volume_24h = to_float(first_of(market, ["volume24hr", "volume_24h", "volume24h"]))
    volume_total = to_float(first_of(market, ["volumeNum", "volume", "volume_total"]))
    open_interest = to_float(first_of(market, ["openInterest", "open_interest"]))
    if open_interest is None:
        condition_id = first_of(market, ["conditionId", "condition_id"])
        open_interest = oi_lookup.get(condition_id) if condition_id else None

    liquidity = to_float(first_of(market, ["liquidityNum", "liquidity"]))
    liquidity_or_oi = open_interest if open_interest is not None else liquidity

    volume = volume_24h if volume_24h is not None else volume_total
    volume_basis = "24h" if volume_24h is not None else ("total" if volume_total is not None else "unknown")

    row = {
        "platform": "polymarket",
        "platform_type": "prediction_market",
        "metric_date": datetime.now(timezone.utc).date().isoformat(),
        "geography": "US",
        "volume": volume,
        "volume_basis": volume_basis,
        "revenue": None,
        "liquidity_or_oi": liquidity_or_oi,
        "market_id": first_of(market, ["id", "conditionId", "slug"]),
        "market_title": first_of(market, ["question", "title"]),
        "category": first_of(market, ["category", "series", "tag"]),
        "implied_probability": implied,
        "raw_price": raw_price,
        "volume_24h": volume_24h,
        "volume_total": volume_total,
        "open_interest": open_interest,
        "status": first_of(market, ["status"]) or ("closed" if market.get("closed") else "open"),
        "resolution": first_of(market, ["resolution", "outcome", "resolvedOutcome"]),
        "volume_is_onchain_derived": True,
    }
    return add_provenance(row, source_url=source_url)


def fetch_polymarket_markets(client: PoliteClient, cfg: Dict[str, Any], logger) -> List[Dict[str, Any]]:
    gamma_base = cfg["gamma_base_url"].rstrip("/")
    endpoint = cfg["markets_endpoint"]
    params = dict(cfg.get("default_params", {}))
    cursor: Optional[str] = None
    page_num = 1
    collected: List[Dict[str, Any]] = []
    condition_ids: List[str] = []

    while True:
        page_params = dict(params)
        if cursor:
            page_params["after_cursor"] = cursor
        response = client.get(url=f"{gamma_base}{endpoint}", params=page_params, use_cache=True, robots_required=False)
        payload = json.loads(response.body.decode("utf-8"))
        save_raw_json(source=f"polymarket_markets_p{page_num}", payload=payload)

        items, next_cursor = _extract_polymarket_items(payload)
        logger.info("polymarket page=%s rows=%s url=%s", page_num, len(items), response.url)
        collected.extend(items)
        for item in items:
            cid = first_of(item, ["conditionId", "condition_id"])
            if cid:
                condition_ids.append(cid)

        page_num += 1
        cursor = next_cursor
        if not cursor:
            break

    oi_lookup = fetch_polymarket_open_interest(
        client=client,
        data_api_base=cfg["data_base_url"],
        condition_ids=sorted(set(condition_ids)),
        logger=logger,
    )

    rows = [normalize_polymarket_market(item, source_url=f"{gamma_base}{endpoint}", oi_lookup=oi_lookup) for item in collected]
    return rows


def assert_valid_prediction_schema(df: pd.DataFrame) -> None:
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise AssertionError(f"Missing columns: {missing}")
    if df["source_url"].isna().any() or (df["source_url"].astype(str).str.strip() == "").any():
        raise AssertionError("source_url must be non-null and non-empty")
    if df["collected_at_utc"].isna().any() or (df["collected_at_utc"].astype(str).str.strip() == "").any():
        raise AssertionError("collected_at_utc must be non-null and non-empty")
    if df["collector_version"].isna().any() or (df["collector_version"].astype(str).str.strip() == "").any():
        raise AssertionError("collector_version must be non-null and non-empty")


def build_prediction_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[EXPECTED_COLUMNS]
    assert_valid_prediction_schema(df)
    return df


def main() -> None:
    ensure_project_dirs()
    config = load_config("config.yaml")
    logger, log_path = configure_logger("collect_prediction_markets")
    client = PoliteClient(config=config, logger=logger)
    logger.info("Run started log_path=%s", log_path)

    kalshi_rows = fetch_kalshi_markets(client, config["prediction_markets"]["kalshi"], logger)
    polymarket_rows = fetch_polymarket_markets(client, config["prediction_markets"]["polymarket"], logger)
    rows = kalshi_rows + polymarket_rows

    df = build_prediction_dataframe(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info("Wrote %s rows to %s", len(df), OUTPUT_PATH)


if __name__ == "__main__":
    main()
