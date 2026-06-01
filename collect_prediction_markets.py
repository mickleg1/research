from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

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

# Shared schema core block (Section 4 of spec), followed by platform-specific extras.
CORE_COLUMNS = [
    "platform",
    "platform_type",
    "metric_date",
    "geography",
    "volume",
    "volume_basis",
    "revenue",
    "liquidity_or_oi",
    "source_url",
    "collected_at_utc",
    "collector_version",
]

EXTRA_COLUMNS = [
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
    # True where value is explicitly API-surfaced as on-chain-derived market data
    # (Polymarket Gamma/Data APIs index blockchain state).
    "volume_is_onchain_derived",
]

EXPECTED_COLUMNS = CORE_COLUMNS + EXTRA_COLUMNS
PROVENANCE_COLUMNS = ["source_url", "collected_at_utc", "collector_version"]


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
    base_params = dict(cfg.get("default_params", {}))
    statuses = ["unopened", "open", "paused", "closed", "settled"]

    rows: List[Dict[str, Any]] = []
    seen_market_ids: Set[str] = set()

    for status in statuses:
        cursor: Optional[str] = None
        page_num = 1
        while True:
            params = dict(base_params)
            params["status"] = status
            if cursor:
                params["cursor"] = cursor

            response = client.get(url=f"{base}{endpoint}", params=params, use_cache=True, robots_required=False)
            payload = json.loads(response.body.decode("utf-8"))
            save_raw_json(source=f"kalshi_markets_{status}_p{page_num}", payload=payload)

            markets = payload.get("markets", []) if isinstance(payload, dict) else []
            logger.info(
                "kalshi status=%s page=%s rows=%s url=%s http=%s",
                status,
                page_num,
                len(markets),
                response.url,
                response.status_code,
            )
            for market in markets:
                market_id = str(first_of(market, ["ticker", "id"], ""))
                if not market_id:
                    continue
                if market_id in seen_market_ids:
                    continue
                seen_market_ids.add(market_id)
                rows.append(normalize_kalshi_market(market, source_url=response.url))

            cursor = payload.get("cursor") if isinstance(payload, dict) else None
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
            logger.warning("Failed to decode Polymarket OI response for chunk start=%s", idx)
            continue
        save_raw_json(source=f"polymarket_oi_{idx}", payload=payload)
        if isinstance(payload, list):
            for item in payload:
                market = item.get("market")
                value = to_float(item.get("value"))
                if market and value is not None:
                    out[market] = value
        logger.info(
            "polymarket_oi chunk_start=%s chunk_size=%s rows=%s url=%s http=%s",
            idx,
            len(chunk),
            len(payload) if isinstance(payload, list) else 0,
            response.url,
            response.status_code,
        )
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


def _fetch_polymarket_scenario(
    client: PoliteClient,
    gamma_base: str,
    endpoint: str,
    base_params: Dict[str, Any],
    scenario_name: str,
    scenario_params: Dict[str, Any],
    logger,
) -> List[Dict[str, Any]]:
    cursor: Optional[str] = None
    page_num = 1
    items_out: List[Dict[str, Any]] = []

    while True:
        params = dict(base_params)
        params.update(scenario_params)
        if cursor:
            params["after_cursor"] = cursor
        response = client.get(url=f"{gamma_base}{endpoint}", params=params, use_cache=True, robots_required=False)
        payload = json.loads(response.body.decode("utf-8"))
        save_raw_json(source=f"polymarket_markets_{scenario_name}_p{page_num}", payload=payload)
        items, next_cursor = _extract_polymarket_items(payload)
        logger.info(
            "polymarket scenario=%s page=%s rows=%s url=%s http=%s",
            scenario_name,
            page_num,
            len(items),
            response.url,
            response.status_code,
        )
        items_out.extend(items)
        page_num += 1
        cursor = next_cursor
        if not cursor:
            break

    return items_out


def fetch_polymarket_markets(client: PoliteClient, cfg: Dict[str, Any], logger) -> List[Dict[str, Any]]:
    gamma_base = cfg["gamma_base_url"].rstrip("/")
    endpoint = cfg["markets_endpoint"]
    base_params = dict(cfg.get("default_params", {}))

    # Gamma commonly defaults to open/active markets; gather both open and closed
    # snapshots to satisfy “every available market” in the spec.
    scenarios = [
        ("open", {"closed": False}),
        ("closed", {"closed": True}),
    ]

    collected: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    condition_ids: Set[str] = set()

    for scenario_name, scenario_params in scenarios:
        items = _fetch_polymarket_scenario(
            client=client,
            gamma_base=gamma_base,
            endpoint=endpoint,
            base_params=base_params,
            scenario_name=scenario_name,
            scenario_params=scenario_params,
            logger=logger,
        )
        for item in items:
            market_id = str(first_of(item, ["id", "conditionId", "slug"], ""))
            if not market_id:
                continue
            if market_id in seen_ids:
                continue
            seen_ids.add(market_id)
            collected.append(item)
            cid = first_of(item, ["conditionId", "condition_id"])
            if cid:
                condition_ids.add(str(cid))

    oi_lookup = fetch_polymarket_open_interest(
        client=client,
        data_api_base=cfg["data_base_url"],
        condition_ids=sorted(condition_ids),
        logger=logger,
    )
    return [normalize_polymarket_market(item, source_url=f"{gamma_base}{endpoint}", oi_lookup=oi_lookup) for item in collected]


def assert_valid_prediction_schema(df: pd.DataFrame) -> None:
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise AssertionError(f"Missing columns: {missing}")
    for col in PROVENANCE_COLUMNS:
        if df[col].isna().any() or (df[col].astype(str).str.strip() == "").any():
            raise AssertionError(f"{col} must be non-null and non-empty")


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
    logger.info(
        "Wrote rows=%s output=%s kalshi_rows=%s polymarket_rows=%s",
        len(df),
        OUTPUT_PATH,
        len(kalshi_rows),
        len(polymarket_rows),
    )


if __name__ == "__main__":
    main()
