from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from collector_common import (
    PoliteClient,
    add_provenance,
    configure_logger,
    ensure_project_dirs,
    load_config,
    save_raw_json,
)

SNAPSHOT_OUTPUT_PATH = "processed/pm_snapshot.csv"
MONTHLY_OUTPUT_PATH = "processed/pm_monthly.csv"

PROVENANCE_COLUMNS = ["source_url", "collected_at_utc", "collector_version"]

SNAPSHOT_COLUMNS = [
    "platform",
    "platform_type",
    "market_id",
    "title",
    "category",
    "category_group",
    "implied_probability",
    "raw_price",
    "volume_24h",
    "volume_total",
    "liquidity_or_oi",
    "status",
    "source_url",
    "collected_at_utc",
    "collector_version",
]

MONTHLY_COLUMNS = [
    "platform",
    "platform_type",
    "month_end",
    "category_group",
    "market_count",
    "volume_24h",
    "volume_total",
    "liquidity_or_oi",
    "volume_basis",
    "source_url",
    "collected_at_utc",
    "collector_version",
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
        cleaned = value.strip().replace(",", "")
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def parse_market_timestamp(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        unit = "ms" if value > 10_000_000_000 else "s"
        parsed = pd.to_datetime(value, unit=unit, errors="coerce", utc=True)
        if pd.isna(parsed):
            return None
        return parsed.to_pydatetime()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.isdigit():
            as_num = float(stripped)
            unit = "ms" if as_num > 10_000_000_000 else "s"
            parsed = pd.to_datetime(as_num, unit=unit, errors="coerce", utc=True)
            if pd.isna(parsed):
                return None
            return parsed.to_pydatetime()
        parsed = pd.to_datetime(stripped, errors="coerce", utc=True)
        if pd.isna(parsed):
            return None
        return parsed.to_pydatetime()
    return None


def extract_market_datetime(market: Dict[str, Any]) -> Optional[datetime]:
    for key in [
        "close_time",
        "expiration_time",
        "settlement_time",
        "settled_time",
        "resolved_time",
        "endDate",
        "end_date",
        "endDateIso",
        "closedTime",
        "resolveTime",
        "event_start_time",
        "event_date",
        "created_at",
    ]:
        parsed = parse_market_timestamp(market.get(key))
        if parsed is not None:
            return parsed
    return None


def parse_outcome_prices(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, list):
        out = [to_float(v) for v in value]
        vals = [v for v in out if v is not None]
        return vals or None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, list):
            out = [to_float(v) for v in parsed]
            vals = [v for v in out if v is not None]
            return vals or None
    return None


def normalize_probability(raw_price: Optional[float], explicit_probability: Optional[float]) -> Optional[float]:
    if explicit_probability is not None:
        return explicit_probability / 100.0 if explicit_probability > 1 else explicit_probability
    if raw_price is None:
        return None
    return raw_price / 100.0 if raw_price > 1 else raw_price


def canonical_key(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def resolve_category(platform: str, market: Dict[str, Any]) -> str:
    if platform == "kalshi":
        raw = first_of(market, ["category", "series_ticker", "event_ticker", "subtitle"], "unknown")
        return str(raw)

    raw = first_of(market, ["category", "series", "tag", "slug", "question"], "unknown")
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and item.strip():
                return item
            if isinstance(item, dict):
                label = first_of(item, ["label", "name", "slug"])
                if label:
                    return str(label)
    return str(raw)


def resolve_category_group(category: str, category_group_map: Dict[str, str]) -> str:
    key = canonical_key(category)
    if key in category_group_map:
        return category_group_map[key]
    # Handle provider category strings that include sports tokens in a larger identifier.
    for token, group in category_group_map.items():
        if group == "sports_event" and token and token in key:
            return "sports_event"
    return "other"


def check_page_cap(page_num: int, max_pages: int, context: str) -> None:
    if page_num > max_pages:
        raise RuntimeError(
            f"Pagination exceeded max_pages={max_pages} in {context}. "
            "Aborting to prevent unbounded collection."
        )


def _extract_kalshi_page(payload: Any) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if not isinstance(payload, dict):
        return [], None
    markets = payload.get("markets")
    if not isinstance(markets, list):
        return [], None
    cursor = payload.get("cursor")
    return markets, str(cursor) if cursor else None


def _extract_polymarket_page(payload: Any) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if isinstance(payload, list):
        return payload, None
    if isinstance(payload, dict):
        if isinstance(payload.get("markets"), list):
            next_cursor = payload.get("next_cursor")
            return payload["markets"], str(next_cursor) if next_cursor else None
        if isinstance(payload.get("data"), list):
            next_cursor = payload.get("next_cursor")
            return payload["data"], str(next_cursor) if next_cursor else None
    return [], None


def should_stop_settled_pagination(markets: Sequence[Dict[str, Any]], lower_bound: datetime) -> bool:
    timestamps = [extract_market_datetime(market) for market in markets]
    timestamps = [ts for ts in timestamps if ts is not None]
    if not timestamps:
        return False
    oldest = min(timestamps)
    return oldest < lower_bound


def normalize_snapshot_row(
    platform: str,
    market: Dict[str, Any],
    source_url: str,
    category_group_map: Dict[str, str],
    status_override: Optional[str] = None,
) -> Dict[str, Any]:
    if platform == "kalshi":
        raw_price = to_float(first_of(market, ["yes_price", "yes_bid", "yes_ask", "last_price", "last_trade_price"]))
        implied = normalize_probability(raw_price, to_float(first_of(market, ["implied_probability"])))
        volume_24h = to_float(first_of(market, ["volume_24h", "volume24h"]))
        volume_total = to_float(first_of(market, ["volume", "total_volume", "volume_total"]))
        liquidity_or_oi = to_float(first_of(market, ["open_interest", "liquidity"]))
        status = status_override or str(first_of(market, ["status"], "unknown"))
        market_id = str(first_of(market, ["ticker", "id"], ""))
        title = str(first_of(market, ["title", "subtitle"], ""))
    else:
        outcome_prices = parse_outcome_prices(first_of(market, ["outcomePrices", "outcome_prices"]))
        raw_price = to_float(first_of(market, ["lastTradePrice", "last_price"]))
        if raw_price is None and outcome_prices:
            raw_price = outcome_prices[0]
        implied = normalize_probability(raw_price, to_float(first_of(market, ["probability", "implied_probability"])))
        volume_24h = to_float(first_of(market, ["volume24hr", "volume_24h", "volume24h"]))
        volume_total = to_float(first_of(market, ["volumeNum", "volume", "volume_total"]))
        liquidity_or_oi = to_float(first_of(market, ["openInterest", "open_interest", "liquidityNum", "liquidity"]))
        status = status_override or ("closed" if bool(first_of(market, ["closed"], False)) else "open")
        market_id = str(first_of(market, ["id", "conditionId", "slug"], ""))
        title = str(first_of(market, ["question", "title"], ""))

    category = resolve_category(platform, market)
    row = {
        "platform": platform,
        "platform_type": "prediction_market",
        "market_id": market_id,
        "title": title,
        "category": category,
        "category_group": resolve_category_group(category, category_group_map),
        "implied_probability": implied,
        "raw_price": raw_price,
        "volume_24h": volume_24h,
        "volume_total": volume_total,
        "liquidity_or_oi": liquidity_or_oi,
        "status": str(status),
    }
    return add_provenance(row, source_url=source_url)


def _filter_window(ts: Optional[datetime], start_dt: Optional[datetime], end_dt: Optional[datetime]) -> bool:
    if ts is None:
        return False
    if start_dt and ts < start_dt:
        return False
    if end_dt and ts > end_dt:
        return False
    return True


def discover_kalshi_sports_series_tickers(
    client: PoliteClient,
    base_url: str,
    logger,
) -> List[str]:
    tickers: List[str] = []
    cursor: Optional[str] = None
    page_num = 1

    while True:
        params: Dict[str, Any] = {"limit": 5000}
        if cursor:
            params["cursor"] = cursor

        response = client.get(url=f"{base_url}/series", params=params, use_cache=True, robots_required=False)
        payload = json.loads(response.body.decode("utf-8"))
        save_raw_json(source=f"kalshi_series_sports_p{page_num}", payload=payload)

        series = payload.get("series", []) if isinstance(payload, dict) else []
        if not isinstance(series, list):
            series = []

        for item in series:
            if not isinstance(item, dict):
                continue
            ticker = str(item.get("ticker") or "").strip()
            category = str(item.get("category") or "").strip().lower()
            if not ticker or category != "sports":
                continue
            tickers.append(ticker)

        cursor = str(payload.get("cursor") or "") if isinstance(payload, dict) else ""
        if not cursor:
            break
        page_num += 1

    unique = sorted(set(tickers))
    logger.info("kalshi sports series discovered=%s", len(unique))
    return unique


def fetch_kalshi_rows(
    client: PoliteClient,
    cfg: Dict[str, Any],
    logger,
    *,
    mode: str,
    max_pages: int,
    category_group_map: Dict[str, str],
    settled_start: Optional[datetime],
    settled_end: Optional[datetime],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    base = cfg["base_url"].rstrip("/")
    endpoint = cfg["markets_endpoint"]
    base_params = dict(cfg.get("default_params", {}))
    statuses = ["open", "settled"]

    rows: List[Dict[str, Any]] = []
    page_total = 0
    market_total = 0
    sports_series_tickers: Optional[List[str]] = None
    sports_series_lookup: set[str] = set()

    if mode in {"monthly", "snapshot"}:
        sports_series_tickers = discover_kalshi_sports_series_tickers(
            client,
            base,
            logger,
        )
        sports_series_lookup = set(sports_series_tickers)

    for status in statuses:
        if status == "settled" and mode == "monthly":
            series_scope: List[Optional[str]] = sports_series_tickers or [None]
        else:
            series_scope = [None]

        for series_ticker in series_scope:
            cursor: Optional[str] = None
            page_num = 1
            while True:
                context = f"kalshi:{status}:{mode}" if series_ticker is None else f"kalshi:{status}:{mode}:{series_ticker}"
                check_page_cap(page_num, max_pages, context)

                params = dict(base_params)
                params["status"] = status
                if mode == "monthly" and status == "settled" and settled_end is not None:
                    params["max_close_ts"] = int(settled_end.timestamp())
                if series_ticker:
                    params["series_ticker"] = series_ticker
                if cursor:
                    params["cursor"] = cursor

                response = client.get(url=f"{base}{endpoint}", params=params, use_cache=True, robots_required=False)
                payload = json.loads(response.body.decode("utf-8"))
                series_suffix = "" if not series_ticker else f"_{series_ticker.lower()}"
                save_raw_json(source=f"kalshi_{mode}_{status}{series_suffix}_p{page_num}", payload=payload)

                markets, next_cursor = _extract_kalshi_page(payload)
                page_total += 1
                market_total += len(markets)
                logger.info(
                    "kalshi mode=%s status=%s series=%s page=%s rows=%s url=%s http=%s",
                    mode,
                    status,
                    series_ticker,
                    page_num,
                    len(markets),
                    response.url,
                    response.status_code,
                )

                for market in markets:
                    market_ts = extract_market_datetime(market)
                    if status == "settled" and not _filter_window(market_ts, settled_start, settled_end):
                        continue
                    row = normalize_snapshot_row(
                        platform="kalshi",
                        market=market,
                        source_url=response.url,
                        category_group_map=category_group_map,
                        status_override=status,
                    )
                    if mode == "monthly" and status == "settled" and series_ticker:
                        # Settled markets are fetched from sports-only series in monthly mode.
                        row["category_group"] = "sports_event"
                    elif mode == "snapshot" and sports_series_lookup:
                        event_ticker = str(
                            first_of(market, ["event_ticker", "eventTicker", "series_ticker", "seriesTicker"], "")
                        ).strip()
                        series_ticker_guess = event_ticker.split("-")[0] if event_ticker else ""
                        if series_ticker_guess and series_ticker_guess in sports_series_lookup:
                            row["category_group"] = "sports_event"
                    row["__market_ts"] = market_ts
                    rows.append(row)

                if status == "open":
                    if next_cursor:
                        logger.info("kalshi mode=%s status=open single-pass stop after page=%s", mode, page_num)
                    break

                if status == "settled" and mode == "snapshot":
                    if next_cursor:
                        logger.info("kalshi mode=%s status=settled single-pass stop after page=%s", mode, page_num)
                    break

                if status == "settled" and settled_start and should_stop_settled_pagination(markets, settled_start):
                    logger.info(
                        "kalshi mode=%s status=settled series=%s stop_at_window page=%s lower_bound=%s",
                        mode,
                        series_ticker,
                        page_num,
                        settled_start.isoformat(),
                    )
                    break

                if not next_cursor:
                    break

                cursor = next_cursor
                page_num += 1

    logger.info("kalshi mode=%s pages=%s markets_seen=%s rows_kept=%s", mode, page_total, market_total, len(rows))
    return rows, {"pages": page_total, "markets_seen": market_total, "rows_kept": len(rows)}


def fetch_polymarket_rows(
    client: PoliteClient,
    cfg: Dict[str, Any],
    logger,
    *,
    mode: str,
    max_pages: int,
    category_group_map: Dict[str, str],
    settled_start: Optional[datetime],
    settled_end: Optional[datetime],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    gamma_base = cfg["gamma_base_url"].rstrip("/")
    endpoint = cfg["markets_endpoint"]
    base_params = dict(cfg.get("default_params", {}))
    scenarios = [("open", {"closed": False}), ("settled", {"closed": True})]

    rows: List[Dict[str, Any]] = []
    page_total = 0
    market_total = 0

    for status, scenario_params in scenarios:
        cursor: Optional[str] = None
        page_num = 1
        while True:
            check_page_cap(page_num, max_pages, f"polymarket:{status}:{mode}")
            params = dict(base_params)
            params.update(scenario_params)
            if cursor:
                params["after_cursor"] = cursor

            response = client.get(url=f"{gamma_base}{endpoint}", params=params, use_cache=True, robots_required=False)
            payload = json.loads(response.body.decode("utf-8"))
            save_raw_json(source=f"polymarket_{mode}_{status}_p{page_num}", payload=payload)

            markets, next_cursor = _extract_polymarket_page(payload)
            page_total += 1
            market_total += len(markets)
            logger.info(
                "polymarket mode=%s status=%s page=%s rows=%s url=%s http=%s",
                mode,
                status,
                page_num,
                len(markets),
                response.url,
                response.status_code,
            )

            for market in markets:
                market_ts = extract_market_datetime(market)
                if status == "settled" and not _filter_window(market_ts, settled_start, settled_end):
                    continue
                row = normalize_snapshot_row(
                    platform="polymarket",
                    market=market,
                    source_url=response.url,
                    category_group_map=category_group_map,
                    status_override=status,
                )
                row["__market_ts"] = market_ts
                rows.append(row)

            if status == "open":
                if next_cursor:
                    logger.info("polymarket mode=%s status=open single-pass stop after page=%s", mode, page_num)
                break

            if status == "settled" and mode == "snapshot":
                if next_cursor:
                    logger.info("polymarket mode=%s status=settled single-pass stop after page=%s", mode, page_num)
                break

            if status == "settled" and settled_start and should_stop_settled_pagination(markets, settled_start):
                logger.info(
                    "polymarket mode=%s status=settled stop_at_window page=%s lower_bound=%s",
                    mode,
                    page_num,
                    settled_start.isoformat(),
                )
                break

            if not next_cursor:
                break

            cursor = next_cursor
            page_num += 1

    logger.info("polymarket mode=%s pages=%s markets_seen=%s rows_kept=%s", mode, page_total, market_total, len(rows))
    return rows, {"pages": page_total, "markets_seen": market_total, "rows_kept": len(rows)}


def build_snapshot_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in SNAPSHOT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[SNAPSHOT_COLUMNS]
    assert_provenance_populated(df, "pm_snapshot")
    return df


def build_monthly_dataframe(snapshot_rows: List[Dict[str, Any]], start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    frame = pd.DataFrame(snapshot_rows)
    if frame.empty:
        df = pd.DataFrame(columns=MONTHLY_COLUMNS)
        return df

    if "__market_ts" not in frame.columns:
        raise AssertionError("monthly build requires __market_ts in snapshot rows")

    frame = frame[frame["category_group"] == "sports_event"].copy()
    frame = frame[frame["__market_ts"].notna()].copy()
    frame = frame[(frame["__market_ts"] >= start_dt) & (frame["__market_ts"] <= end_dt)].copy()

    if frame.empty:
        return pd.DataFrame(columns=MONTHLY_COLUMNS)

    frame["month_end"] = pd.to_datetime(frame["__market_ts"], utc=True).dt.tz_convert(None).dt.to_period("M").dt.to_timestamp("M")

    grouped = frame.groupby(["platform", "platform_type", "month_end", "category_group"], as_index=False).agg(
        market_count=("market_id", "nunique"),
        volume_24h=("volume_24h", "sum"),
        volume_total=("volume_total", "sum"),
        liquidity_or_oi=("liquidity_or_oi", "sum"),
        source_url=("source_url", lambda s: ";".join(sorted(set(str(v) for v in s if str(v).strip())))),
    )
    grouped["month_end"] = grouped["month_end"].dt.date.astype(str)
    grouped["volume_basis"] = "monthly_month_end_snapshot"

    rows: List[Dict[str, Any]] = []
    for _, row in grouped.iterrows():
        record = {
            "platform": row["platform"],
            "platform_type": row["platform_type"],
            "month_end": row["month_end"],
            "category_group": row["category_group"],
            "market_count": int(row["market_count"]),
            "volume_24h": to_float(row["volume_24h"]),
            "volume_total": to_float(row["volume_total"]),
            "liquidity_or_oi": to_float(row["liquidity_or_oi"]),
            "volume_basis": row["volume_basis"],
        }
        rows.append(add_provenance(record, source_url=row["source_url"]))

    out = pd.DataFrame(rows)
    for col in MONTHLY_COLUMNS:
        if col not in out.columns:
            out[col] = None
    out = out[MONTHLY_COLUMNS]
    assert_provenance_populated(out, "pm_monthly")
    return out


def assert_provenance_populated(df: pd.DataFrame, label: str) -> None:
    missing = [col for col in PROVENANCE_COLUMNS if col not in df.columns]
    if missing:
        raise AssertionError(f"{label} missing provenance columns: {missing}")
    for col in PROVENANCE_COLUMNS:
        if df[col].isna().any() or (df[col].astype(str).str.strip() == "").any():
            raise AssertionError(f"{label} has null/blank values in {col}")


def parse_series_bound(value: Any, *, bound: str) -> datetime:
    raw = str(value).strip()
    if not raw:
        raise ValueError(f"prediction_markets.series_{bound} must be a non-empty date")

    if re.fullmatch(r"\d{4}-\d{2}", raw):
        parsed = pd.to_datetime(f"{raw}-01", utc=True, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"prediction_markets.series_{bound} must be a valid ISO date")
        if bound == "start":
            out = parsed
        else:
            out = parsed + pd.offsets.MonthEnd(0)
            out = out.replace(hour=23, minute=59, second=59, microsecond=0)
        return out.to_pydatetime()

    parsed = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"prediction_markets.series_{bound} must be a valid ISO date")

    # If caller provided a date without time for the end bound, include the full day.
    if bound == "end" and re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        parsed = parsed.replace(hour=23, minute=59, second=59, microsecond=0)
    return parsed.to_pydatetime()


def load_mode_config(config: Dict[str, Any]) -> Tuple[Dict[str, str], int, int, datetime, datetime]:
    pm_cfg = config["prediction_markets"]
    category_group_map_raw = pm_cfg.get("category_group_map", {})
    category_group_map = {canonical_key(str(k)): str(v) for k, v in category_group_map_raw.items()}
    if not category_group_map:
        category_group_map = {"sports": "sports_event"}

    snapshot_lookback_days = int(pm_cfg.get("snapshot_lookback_days", 90))
    max_pages = int(pm_cfg.get("max_pages", 25))

    series_start = parse_series_bound(pm_cfg.get("series_start"), bound="start")
    series_end = parse_series_bound(pm_cfg.get("series_end"), bound="end")

    return (
        category_group_map,
        snapshot_lookback_days,
        max_pages,
        series_start,
        series_end,
    )


def collect_snapshot_mode(client: PoliteClient, config: Dict[str, Any], logger) -> pd.DataFrame:
    pm_cfg = config["prediction_markets"]
    category_group_map, snapshot_lookback_days, max_pages, _, _ = load_mode_config(config)

    now_utc = datetime.now(timezone.utc)
    settled_start = now_utc - timedelta(days=snapshot_lookback_days)

    kalshi_rows, kalshi_stats = fetch_kalshi_rows(
        client,
        pm_cfg["kalshi"],
        logger,
        mode="snapshot",
        max_pages=max_pages,
        category_group_map=category_group_map,
        settled_start=settled_start,
        settled_end=now_utc,
    )
    polymarket_rows, polymarket_stats = fetch_polymarket_rows(
        client,
        pm_cfg["polymarket"],
        logger,
        mode="snapshot",
        max_pages=max_pages,
        category_group_map=category_group_map,
        settled_start=settled_start,
        settled_end=now_utc,
    )

    rows = kalshi_rows + polymarket_rows
    df = build_snapshot_dataframe(rows)
    df.to_csv(SNAPSHOT_OUTPUT_PATH, index=False)
    logger.info(
        "snapshot_complete rows=%s output=%s kalshi_pages=%s polymarket_pages=%s",
        len(df),
        SNAPSHOT_OUTPUT_PATH,
        kalshi_stats["pages"],
        polymarket_stats["pages"],
    )
    return df


def collect_monthly_mode(client: PoliteClient, config: Dict[str, Any], logger) -> pd.DataFrame:
    pm_cfg = config["prediction_markets"]
    category_group_map, _, max_pages, series_start, series_end = load_mode_config(config)

    kalshi_rows, kalshi_stats = fetch_kalshi_rows(
        client,
        pm_cfg["kalshi"],
        logger,
        mode="monthly",
        max_pages=max_pages,
        category_group_map=category_group_map,
        settled_start=series_start,
        settled_end=series_end,
    )
    polymarket_rows, polymarket_stats = fetch_polymarket_rows(
        client,
        pm_cfg["polymarket"],
        logger,
        mode="monthly",
        max_pages=max_pages,
        category_group_map=category_group_map,
        settled_start=series_start,
        settled_end=series_end,
    )

    monthly_df = build_monthly_dataframe(kalshi_rows + polymarket_rows, start_dt=series_start, end_dt=series_end)
    monthly_df.to_csv(MONTHLY_OUTPUT_PATH, index=False)
    logger.info(
        "monthly_complete rows=%s output=%s kalshi_pages=%s polymarket_pages=%s",
        len(monthly_df),
        MONTHLY_OUTPUT_PATH,
        kalshi_stats["pages"],
        polymarket_stats["pages"],
    )
    return monthly_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded prediction market collector")
    parser.add_argument(
        "--mode",
        choices=["snapshot", "monthly"],
        default="snapshot",
        help="snapshot=structural market snapshot, monthly=sports_event monthly time series",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    config = load_config("config.yaml")
    logger, log_path = configure_logger("collect_prediction_markets")
    client = PoliteClient(config=config, logger=logger)
    logger.info("Run started log_path=%s mode=%s", log_path, args.mode)

    if args.mode == "snapshot":
        collect_snapshot_mode(client, config, logger)
    else:
        collect_monthly_mode(client, config, logger)

    logger.info("Run finished mode=%s", args.mode)


if __name__ == "__main__":
    main()
