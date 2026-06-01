from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import pandas as pd
import pdfplumber

from collector_common import (
    PoliteClient,
    add_provenance,
    configure_logger,
    ensure_project_dirs,
    load_config,
    now_utc_stamp,
    save_raw_bytes,
)

OUTPUT_PATH = "processed/sportsbook_records.csv"
REPORT_LINK_PATTERN = re.compile(
    r'<a[^>]+href=[\'"]([^\'"]+\.(?:pdf|csv|xlsx|xls))[\'"][^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
MONTH_PATTERN = re.compile(r"(20\d{2})[-_ ]?(0[1-9]|1[0-2])")
REPORT_KEYWORDS = ("sports", "wager", "bet", "revenue", "handle", "gaming win", "ggr")

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
    "state",
    "month",
    "operator",
    "handle",
    "gross_revenue",
    "validation_status",
]

EXPECTED_COLUMNS = CORE_COLUMNS + EXTRA_COLUMNS


def canonical_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("$", "").replace("(", "-").replace(")", "").strip()
        if cleaned in {"", "-", "—", "N/A"}:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def parse_month_from_text(text: str) -> Optional[str]:
    match = MONTH_PATTERN.search(text)
    if not match:
        return None
    return f"{match.group(1)}-{match.group(2)}"


def discover_report_links(base_url: str, html: str) -> List[str]:
    links: List[str] = []
    for href, anchor_text in REPORT_LINK_PATTERN.findall(html):
        haystack = f"{href} {anchor_text}".lower()
        if not any(keyword in haystack for keyword in REPORT_KEYWORDS):
            continue
        links.append(urljoin(base_url, href))
    return sorted(set(links))


def state_raw_dir(state_code: str) -> Path:
    path = Path("raw") / state_code.lower()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _layout_path(state_code: str) -> Path:
    return state_raw_dir(state_code) / "_layout_state.json"


def load_layout_state(state_code: str) -> Dict[str, Any]:
    path = _layout_path(state_code)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_layout_state(state_code: str, data: Dict[str, Any]) -> None:
    _layout_path(state_code).write_text(json.dumps(data, indent=2), encoding="utf-8")


def fingerprint_dataframe(df: pd.DataFrame) -> str:
    cols = [canonical_col(str(c)) for c in df.columns]
    return "|".join(cols)


def parse_pdf_tables(path: str, logger) -> List[pd.DataFrame]:
    tables: List[pd.DataFrame] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_tables() or []
            for raw_table in extracted:
                if not raw_table or len(raw_table) < 2:
                    continue
                header = [str(h).strip() if h is not None else "" for h in raw_table[0]]
                body = raw_table[1:]
                tables.append(pd.DataFrame(body, columns=header))

    if tables:
        return tables

    # Fallback parser requested by spec.
    try:
        import camelot  # type: ignore

        logger.info("pdfplumber found no tables; trying camelot fallback file=%s", path)
        camelot_tables = camelot.read_pdf(path, pages="all", flavor="stream")
        for table in camelot_tables:
            df = table.df
            if df is None or df.empty or len(df) < 2:
                continue
            header = [str(x).strip() for x in df.iloc[0].tolist()]
            body = df.iloc[1:].copy()
            body.columns = header
            tables.append(body.reset_index(drop=True))
    except Exception as exc:
        logger.warning("camelot fallback unavailable_or_failed file=%s err=%s", path, exc)

    return tables


def parse_spreadsheet(path: str) -> List[pd.DataFrame]:
    dfs: List[pd.DataFrame] = []
    if path.lower().endswith(".csv"):
        dfs.append(pd.read_csv(path))
        return dfs
    xls = pd.ExcelFile(path)
    for sheet in xls.sheet_names:
        dfs.append(pd.read_excel(xls, sheet_name=sheet))
    return dfs


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    indexed = {canonical_col(str(c)): c for c in df.columns}
    for cand in candidates:
        if cand in indexed:
            return indexed[cand]
    return None


def normalize_operator(raw_operator: Any, alias_map: Dict[str, str]) -> Optional[str]:
    if raw_operator is None:
        return None
    text = str(raw_operator).strip()
    if not text:
        return None
    for alias, canonical in alias_map.items():
        if text.lower() == alias.lower():
            return canonical
    return text.lower().replace(" ", "_")


def extract_operator_rows(
    df: pd.DataFrame,
    state_code: str,
    month: Optional[str],
    alias_map: Dict[str, str],
    source_url: str,
) -> Tuple[List[Dict[str, Any]], bool, Optional[Tuple[float, float]]]:
    operator_col = find_column(df, ["operator", "licensee", "partner", "brand"])
    handle_col = find_column(df, ["handle", "wagersaccepted", "amountwagered", "totalsportswageringhandle"])
    revenue_col = find_column(
        df,
        ["grossrevenue", "grossgamingrevenue", "sportswageringgrossrevenue", "ggr", "revenue"],
    )

    if not operator_col or not handle_col or not revenue_col:
        return [], False, None

    records: List[Dict[str, Any]] = []
    total_handle: Optional[float] = None
    total_revenue: Optional[float] = None

    for _, row in df.iterrows():
        op_raw = row.get(operator_col)
        op_str = "" if op_raw is None else str(op_raw).strip()
        if not op_str:
            continue

        handle_val = to_float(row.get(handle_col))
        revenue_val = to_float(row.get(revenue_col))

        if "total" in op_str.lower() or "statewide" in op_str.lower():
            if total_handle is None and handle_val is not None:
                total_handle = handle_val
            if total_revenue is None and revenue_val is not None:
                total_revenue = revenue_val
            continue

        operator = normalize_operator(op_str, alias_map)
        if operator is None:
            continue
        record = {
            "platform": operator,
            "platform_type": "sportsbook",
            "metric_date": month or "",
            "geography": state_code,
            "volume": handle_val,
            "volume_basis": "monthly_handle",
            "revenue": revenue_val,
            "liquidity_or_oi": None,
            "state": state_code,
            "month": month,
            "operator": operator,
            "handle": handle_val,
            "gross_revenue": revenue_val,
            "validation_status": "unvalidated",
        }
        records.append(add_provenance(record, source_url=source_url))

    summary = None
    if total_handle is not None and total_revenue is not None:
        summary = (total_handle, total_revenue)

    return records, bool(records), summary


def move_to_needs_review(path: str, reason: str, logger) -> None:
    dest_dir = Path("raw/needs_review")
    dest_dir.mkdir(parents=True, exist_ok=True)
    src = Path(path)
    stamp = now_utc_stamp()
    dest = dest_dir / f"{src.stem}_{stamp}{src.suffix}"
    shutil.copy2(src, dest)
    logger.warning("NEEDS_REVIEW file=%s copied_to=%s reason=%s", path, str(dest), reason)


def apply_reconciliation(
    records: List[Dict[str, Any]],
    summary_totals: Dict[Tuple[str, str], Tuple[float, float]],
) -> None:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for rec in records:
        key = (str(rec.get("state")), str(rec.get("month")))
        grouped.setdefault(key, []).append(rec)

    for key, recs in grouped.items():
        summary = summary_totals.get(key)
        if not summary:
            for rec in recs:
                rec["validation_status"] = "no_summary_total_found"
            continue

        expected_handle, expected_revenue = summary
        sum_handle = sum((r.get("handle") or 0.0) for r in recs)
        sum_revenue = sum((r.get("gross_revenue") or 0.0) for r in recs)

        # "within rounding" tolerance
        handle_ok = abs(sum_handle - expected_handle) <= 1.0
        revenue_ok = abs(sum_revenue - expected_revenue) <= 1.0
        status = "reconciled" if handle_ok and revenue_ok else "mismatch"
        for rec in recs:
            rec["validation_status"] = status


def collect_state_reports(
    client: PoliteClient,
    state_code: str,
    state_cfg: Dict[str, Any],
    logger,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    alias_map = state_cfg.get("operator_aliases", {})
    layout_state = load_layout_state(state_code)
    summary_totals: Dict[Tuple[str, str], Tuple[float, float]] = {}

    report_links: List[str] = []
    for index_url in state_cfg.get("report_index_urls", []):
        try:
            response = client.get(index_url, use_cache=True, robots_required=True)
        except PermissionError as exc:
            logger.warning("index_blocked_by_robots state=%s index=%s err=%s", state_code, index_url, exc)
            continue
        if response.status_code >= 400:
            logger.warning("index_fetch_failed state=%s index=%s status=%s", state_code, index_url, response.status_code)
            continue

        html = response.body.decode("utf-8", errors="ignore")
        links = discover_report_links(index_url, html)
        logger.info("state=%s index=%s links_found=%s", state_code, index_url, len(links))
        report_links.extend(links)

    for link in sorted(set(report_links)):
        parsed = urlparse(link)
        ext = Path(parsed.path).suffix.lower().lstrip(".")
        if ext not in {"pdf", "csv", "xlsx", "xls"}:
            continue

        result = client.get(link, use_cache=True, robots_required=False)
        raw_path = save_raw_bytes(
            source=f"{state_code.lower()}_{ext}",
            ext=ext,
            content=result.body,
            subdir=str(state_raw_dir(state_code)),
        )
        month = parse_month_from_text(Path(parsed.path).name) or parse_month_from_text(link)
        logger.info("downloaded state=%s file=%s month=%s", state_code, raw_path, month)

        try:
            tables = parse_pdf_tables(raw_path, logger=logger) if ext == "pdf" else parse_spreadsheet(raw_path)
        except Exception as exc:
            logger.warning("parse_failed state=%s file=%s err=%s", state_code, raw_path, exc)
            move_to_needs_review(raw_path, reason=f"parse_failed:{exc}", logger=logger)
            continue

        parsed_any = False
        for table in tables:
            if table.empty:
                continue
            extracted, matched, summary = extract_operator_rows(
                table,
                state_code=state_code,
                month=month,
                alias_map=alias_map,
                source_url=link,
            )
            if not matched:
                continue

            # Compare against last successful parse layout for this state.
            fp = fingerprint_dataframe(table)
            last_fp = layout_state.get("last_successful_fingerprint")
            if last_fp and last_fp != fp:
                move_to_needs_review(raw_path, reason="layout_fingerprint_changed", logger=logger)
                parsed_any = False
                extracted = []
                break

            parsed_any = True
            records.extend(extracted)
            layout_state["last_successful_fingerprint"] = fp
            layout_state["last_successful_source"] = link
            if month and summary is not None:
                summary_totals[(state_code, month)] = summary

        if not parsed_any:
            logger.warning("no_operator_rows_or_layout_changed state=%s file=%s", state_code, raw_path)

    apply_reconciliation(records, summary_totals)
    save_layout_state(state_code, layout_state)
    return records


def build_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[EXPECTED_COLUMNS]


def main() -> None:
    ensure_project_dirs()
    config = load_config("config.yaml")
    logger, log_path = configure_logger("collect_sportsbook_records")
    client = PoliteClient(config=config, logger=logger)
    logger.info("Run started log_path=%s", log_path)

    all_records: List[Dict[str, Any]] = []
    states_cfg = config.get("sportsbook_records", {}).get("states", {})
    for state_code, state_cfg in states_cfg.items():
        logger.info("Collecting state=%s regulator=%s", state_code, state_cfg.get("regulator"))
        all_records.extend(collect_state_reports(client, state_code, state_cfg, logger))

    df = build_dataframe(all_records)
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info("Wrote %s rows to %s", len(df), OUTPUT_PATH)


if __name__ == "__main__":
    main()
