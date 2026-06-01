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
REPORT_LINK_PATTERN = re.compile(r'href=[\'"]([^\'"]+\.(?:pdf|csv|xlsx|xls))[\'"]', re.IGNORECASE)
MONTH_PATTERN = re.compile(r"(20\d{2})[-_ ]?(0[1-9]|1[0-2])")

EXPECTED_COLUMNS = [
    "platform",
    "platform_type",
    "metric_date",
    "geography",
    "volume",
    "volume_basis",
    "revenue",
    "liquidity_or_oi",
    "state",
    "month",
    "operator",
    "handle",
    "gross_revenue",
    "validation_status",
    "source_url",
    "collected_at_utc",
    "collector_version",
]


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
    for href in REPORT_LINK_PATTERN.findall(html):
        links.append(urljoin(base_url, href))
    return sorted(set(links))


def state_raw_dir(state_code: str) -> Path:
    path = Path("raw") / state_code.lower()
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_layout_fingerprints(state_code: str) -> Dict[str, str]:
    path = state_raw_dir(state_code) / "_layout_fingerprints.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_layout_fingerprints(state_code: str, data: Dict[str, str]) -> None:
    path = state_raw_dir(state_code) / "_layout_fingerprints.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def fingerprint_dataframe(df: pd.DataFrame) -> str:
    cols = [canonical_col(str(c)) for c in df.columns]
    return "|".join(cols)


def parse_pdf_tables(path: str) -> List[pd.DataFrame]:
    tables: List[pd.DataFrame] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_tables() or []
            for raw_table in extracted:
                if not raw_table or len(raw_table) < 2:
                    continue
                header = [str(h).strip() if h is not None else "" for h in raw_table[0]]
                body = raw_table[1:]
                df = pd.DataFrame(body, columns=header)
                tables.append(df)
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
) -> Tuple[List[Dict[str, Any]], bool]:
    operator_col = find_column(df, ["operator", "licensee", "partner", "brand"])
    handle_col = find_column(df, ["handle", "wagersaccepted", "amountwagered", "totalsportswageringhandle"])
    revenue_col = find_column(
        df,
        [
            "grossrevenue",
            "grossgamingrevenue",
            "sportswageringgrossrevenue",
            "ggr",
            "revenue",
        ],
    )

    if not operator_col or not handle_col or not revenue_col:
        return [], False

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
        if "total" in op_str.lower():
            total_handle = handle_val if total_handle is None else total_handle
            total_revenue = revenue_val if total_revenue is None else total_revenue
            continue

        operator = normalize_operator(op_str, alias_map)
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

    if records and total_handle is not None and total_revenue is not None:
        sum_handle = sum((r["handle"] or 0.0) for r in records)
        sum_revenue = sum((r["gross_revenue"] or 0.0) for r in records)
        handle_ok = abs(sum_handle - total_handle) <= 1.0
        revenue_ok = abs(sum_revenue - total_revenue) <= 1.0
        status = "reconciled" if handle_ok and revenue_ok else "mismatch"
        for rec in records:
            rec["validation_status"] = status
    elif records:
        for rec in records:
            rec["validation_status"] = "no_summary_total_found"

    return records, True


def move_to_needs_review(path: str, reason: str, logger) -> None:
    dest_dir = Path("raw/needs_review")
    dest_dir.mkdir(parents=True, exist_ok=True)
    src = Path(path)
    stamp = now_utc_stamp()
    dest = dest_dir / f"{src.stem}_{stamp}{src.suffix}"
    shutil.copy2(src, dest)
    logger.warning("FORMAT_CHANGE file=%s copied_to=%s reason=%s", path, str(dest), reason)


def collect_state_reports(
    client: PoliteClient,
    state_code: str,
    state_cfg: Dict[str, Any],
    logger,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    alias_map = state_cfg.get("operator_aliases", {})
    fingerprints = load_layout_fingerprints(state_code)

    report_links: List[str] = []
    for index_url in state_cfg.get("report_index_urls", []):
        response = client.get(index_url, use_cache=True, robots_required=True)
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
        raw_path = save_raw_bytes(source=f"{state_code.lower()}_{ext}", ext=ext, content=result.body, subdir=str(state_raw_dir(state_code)))
        month = parse_month_from_text(Path(parsed.path).name) or parse_month_from_text(link)
        logger.info("downloaded state=%s file=%s month=%s", state_code, raw_path, month)

        try:
            tables = parse_pdf_tables(raw_path) if ext == "pdf" else parse_spreadsheet(raw_path)
        except Exception as exc:
            logger.warning("parse_failed state=%s file=%s err=%s", state_code, raw_path, exc)
            move_to_needs_review(raw_path, reason=f"parse_failed:{exc}", logger=logger)
            continue

        parsed_any = False
        for table in tables:
            if table.empty:
                continue
            fp = fingerprint_dataframe(table)
            fp_key = f"{state_code}:{ext}"
            if fp_key in fingerprints and fingerprints[fp_key] != fp:
                move_to_needs_review(raw_path, reason="layout_fingerprint_changed", logger=logger)
                parsed_any = False
                break
            fingerprints.setdefault(fp_key, fp)

            extracted, matched = extract_operator_rows(
                table,
                state_code=state_code,
                month=month,
                alias_map=alias_map,
                source_url=link,
            )
            if matched and extracted:
                parsed_any = True
                records.extend(extracted)

        if not parsed_any:
            logger.warning("no_operator_rows state=%s file=%s", state_code, raw_path)

    save_layout_fingerprints(state_code, fingerprints)
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
