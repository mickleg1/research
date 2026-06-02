from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
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
    r"<a[^>]+href=[\"']([^\"']+\.(?:pdf|csv|xlsx|xls))[\"'][^>]*>(.*?)</a>",
    re.IGNORECASE | re.DOTALL,
)
MONTH_PATTERN = re.compile(r"(20\d{2})[-_/ ]?(0[1-9]|1[0-2])")
MONTH_COMPACT_PATTERN = re.compile(r"(20\d{2})(0[1-9]|1[0-2])")
MONTH_NAME_PATTERN = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)[-_ ]*(20\d{2})\b",
    re.IGNORECASE,
)
MONTH_NAME_REVERSE_PATTERN = re.compile(
    r"\b(20\d{2})[-_ ]*(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)
REPORT_KEYWORDS = ("sports", "wager", "bet", "revenue", "handle", "gaming win", "ggr")

MONTH_NAME_TO_NUM = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
}

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
    if pd.isna(value):
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
    if not text:
        return None

    match = MONTH_PATTERN.search(text)
    if match:
        return f"{match.group(1)}-{match.group(2)}"

    compact = MONTH_COMPACT_PATTERN.search(text)
    if compact:
        return f"{compact.group(1)}-{compact.group(2)}"

    month_name = MONTH_NAME_PATTERN.search(text)
    if month_name:
        month_num = MONTH_NAME_TO_NUM[month_name.group(1).lower()]
        return f"{month_name.group(2)}-{month_num}"

    reverse_name = MONTH_NAME_REVERSE_PATTERN.search(text)
    if reverse_name:
        month_num = MONTH_NAME_TO_NUM[reverse_name.group(2).lower()]
        return f"{reverse_name.group(1)}-{month_num}"

    return None


def discover_report_links(
    base_url: str,
    html: str,
    include_keywords: Optional[List[str]] = None,
    exclude_keywords: Optional[List[str]] = None,
) -> List[str]:
    links: List[str] = []
    include = [kw.lower() for kw in (include_keywords or []) if str(kw).strip()]
    exclude = [kw.lower() for kw in (exclude_keywords or []) if str(kw).strip()]

    for href, anchor_text in REPORT_LINK_PATTERN.findall(html):
        full_url = urljoin(base_url, href)
        haystack = f"{full_url} {anchor_text}".lower()

        if include:
            if not any(keyword in haystack for keyword in include):
                continue
        else:
            if not any(keyword in haystack for keyword in REPORT_KEYWORDS):
                continue

        if exclude and any(keyword in haystack for keyword in exclude):
            continue

        links.append(full_url)

    return sorted(set(links))


def _link_priority(url: str) -> Tuple[int, str]:
    lower = url.lower()
    # Prefer machine-readable files first.
    if lower.endswith(".csv"):
        return (0, lower)
    if lower.endswith(".xlsx"):
        return (1, lower)
    if lower.endswith(".xls"):
        return (2, lower)
    if lower.endswith(".pdf"):
        return (3, lower)
    return (4, lower)


def infer_month_from_tables(tables: List[pd.DataFrame]) -> Optional[str]:
    for table in tables:
        if table.empty:
            continue
        sample = table.head(8).astype(str)
        text_blob = " ".join(" ".join(row) for row in sample.values.tolist())
        month = parse_month_from_text(text_blob)
        if month:
            return month
    return None


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
        dfs.append(pd.read_csv(path, dtype=str))
        return dfs
    xls = pd.ExcelFile(path)
    for sheet in xls.sheet_names:
        dfs.append(pd.read_excel(xls, sheet_name=sheet, dtype=str))
    return dfs


def parse_ny_statewide_excel(path: str, source_url: str, logger) -> Tuple[List[Dict[str, Any]], List[str]]:
    xls = pd.ExcelFile(path)
    rows: List[Dict[str, Any]] = []
    empty_sheets: List[str] = []

    for sheet in xls.sheet_names:
        raw = pd.read_excel(xls, sheet_name=sheet, header=None, dtype=object)
        sheet_rows = 0
        if len(raw) <= 13:
            empty_sheets.append(sheet)
            logger.warning("ny_sheet_no_rows file=%s sheet=%s reason=insufficient_rows", path, sheet)
            continue

        # Per provided structure: row 12 is header, row 13 starts data.
        for row_idx in range(13, len(raw)):
            month_raw = raw.iat[row_idx, 0] if raw.shape[1] > 0 else None
            handle_raw = raw.iat[row_idx, 2] if raw.shape[1] > 2 else None
            ggr_raw = raw.iat[row_idx, 3] if raw.shape[1] > 3 else None

            month_dt = pd.to_datetime(month_raw, errors="coerce")
            if pd.isna(month_dt):
                continue

            handle = to_float(handle_raw)
            gross_revenue = to_float(ggr_raw)
            if handle is None and gross_revenue is None:
                continue

            month_value = month_dt.strftime("%Y-%m")
            metric_date = month_dt.strftime("%Y-%m-%d")
            record = {
                "platform": "statewide_total",
                "platform_type": "sportsbook",
                "metric_date": metric_date,
                "geography": "NY",
                "volume": handle,
                "volume_basis": "monthly_handle",
                "revenue": gross_revenue,
                "liquidity_or_oi": None,
                "state": "NY",
                "month": month_value,
                "operator": "STATEWIDE_TOTAL",
                "handle": handle,
                "gross_revenue": gross_revenue,
                "validation_status": "statewide_total",
            }
            rows.append(add_provenance(record, source_url=source_url))
            sheet_rows += 1

        if sheet_rows == 0:
            empty_sheets.append(sheet)
            logger.warning("ny_sheet_no_rows file=%s sheet=%s reason=no_data_rows", path, sheet)

    return rows, empty_sheets



def _parse_numeric_candidates(line: str) -> List[float]:
    candidates: List[float] = []
    for token in re.findall(r"\(?\$?\s*-?\d[\d,\s]*(?:\.\d+)?\)?", line):
        raw = token.strip()
        if not raw:
            continue
        cleaned = raw.replace("$", "").replace(",", "").replace(" ", "")
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = "-" + cleaned[1:-1]
        if cleaned in {"", "-"}:
            continue
        try:
            value = float(cleaned)
        except ValueError:
            continue

        # Skip small row numbers / indices.
        has_comma = "," in raw
        if abs(value) < 1000 and not has_comma:
            continue
        candidates.append(value)
    return candidates


def _extract_nj_operator(text: str) -> Optional[str]:
    match = re.search(
        r"^\s*([A-Z0-9&.,'/ -]+?)\s*\n\s*(?:MONTHLY SPORTS WAGERING TAX RETURN|ONLINE SPORTS WAGERING SKIN DETAIL REPORT)",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if match:
        return " ".join(match.group(1).split())

    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _extract_nj_month(text: str) -> Optional[str]:
    m = re.search(r"FOR THE MONTH OF\s+([A-Z]+)\s+(20\d{2})", text, flags=re.IGNORECASE)
    if not m:
        return parse_month_from_text(text)
    month_name = m.group(1).lower()
    year = m.group(2)
    month_num = MONTH_NAME_TO_NUM.get(month_name)
    if not month_num:
        return parse_month_from_text(text)
    return f"{year}-{month_num}"


def _extract_nj_gross_revenue(text: str) -> Optional[float]:
    lines = text.splitlines()
    priorities = [
        "monthly online sports wagering gross revenue",
        "monthly online sportsbook gross revenue",
        "monthly online sports wagering gross",
    ]

    for needle in priorities:
        for line in lines:
            if needle in line.lower():
                values = _parse_numeric_candidates(line)
                if values:
                    return values[0]

    for line in lines:
        lower = line.lower()
        if "monthly" in lower and "gross revenue" in lower and "online" in lower:
            values = _parse_numeric_candidates(line)
            if values:
                return values[0]

    return None


def _extract_nj_handle(text: str) -> Optional[float]:
    handle_needles = [
        "total handle",
        "amount wagered",
        "sports wagering handle",
        "wagers accepted",
    ]
    for line in text.splitlines():
        lower = line.lower()
        if any(needle in lower for needle in handle_needles):
            values = _parse_numeric_candidates(line)
            if values:
                return values[0]
    return None


def parse_nj_monthly_pdf(path: str, source_url: str, logger) -> Tuple[Optional[Dict[str, Any]], str]:
    with pdfplumber.open(path) as pdf:
        page_texts: List[str] = []
        for page in pdf.pages:
            page_texts.append(page.extract_text() or "")

    full_text = "\n".join(page_texts)
    if not full_text.strip():
        return None, "empty_text"

    operator = _extract_nj_operator(full_text)
    month = _extract_nj_month(full_text)
    gross_revenue = _extract_nj_gross_revenue(full_text)
    handle = _extract_nj_handle(full_text)

    if not operator:
        return None, "missing_operator"
    if not month:
        return None, "missing_month"
    if gross_revenue is None:
        return None, "missing_monthly_online_gross_revenue"

    volume = handle if handle is not None else gross_revenue
    volume_basis = "monthly_handle" if handle is not None else "monthly_revenue"

    record = {
        "platform": operator,
        "platform_type": "sportsbook",
        "metric_date": f"{month}-01",
        "geography": "NJ",
        "volume": volume,
        "volume_basis": volume_basis,
        "revenue": gross_revenue,
        "liquidity_or_oi": None,
        "state": "NJ",
        "month": month,
        "operator": operator,
        "handle": handle,
        "gross_revenue": gross_revenue,
        "validation_status": "parsed_nj_text",
    }
    return add_provenance(record, source_url=source_url), "ok"


def dataframe_views(df: pd.DataFrame) -> List[pd.DataFrame]:
    views: List[pd.DataFrame] = [df]
    limit = min(5, len(df) - 1)
    for header_row in range(0, max(limit, 0)):
        candidate = df.copy()
        header = [str(x).strip() if x is not None else "" for x in candidate.iloc[header_row].tolist()]
        if sum(1 for h in header if h) < 2:
            continue
        body = candidate.iloc[header_row + 1 :].copy()
        body.columns = header
        body = body.reset_index(drop=True)
        views.append(body)
    return views


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
    operator_col = find_column(
        df,
        [
            "operator",
            "operatorname",
            "licensee",
            "partner",
            "brand",
            "sportswageringoperator",
            "sportsbook",
        ],
    )
    handle_col = find_column(
        df,
        [
            "handle",
            "wagersaccepted",
            "amountwagered",
            "totalsportswageringhandle",
            "sportswageringhandle",
            "totalhandle",
        ],
    )
    revenue_col = find_column(
        df,
        [
            "grossrevenue",
            "grossgamingrevenue",
            "sportswageringgrossrevenue",
            "ggr",
            "revenue",
            "taxablegrossrevenue",
            "totalgrossrevenue",
        ],
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

        lowered = op_str.lower()
        if "total" in lowered or "statewide" in lowered:
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

        handle_ok = abs(sum_handle - expected_handle) <= 1.0
        revenue_ok = abs(sum_revenue - expected_revenue) <= 1.0
        status = "reconciled" if handle_ok and revenue_ok else "mismatch"
        for rec in recs:
            rec["validation_status"] = status


def client_get_with_state_ua(
    client: PoliteClient,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    use_cache: bool = True,
    robots_required: bool = False,
    user_agent_override: Optional[str] = None,
):
    if not user_agent_override:
        return client.get(url, params=params, use_cache=use_cache, robots_required=robots_required)

    original_ua = client.session.headers.get("User-Agent")
    client.session.headers["User-Agent"] = user_agent_override
    try:
        return client.get(url, params=params, use_cache=use_cache, robots_required=robots_required)
    finally:
        if original_ua is None:
            client.session.headers.pop("User-Agent", None)
        else:
            client.session.headers["User-Agent"] = original_ua


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
    ua_override = state_cfg.get("user_agent_override")
    include_keywords = state_cfg.get("report_link_include_keywords", [])
    exclude_keywords = state_cfg.get("report_link_exclude_keywords", [])

    report_links: List[str] = []
    for index_url in state_cfg.get("report_index_urls", []):
        try:
            response = client_get_with_state_ua(
                client,
                index_url,
                use_cache=True,
                robots_required=True,
                user_agent_override=ua_override,
            )
        except PermissionError as exc:
            logger.warning("index_blocked_by_robots state=%s index=%s err=%s", state_code, index_url, exc)
            continue

        if response.status_code >= 400:
            logger.warning("index_fetch_failed state=%s index=%s status=%s", state_code, index_url, response.status_code)
            continue

        html = response.body.decode("utf-8", errors="ignore")
        links = discover_report_links(index_url, html, include_keywords=include_keywords, exclude_keywords=exclude_keywords)
        logger.info("state=%s index=%s links_found=%s", state_code, index_url, len(links))
        report_links.extend(links)

    report_links.extend([str(x) for x in state_cfg.get("report_urls", []) if str(x).strip()])
    ny_seen_months: Set[str] = set()

    for link in sorted(set(report_links), key=_link_priority):
        result = client_get_with_state_ua(
            client,
            link,
            use_cache=True,
            robots_required=False,
            user_agent_override=ua_override,
        )
        final_parsed = urlparse(result.url)
        ext = Path(final_parsed.path).suffix.lower().lstrip(".")
        if ext not in {"pdf", "csv", "xlsx", "xls"}:
            # Cached short-link responses can hide redirect target extension.
            link_ext = Path(urlparse(link).path).suffix.lower().lstrip(".")
            if link_ext in {"pdf", "csv", "xlsx", "xls"}:
                ext = link_ext
            else:
                result = client_get_with_state_ua(
                    client,
                    link,
                    use_cache=False,
                    robots_required=False,
                    user_agent_override=ua_override,
                )
                final_parsed = urlparse(result.url)
                ext = Path(final_parsed.path).suffix.lower().lstrip(".")

        if ext not in {"pdf", "csv", "xlsx", "xls"}:
            logger.info("skip_non_report_file state=%s requested=%s resolved=%s", state_code, link, result.url)
            continue

        raw_path = save_raw_bytes(
            source=f"{state_code.lower()}_{ext}",
            ext=ext,
            content=result.body,
            subdir=str(state_raw_dir(state_code)),
        )

        month = (
            parse_month_from_text(Path(final_parsed.path).name)
            or parse_month_from_text(result.url)
            or parse_month_from_text(link)
        )
        logger.info("downloaded state=%s file=%s month=%s source=%s resolved=%s", state_code, raw_path, month, link, result.url)

        parsed_any = False
        if state_code == "NY" and ext in {"xlsx", "xls"}:
            try:
                ny_rows, empty_sheets = parse_ny_statewide_excel(raw_path, source_url=result.url, logger=logger)
            except Exception as exc:
                logger.warning("parse_failed state=%s file=%s err=%s", state_code, raw_path, exc)
                move_to_needs_review(raw_path, reason=f"parse_failed:{exc}", logger=logger)
                continue

            deduped_rows: List[Dict[str, Any]] = []
            for row in ny_rows:
                month_key = str(row.get("month") or "")
                if not month_key:
                    continue
                if month_key in ny_seen_months:
                    continue
                ny_seen_months.add(month_key)
                deduped_rows.append(row)

            if deduped_rows:
                parsed_any = True
                records.extend(deduped_rows)
                layout_state["last_successful_fingerprint"] = "ny_statewide_header_row_12"
                layout_state["last_successful_source"] = result.url

            if empty_sheets:
                move_to_needs_review(raw_path, reason=f"sheet_no_rows:{','.join(empty_sheets[:5])}", logger=logger)

        elif state_code == "NJ" and ext == "pdf":
            try:
                nj_row, reason = parse_nj_monthly_pdf(raw_path, source_url=result.url, logger=logger)
            except Exception as exc:
                logger.warning("parse_failed state=%s file=%s err=%s", state_code, raw_path, exc)
                move_to_needs_review(raw_path, reason=f"parse_failed:{exc}", logger=logger)
                continue

            if nj_row is not None:
                parsed_any = True
                records.append(nj_row)
                layout_state["last_successful_fingerprint"] = "nj_text_monthly_online_gross_revenue"
                layout_state["last_successful_source"] = result.url
            else:
                logger.warning("nj_text_parse_failed file=%s reason=%s", raw_path, reason)

        else:
            try:
                tables = parse_pdf_tables(raw_path, logger=logger) if ext == "pdf" else parse_spreadsheet(raw_path)
            except Exception as exc:
                logger.warning("parse_failed state=%s file=%s err=%s", state_code, raw_path, exc)
                move_to_needs_review(raw_path, reason=f"parse_failed:{exc}", logger=logger)
                continue

            if month is None:
                month = infer_month_from_tables(tables)

            for table in tables:
                if table.empty:
                    continue

                for view in dataframe_views(table):
                    extracted, matched, summary = extract_operator_rows(
                        view,
                        state_code=state_code,
                        month=month,
                        alias_map=alias_map,
                        source_url=result.url,
                    )
                    if not matched:
                        continue

                    fp = fingerprint_dataframe(view)
                    last_fp = layout_state.get("last_successful_fingerprint")
                    if last_fp and last_fp != fp:
                        move_to_needs_review(raw_path, reason="layout_fingerprint_changed", logger=logger)
                        parsed_any = False
                        extracted = []
                        break

                    parsed_any = True
                    records.extend(extracted)
                    layout_state["last_successful_fingerprint"] = fp
                    layout_state["last_successful_source"] = result.url
                    if month and summary is not None:
                        summary_totals[(state_code, month)] = summary
                    break

                if parsed_any:
                    break

        if not parsed_any:
            logger.warning("no_operator_rows_or_layout_changed state=%s file=%s", state_code, raw_path)
            move_to_needs_review(raw_path, reason="no_rows_extracted", logger=logger)

    apply_reconciliation(records, summary_totals)
    save_layout_state(state_code, layout_state)
    return records


def build_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[EXPECTED_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect sportsbook records from configured states")
    parser.add_argument(
        "--states",
        default="",
        help="Comma-separated state codes to run (e.g., NY or NJ,PA,NY). Empty means all configured states.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    config = load_config("config.yaml")
    logger, log_path = configure_logger("collect_sportsbook_records")
    client = PoliteClient(config=config, logger=logger)
    logger.info("Run started log_path=%s", log_path)

    all_records: List[Dict[str, Any]] = []
    states_cfg = config.get("sportsbook_records", {}).get("states", {})
    selected_states = {s.strip().upper() for s in args.states.split(",") if s.strip()}

    for state_code, state_cfg in states_cfg.items():
        if selected_states and state_code.upper() not in selected_states:
            continue
        logger.info("Collecting state=%s regulator=%s", state_code, state_cfg.get("regulator"))
        all_records.extend(collect_state_reports(client, state_code, state_cfg, logger))

    df = build_dataframe(all_records)
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info("Wrote %s rows to %s", len(df), OUTPUT_PATH)



if __name__ == "__main__":
    main()
