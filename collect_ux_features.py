from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import pandas as pd

from collector_common import (
    PoliteClient,
    configure_logger,
    ensure_project_dirs,
    git_short_hash,
    load_config,
    now_utc_iso,
    now_utc_stamp,
)

UX_MATRIX_PATH = "processed/ux_feature_matrix.csv"
EVIDENCE_INDEX_PATH = "processed/ux_evidence_index.csv"
UX_COLUMNS = [
    "platform",
    "funding_methods",
    "market_types",
    "in_play_live",
    "social_sharing",
    "fee_structure",
    "cash_out",
    "platform_availability",
    "onboarding_kyc_flow",
    "notes",
]
EVIDENCE_INDEX_COLUMNS = [
    "platform",
    "source_url",
    "fetched_url",
    "http_status",
    "from_cache",
    "html_path",
    "text_path",
    "meta_path",
    "collected_at_utc",
    "collector_version",
]


def html_to_text(html: str) -> str:
    text = re.sub(r"<script.*?>.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;|&#160;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def slug_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/") or "root"
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", path)
    return f"{parsed.netloc}_{safe}"[:120]


def resolve_source_pages(ux_cfg: Dict[str, Any]) -> Dict[str, List[str]]:
    source_pages = ux_cfg.get("source_pages")
    if isinstance(source_pages, dict) and source_pages:
        out: Dict[str, List[str]] = {}
        for platform, urls in source_pages.items():
            if not isinstance(urls, list):
                continue
            cleaned = [str(url).strip() for url in urls if str(url).strip()]
            out[str(platform)] = cleaned
        return out

    # Backward-compatible fallback for config style: kalshi_urls, draftkings_urls, etc.
    out: Dict[str, List[str]] = {}
    for key, urls in ux_cfg.items():
        if not key.endswith("_urls") or not isinstance(urls, list):
            continue
        platform = key[: -len("_urls")]
        cleaned = [str(url).strip() for url in urls if str(url).strip()]
        out[platform] = cleaned
    return out


def create_blank_matrix(platforms: List[str]) -> pd.DataFrame:
    rows = []
    for platform in platforms:
        rows.append(
            {
                "platform": platform,
                "funding_methods": "",
                "market_types": "",
                "in_play_live": "",
                "social_sharing": "",
                "fee_structure": "",
                "cash_out": "",
                "platform_availability": "",
                "onboarding_kyc_flow": "",
                "notes": "MANUAL: complete from evidence and in-app verification where necessary.",
            }
        )
    return pd.DataFrame(rows, columns=UX_COLUMNS)


def save_evidence(
    platform: str,
    source_url: str,
    fetched_url: str,
    status_code: int,
    from_cache: bool,
    html_bytes: bytes,
) -> Dict[str, Any]:
    out_dir = Path("raw/ux_pages") / platform
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = now_utc_stamp()
    slug = slug_from_url(source_url)
    html_path = out_dir / f"{slug}_{stamp}.html"
    txt_path = out_dir / f"{slug}_{stamp}.txt"
    meta_path = out_dir / f"{slug}_{stamp}.json"

    html_path.write_bytes(html_bytes)
    decoded = html_bytes.decode("utf-8", errors="ignore")
    txt_path.write_text(html_to_text(decoded), encoding="utf-8")

    collected_at_utc = now_utc_iso()
    collector_version = git_short_hash()
    metadata = {
        "platform": platform,
        "source_url": source_url,
        "fetched_url": fetched_url,
        "http_status": status_code,
        "from_cache": from_cache,
        "html_path": str(html_path),
        "text_path": str(txt_path),
        "meta_path": str(meta_path),
        "collected_at_utc": collected_at_utc,
        "collector_version": collector_version,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def collect_platform_pages(
    client: PoliteClient,
    platform: str,
    urls: List[str],
    logger,
) -> List[Dict[str, Any]]:
    evidence_rows: List[Dict[str, Any]] = []
    for url in urls:
        try:
            # Public marketing/product pages are crawled robots-aware.
            result = client.get(url=url, use_cache=True, robots_required=True)
        except PermissionError as exc:
            logger.warning("robots_blocked platform=%s url=%s err=%s", platform, url, exc)
            continue
        except Exception as exc:
            logger.warning("fetch_failed platform=%s url=%s err=%s", platform, url, exc)
            continue

        if result.status_code >= 400:
            logger.warning(
                "fetch_non_success platform=%s url=%s status=%s",
                platform,
                url,
                result.status_code,
            )
            continue

        metadata = save_evidence(
            platform=platform,
            source_url=url,
            fetched_url=result.url,
            status_code=result.status_code,
            from_cache=result.from_cache,
            html_bytes=result.body,
        )
        evidence_rows.append(metadata)
        logger.info("saved_evidence platform=%s url=%s bytes=%s", platform, url, len(result.body))
    return evidence_rows


def main() -> None:
    ensure_project_dirs()
    config = load_config("config.yaml")
    logger, log_path = configure_logger("collect_ux_features")
    logger.info("Run started log_path=%s", log_path)
    client = PoliteClient(config=config, logger=logger)

    ux_cfg = config.get("ux_features", {})
    source_pages = resolve_source_pages(ux_cfg)
    configured_platforms = ux_cfg.get("platforms")
    if isinstance(configured_platforms, list) and configured_platforms:
        platforms = [str(p) for p in configured_platforms]
    else:
        platforms = sorted(source_pages.keys())

    matrix_df = create_blank_matrix(platforms)
    matrix_df.to_csv(UX_MATRIX_PATH, index=False)
    logger.info("Wrote blank feature matrix to %s rows=%s", UX_MATRIX_PATH, len(matrix_df))

    all_evidence: List[Dict[str, Any]] = []
    for platform in platforms:
        urls = source_pages.get(platform, [])
        if not urls:
            logger.warning("no_source_pages platform=%s", platform)
            continue
        all_evidence.extend(collect_platform_pages(client=client, platform=platform, urls=urls, logger=logger))

    evidence_df = pd.DataFrame(all_evidence)
    for col in EVIDENCE_INDEX_COLUMNS:
        if col not in evidence_df.columns:
            evidence_df[col] = None
    evidence_df = evidence_df[EVIDENCE_INDEX_COLUMNS]
    evidence_df.to_csv(EVIDENCE_INDEX_PATH, index=False)
    logger.info("Wrote evidence index to %s rows=%s", EVIDENCE_INDEX_PATH, len(evidence_df))


if __name__ == "__main__":
    main()
