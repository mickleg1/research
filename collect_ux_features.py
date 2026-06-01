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
    url: str,
    html_bytes: bytes,
) -> None:
    out_dir = Path("raw/ux_pages") / platform
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = now_utc_stamp()
    slug = slug_from_url(url)
    html_path = out_dir / f"{slug}_{stamp}.html"
    txt_path = out_dir / f"{slug}_{stamp}.txt"
    meta_path = out_dir / f"{slug}_{stamp}.json"

    html_path.write_bytes(html_bytes)
    decoded = html_bytes.decode("utf-8", errors="ignore")
    txt_path.write_text(html_to_text(decoded), encoding="utf-8")

    metadata = {
        "source_url": url,
        "collected_at_utc": now_utc_iso(),
        "collector_version": git_short_hash(),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def collect_platform_pages(
    client: PoliteClient,
    platform: str,
    urls: List[str],
    logger,
) -> None:
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

        save_evidence(platform=platform, url=url, html_bytes=result.body)
        logger.info("saved_evidence platform=%s url=%s bytes=%s", platform, url, len(result.body))


def main() -> None:
    ensure_project_dirs()
    config = load_config("config.yaml")
    logger, log_path = configure_logger("collect_ux_features")
    logger.info("Run started log_path=%s", log_path)
    client = PoliteClient(config=config, logger=logger)

    ux_cfg = config.get("ux_features", {})
    platforms = ux_cfg.get("platforms", [])
    source_pages = ux_cfg.get("source_pages", {})

    matrix_df = create_blank_matrix(platforms)
    matrix_df.to_csv(UX_MATRIX_PATH, index=False)
    logger.info("Wrote blank feature matrix to %s rows=%s", UX_MATRIX_PATH, len(matrix_df))

    for platform in platforms:
        urls = source_pages.get(platform, [])
        collect_platform_pages(client=client, platform=platform, urls=urls, logger=logger)


if __name__ == "__main__":
    main()
