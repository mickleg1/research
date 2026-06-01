from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode, urlparse
from urllib.robotparser import RobotFileParser

import requests
import yaml
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


PROJECT_DIRS = ["raw", "processed", "logs", "docs", "raw/cache", "raw/needs_review", "raw/ux_pages"]


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_project_dirs(base_dir: str = ".") -> None:
    for directory in PROJECT_DIRS:
        Path(base_dir, directory).mkdir(parents=True, exist_ok=True)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def now_utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_short_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def configure_logger(module_name: str) -> Tuple[logging.Logger, str]:
    stamp = now_utc_stamp()
    log_path = Path("logs") / f"{module_name}_{stamp}.log"
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger, str(log_path)


def _query_string(params: Optional[Dict[str, Any]]) -> str:
    if not params:
        return ""
    clean = {k: v for k, v in params.items() if v is not None}
    return urlencode(clean, doseq=True)


def _cache_key(url: str, params: Optional[Dict[str, Any]]) -> str:
    q = _query_string(params)
    raw = f"{url}?{q}" if q else url
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def save_raw_bytes(source: str, ext: str, content: bytes, subdir: str = "raw") -> str:
    Path(subdir).mkdir(parents=True, exist_ok=True)
    name = f"{source}_{now_utc_stamp()}.{ext}"
    path = Path(subdir) / name
    with open(path, "wb") as handle:
        handle.write(content)
    return str(path)


def save_raw_json(source: str, payload: Any, subdir: str = "raw") -> str:
    Path(subdir).mkdir(parents=True, exist_ok=True)
    name = f"{source}_{now_utc_stamp()}.json"
    path = Path(subdir) / name
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return str(path)


def build_user_agent(config: Dict[str, Any]) -> str:
    return config.get(
        "user_agent",
        f"sportsbook-prediction-market-collector/1.0 (contact: {config.get('contact_email', 'unknown')})",
    )


@dataclass
class FetchResult:
    url: str
    status_code: int
    body: bytes
    from_cache: bool


class PoliteClient:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": build_user_agent(config)})
        self.timeout = int(config.get("request_timeout_sec", 30))
        self.min_interval = 1.0 / float(config.get("request_rate_limit_per_sec", 1.0))
        self._last_request_ts = 0.0
        self._robots_cache: Dict[str, RobotFileParser] = {}

    def _sleep_if_needed(self) -> None:
        elapsed = time.time() - self._last_request_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def _robots_parser_for(self, url: str) -> RobotFileParser:
        parsed = urlparse(url)
        host = f"{parsed.scheme}://{parsed.netloc}"
        if host in self._robots_cache:
            return self._robots_cache[host]

        rp = RobotFileParser()
        rp.set_url(f"{host}/robots.txt")
        try:
            rp.read()
        except Exception as exc:
            self.logger.warning("robots.txt read failed for %s: %s", host, exc)
        self._robots_cache[host] = rp
        return rp

    def _is_allowed_by_robots(self, url: str) -> bool:
        rp = self._robots_parser_for(url)
        return rp.can_fetch(self.session.headers.get("User-Agent", "*"), url)

    @retry(
        retry=retry_if_exception_type((requests.RequestException, requests.HTTPError)),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _request(self, url: str, params: Optional[Dict[str, Any]] = None) -> requests.Response:
        self._sleep_if_needed()
        response = self.session.get(url, params=params, timeout=self.timeout)
        self._last_request_ts = time.time()
        if response.status_code >= 500:
            response.raise_for_status()
        return response

    def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        robots_required: bool = False,
    ) -> FetchResult:
        if robots_required and not self._is_allowed_by_robots(url):
            raise PermissionError(f"Blocked by robots.txt: {url}")

        cache_key = _cache_key(url, params)
        cache_path = Path("raw/cache") / f"{cache_key}.bin"
        if use_cache and cache_path.exists():
            body = cache_path.read_bytes()
            self.logger.info("CACHE_HIT url=%s", url)
            return FetchResult(url=url, status_code=200, body=body, from_cache=True)

        response = self._request(url=url, params=params)
        body = response.content
        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(body)

        self.logger.info(
            "REQUEST url=%s status=%s bytes=%s cache=%s",
            response.url,
            response.status_code,
            len(body),
            False,
        )
        return FetchResult(url=str(response.url), status_code=response.status_code, body=body, from_cache=False)


def add_provenance(row: Dict[str, Any], source_url: str) -> Dict[str, Any]:
    row["source_url"] = source_url
    row["collected_at_utc"] = now_utc_iso()
    row["collector_version"] = git_short_hash()
    return row
