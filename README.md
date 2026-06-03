# Sportsbooks vs. Prediction Markets Data Collection

Reproducible data collection toolkit for an academic comparison of:

1. Prediction-market transparency metrics (Kalshi + Polymarket), and
2. Sportsbook public-record metrics (state regulator handle/revenue), plus
3. Public product/UX evidence collection for manual coding.

## Project layout

```
/raw            # untouched API responses, downloaded files, cached responses
/processed      # tidy CSV outputs ready for analysis
/logs           # one log file per run
/docs           # source notes and methodology docs
config.yaml     # endpoints, operators, aliases, and rate limits
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional environment file:

```bash
cp .env.example .env   # if needed for future authenticated sources
```

## Running collectors

### 1) Prediction markets (Module A)

```bash
python3 collect_prediction_markets.py
```

Output:

- `/processed/prediction_markets.csv`
- raw JSON pages in `/raw`
- run log in `/logs`

### 2) Sportsbook records (Module B)

```bash
python3 collect_sportsbook_records.py
```

Output:

- `/processed/sportsbook_records.csv`
- downloaded reports in `/raw`
- parsing warnings and reconciliation notices in `/logs`
- format-change exceptions copied to `/raw/needs_review/`

### 3) UX evidence pack (Module C)

```bash
python3 collect_ux_features.py
```

Output:

- `/processed/ux_feature_matrix.csv` (blank/manual-coded matrix scaffold)
- HTML + text evidence in `/raw/ux_pages/{platform}/`
- source metadata sidecars (`.json`)

### 4) Merge metrics (A + B)

```bash
python3 merge.py
```

Output:

- `/processed/metrics_combined.csv`

### 5) Analysis exhibits (Module D)

```bash
python3 analyze.py
```

Output:

- `/outputs/figures/*.png` and `/outputs/figures/*.svg`
- `/outputs/tables/*.tex` and `/outputs/tables/*.xlsx`
- `/outputs/captions.md`
- validation trace logs in `/logs/analyze_*.log`

## Tests

```bash
pytest -q
```

## Provenance & reproducibility

Every output row includes:

- `source_url`
- `collected_at_utc` (ISO-8601)
- `collector_version` (git short hash)

Raw files are timestamped and append-only:

`{source}_{YYYYMMDDTHHMMSSZ}.{ext}`

Dependencies are pinned in `requirements.txt`.

## Source inventory (public only)

All source references are recorded in:

- `docs/sources.md`

For each source, include endpoint/URL, collection date, and terms/licensing notes.

## Data/ethics guardrails

- Public APIs, public regulator records, and public marketing/store pages only.
- No account logins, no paywall bypassing, no anti-bot circumvention.
- Respect documented API limits and `robots.txt` for HTML crawling.

