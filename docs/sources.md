# Source Notes and Documentation Inventory

Updated: 2026-06-01 (UTC)

## Module A API documentation references

### Kalshi

- API intro / environments:
  - https://docs.kalshi.com/getting_started/api_environments
  - https://docs.kalshi.com/getting_started/quick_start_market_data
- Market endpoints:
  - https://docs.kalshi.com/api-reference/market/get-markets.md
  - https://docs.kalshi.com/api-reference/market/get-market

Notes:
- Public market data endpoint base URL: `https://external-api.kalshi.com/trade-api/v2`
- API docs indicate pagination via `cursor`, and no auth needed for the market-data quick-start endpoints.

### Polymarket

- API introduction:
  - https://docs.polymarket.com/api-reference/introduction
- Gamma market discovery:
  - https://docs.polymarket.com/api-reference/markets/list-markets
  - https://docs.polymarket.com/api-reference/markets/list-markets-keyset-pagination
  - https://docs.polymarket.com/market-data/fetching-markets
- Data API open-interest:
  - https://docs.polymarket.com/api-reference/misc/get-open-interest

Notes:
- Gamma base URL: `https://gamma-api.polymarket.com`
- Data API base URL: `https://data-api.polymarket.com`
- Gamma and Data API docs state public access (no auth required).
- Keyset pagination uses `after_cursor` and `next_cursor`.

## Module B public regulator sources (seed set)

- New Jersey DGE:
  - https://www.nj.gov/oag/ge/financials.html
- Pennsylvania PGCB:
  - https://gamingcontrolboard.pa.gov/?p=financial-reports
- New York State Gaming Commission:
  - https://gaming.ny.gov/gaming/sportswagering.php

## Module C public product evidence sources

- Official product/marketing pages and public app-store listing pages configured in `config.yaml`.
- Evidence collection is crawl-respectful, robots-aware, and public-only.

## Terms/licensing basis

- APIs and pages listed above are publicly accessible documentation and public website content.
- State regulator reports are public government records.
- Reuse in an academic context should still cite source URLs and access dates.
