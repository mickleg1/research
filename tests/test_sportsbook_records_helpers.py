import pandas as pd

from collect_sportsbook_records import (
    apply_reconciliation,
    discover_report_links,
    extract_operator_rows,
    parse_month_from_text,
)


def test_parse_month_from_text() -> None:
    assert parse_month_from_text("sports_2025-09.pdf") == "2025-09"
    assert parse_month_from_text("sports report 202403 final") == "2024-03"
    assert parse_month_from_text("no-month-here") is None


def test_discover_report_links() -> None:
    html = """
    <a href="/reports/sports-jan-2025.pdf">Sports Betting Revenue</a>
    <a href="https://example.com/reports/feb-2025.xlsx">General report</a>
    """
    links = discover_report_links("https://example.com/financials", html)
    assert "https://example.com/reports/sports-jan-2025.pdf" in links
    assert "https://example.com/reports/feb-2025.xlsx" not in links


def test_extract_rows_and_reconcile() -> None:
    df = pd.DataFrame(
        {
            "Operator": ["DraftKings Sportsbook", "FanDuel Sportsbook", "Total"],
            "Handle": ["1000", "2000", "3000"],
            "Gross Revenue": ["100", "250", "350"],
        }
    )
    rows, matched, summary = extract_operator_rows(
        df=df,
        state_code="NJ",
        month="2025-01",
        alias_map={"DraftKings Sportsbook": "draftkings", "FanDuel Sportsbook": "fanduel"},
        source_url="https://example.com/report.pdf",
    )

    assert matched is True
    assert len(rows) == 2
    assert summary == (3000.0, 350.0)

    apply_reconciliation(rows, {("NJ", "2025-01"): summary})
    assert all(row["validation_status"] == "reconciled" for row in rows)
