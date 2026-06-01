import pandas as pd

from collect_ux_features import (
    UX_COLUMNS,
    create_blank_matrix,
    html_to_text,
    resolve_source_pages,
)


def test_html_to_text_removes_script_and_tags() -> None:
    html = "<html><head><script>var x = 1;</script></head><body><h1>Title</h1><p>A &amp; B</p></body></html>"
    out = html_to_text(html)
    assert "var x" not in out
    assert "Title" in out
    assert "A & B" in out


def test_resolve_source_pages_prefers_explicit_map() -> None:
    ux_cfg = {
        "source_pages": {
            "kalshi": ["https://kalshi.com", "  https://kalshi.com/markets  "],
            "polymarket": ["https://polymarket.com"],
        },
        "kalshi_urls": ["https://should-not-be-used.com"],
    }
    out = resolve_source_pages(ux_cfg)
    assert out["kalshi"] == ["https://kalshi.com", "https://kalshi.com/markets"]
    assert out["polymarket"] == ["https://polymarket.com"]


def test_resolve_source_pages_supports_legacy_suffix_keys() -> None:
    ux_cfg = {
        "draftkings_urls": ["https://draftkings.com"],
        "fanduel_urls": ["https://fanduel.com"],
    }
    out = resolve_source_pages(ux_cfg)
    assert out == {"draftkings": ["https://draftkings.com"], "fanduel": ["https://fanduel.com"]}


def test_create_blank_matrix_shape_and_columns() -> None:
    df = create_blank_matrix(["kalshi", "polymarket"])
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == UX_COLUMNS
    assert len(df) == 2
    assert set(df["platform"].tolist()) == {"kalshi", "polymarket"}
