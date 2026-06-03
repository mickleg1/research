from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

from collector_common import configure_logger, ensure_project_dirs

METRICS_PATH = Path("processed/metrics_combined.csv")
UX_MATRIX_PATH = Path("processed/ux_feature_matrix.csv")
UX_EVIDENCE_INDEX_PATH = Path("processed/ux_evidence_index.csv")

OUTPUT_ROOT = Path("outputs")
FIGURES_DIR = OUTPUT_ROOT / "figures"
TABLES_DIR = OUTPUT_ROOT / "tables"
CAPTIONS_PATH = OUTPUT_ROOT / "captions.md"

PROVENANCE_COLUMNS = ["source_url", "collected_at_utc", "collector_version"]


@dataclass
class ExhibitResult:
    exhibit_id: str
    title: str
    outputs: List[Path]
    caption: str


def ensure_analysis_dirs() -> None:
    ensure_project_dirs()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def require_columns(df: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"{label} missing required columns: {missing}")


def load_processed_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Required file not found: {METRICS_PATH}")
    if not UX_MATRIX_PATH.exists():
        raise FileNotFoundError(f"Required file not found: {UX_MATRIX_PATH}")

    metrics_df = pd.read_csv(METRICS_PATH)
    ux_df = pd.read_csv(UX_MATRIX_PATH)

    if UX_EVIDENCE_INDEX_PATH.exists():
        ux_evidence_df = pd.read_csv(UX_EVIDENCE_INDEX_PATH)
    else:
        ux_evidence_df = pd.DataFrame(columns=PROVENANCE_COLUMNS)

    return metrics_df, ux_df, ux_evidence_df


def prepare_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    out = metrics_df.copy()
    require_columns(
        out,
        [
            "platform",
            "platform_type",
            "metric_date",
            "volume",
            "volume_basis",
            "source_url",
            "collected_at_utc",
            "collector_version",
        ],
        "metrics_combined",
    )

    out["metric_date"] = pd.to_datetime(out["metric_date"], errors="coerce", utc=True)
    out["metric_month"] = out["metric_date"].dt.tz_convert(None).dt.to_period("M").dt.to_timestamp()

    if "month" in out.columns:
        month_series = out["month"].astype(str)
        parsed_month = pd.to_datetime(month_series, errors="coerce", format="%Y-%m")
        out.loc[out["metric_month"].isna(), "metric_month"] = parsed_month

    for col in ["volume", "revenue", "liquidity_or_oi", "handle", "gross_revenue"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "operator" in out.columns:
        out["operator"] = out["operator"].astype(str).replace({"nan": None})

    return out


def unique_non_null(values: Iterable[object]) -> List[str]:
    series = pd.Series(list(values), dtype="object").dropna().astype(str)
    series = series[series.str.strip() != ""]
    return sorted(series.unique().tolist())


def assert_single_volume_basis(df: pd.DataFrame, context: str) -> str:
    bases = unique_non_null(df.get("volume_basis", pd.Series(dtype="object")))
    if len(bases) != 1:
        raise AssertionError(
            f"{context} requires exactly one volume_basis before aggregation, found: {bases}"
        )
    return bases[0]


def assert_grouping_respects_volume_basis(df: pd.DataFrame, group_cols: Sequence[str], context: str) -> None:
    bases = unique_non_null(df.get("volume_basis", pd.Series(dtype="object")))
    if len(bases) > 1 and "volume_basis" not in group_cols:
        raise AssertionError(
            f"{context} attempted to combine mismatched volume_basis values {bases} "
            "without grouping by volume_basis"
        )


def assert_operator_normalized(operators: pd.Series) -> None:
    bad = []
    for raw in operators.dropna().astype(str):
        text = raw.strip()
        if not text:
            continue
        if not re.fullmatch(r"[a-z0-9_]+", text):
            bad.append(text)
    if bad:
        sample = sorted(set(bad))[:10]
        raise AssertionError(
            "Operator aliases appear un-normalized; expected lowercase snake_case values. "
            f"Examples: {sample}"
        )


def assert_provenance_not_null(df: pd.DataFrame, label: str) -> None:
    require_columns(df, PROVENANCE_COLUMNS, label)
    for col in PROVENANCE_COLUMNS:
        missing_mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
        if missing_mask.any():
            raise AssertionError(f"{label} has null/blank provenance in column {col}")


def _format_collection_window(df: pd.DataFrame) -> str:
    if "collected_at_utc" not in df.columns:
        return "unknown"
    stamps = pd.to_datetime(df["collected_at_utc"], errors="coerce", utc=True).dropna()
    if stamps.empty:
        return "unknown"
    return f"{stamps.min().isoformat()} to {stamps.max().isoformat()}"


def _format_sources(df: pd.DataFrame) -> str:
    if "source_url" not in df.columns:
        return "unknown"
    sources = unique_non_null(df["source_url"].tolist())
    if not sources:
        return "unknown"
    if len(sources) <= 6:
        return "; ".join(sources)
    preview = "; ".join(sources[:6])
    return f"{preview}; ... (+{len(sources) - 6} more)"


def build_caption(
    title: str,
    what_it_shows: str,
    source_df: pd.DataFrame,
    volume_basis_text: str,
    output_paths: Sequence[Path],
) -> str:
    output_text = ", ".join(str(path) for path in output_paths)
    return (
        f"### {title}\n"
        f"- What it shows: {what_it_shows}\n"
        f"- Data sources: {_format_sources(source_df)}\n"
        f"- Collection window (UTC): {_format_collection_window(source_df)}\n"
        f"- volume_basis: {volume_basis_text}\n"
        f"- Outputs: {output_text}\n"
    )


def save_figure(fig: plt.Figure, stem: str) -> List[Path]:
    png_path = FIGURES_DIR / f"{stem}.png"
    svg_path = FIGURES_DIR / f"{stem}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, svg_path]


def write_table_dual(df: pd.DataFrame, stem: str) -> List[Path]:
    tex_path = TABLES_DIR / f"{stem}.tex"
    xlsx_path = TABLES_DIR / f"{stem}.xlsx"
    df.to_latex(tex_path, index=False, escape=False)
    df.to_excel(xlsx_path, index=False)
    return [tex_path, xlsx_path]


def fig_volume_vs_handle(metrics_df: pd.DataFrame) -> ExhibitResult:
    pm = metrics_df[
        (metrics_df["platform_type"] == "prediction_market")
        & metrics_df["volume"].notna()
        & metrics_df["metric_month"].notna()
    ].copy()
    sb = metrics_df[
        (metrics_df["platform_type"] == "sportsbook")
        & metrics_df["volume"].notna()
        & metrics_df["metric_month"].notna()
    ].copy()

    if pm.empty or sb.empty:
        raise ValueError("Figure 1 requires both prediction market and sportsbook rows with month+volume")

    assert_grouping_respects_volume_basis(pm, ["metric_month", "volume_basis"], "Figure 1 PM monthly sum")
    assert_grouping_respects_volume_basis(sb, ["metric_month", "volume_basis"], "Figure 1 sportsbook monthly sum")

    pm_monthly = (
        pm.groupby(["metric_month", "volume_basis"], as_index=False)["volume"].sum(min_count=1)
        .sort_values("metric_month")
    )
    sb_monthly = (
        sb.groupby(["metric_month", "volume_basis"], as_index=False)["volume"].sum(min_count=1)
        .sort_values("metric_month")
    )

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 7), sharex=True)

    for basis, group in pm_monthly.groupby("volume_basis"):
        axes[0].plot(group["metric_month"], group["volume"], marker="o", linewidth=1.6, label=str(basis))
    axes[0].set_title("Prediction-market volume by month")
    axes[0].set_ylabel("Volume")
    axes[0].legend(title="volume_basis")
    axes[0].grid(alpha=0.25)

    for basis, group in sb_monthly.groupby("volume_basis"):
        axes[1].plot(group["metric_month"], group["volume"], marker="o", linewidth=1.6, label=str(basis))
    axes[1].set_title("Sportsbook handle by month")
    axes[1].set_ylabel("Handle")
    axes[1].set_xlabel("Month")
    axes[1].legend(title="volume_basis")
    axes[1].grid(alpha=0.25)

    outputs = save_figure(fig, "fig_volume_vs_handle")

    pm_bases = unique_non_null(pm_monthly["volume_basis"].tolist())
    sb_bases = unique_non_null(sb_monthly["volume_basis"].tolist())
    volume_basis_text = (
        f"Prediction-market panel bases: {pm_bases}; "
        f"sportsbook panel bases: {sb_bases}. "
        "Series are intentionally separated to avoid mixing incompatible units."
    )
    caption = build_caption(
        title="Figure 1 — Volume vs. handle over time",
        what_it_shows=(
            "Monthly prediction-market volume and monthly sportsbook handle in separate panels "
            "to preserve unit compatibility."
        ),
        source_df=pd.concat([pm, sb], ignore_index=True),
        volume_basis_text=volume_basis_text,
        output_paths=outputs,
    )
    return ExhibitResult("fig_volume_vs_handle", "Figure 1", outputs, caption)


def fig_data_granularity(metrics_df: pd.DataFrame) -> ExhibitResult:
    granularity = (
        metrics_df[metrics_df["metric_month"].notna()]
        .groupby(["metric_month", "platform_type"], as_index=False)
        .size()
        .rename(columns={"size": "points"})
        .sort_values("metric_month")
    )
    if granularity.empty:
        raise ValueError("Figure 2 requires month-resolvable rows in metrics_combined.csv")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for platform_type, group in granularity.groupby("platform_type"):
        ax.plot(group["metric_month"], group["points"], marker="o", linewidth=1.6, label=str(platform_type))

    ax.set_title("Data granularity contrast: observations available per month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Data points available")
    ax.legend(title="platform_type")
    ax.grid(alpha=0.25)

    outputs = save_figure(fig, "fig_data_granularity")
    all_bases = unique_non_null(metrics_df["volume_basis"].tolist())
    caption = build_caption(
        title="Figure 2 — Transparency/granularity contrast",
        what_it_shows=(
            "Count of processed observations available each month by platform type. "
            "This is a data-availability exhibit, not a betting-volume comparison."
        ),
        source_df=metrics_df,
        volume_basis_text=f"Not a volume aggregate; underlying dataset includes bases: {all_bases}",
        output_paths=outputs,
    )
    return ExhibitResult("fig_data_granularity", "Figure 2", outputs, caption)


def fig_operator_concentration(metrics_df: pd.DataFrame, top_n: int = 5) -> ExhibitResult:
    require_columns(metrics_df, ["operator", "volume", "metric_month", "volume_basis"], "metrics_combined")
    sb = metrics_df[
        (metrics_df["platform_type"] == "sportsbook")
        & metrics_df["operator"].notna()
        & metrics_df["metric_month"].notna()
        & metrics_df["volume"].notna()
    ].copy()
    if sb.empty:
        raise ValueError("Figure 3 requires sportsbook operator rows with month and volume")

    assert_operator_normalized(sb["operator"])
    basis = assert_single_volume_basis(sb, "Figure 3 operator concentration")

    grouped = (
        sb.groupby(["metric_month", "operator"], as_index=False)["volume"].sum(min_count=1)
        .sort_values(["metric_month", "operator"])
    )
    totals = grouped.groupby("metric_month", as_index=False)["volume"].sum(min_count=1).rename(
        columns={"volume": "month_total"}
    )
    grouped = grouped.merge(totals, on="metric_month", how="left")
    grouped["share"] = grouped["volume"] / grouped["month_total"]

    top_operators = (
        grouped.groupby("operator", as_index=False)["volume"].sum(min_count=1)
        .sort_values("volume", ascending=False)
        .head(top_n)["operator"]
        .tolist()
    )

    grouped["operator_plot"] = grouped["operator"].where(grouped["operator"].isin(top_operators), "other")
    plot_df = (
        grouped.groupby(["metric_month", "operator_plot"], as_index=False)["share"].sum(min_count=1)
        .pivot(index="metric_month", columns="operator_plot", values="share")
        .fillna(0.0)
    )
    plot_df = plot_df.loc[:, plot_df.sum().sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    x_values = plot_df.index.to_pydatetime()
    y_values = [plot_df[col].to_numpy() for col in plot_df.columns]
    ax.stackplot(x_values, y_values, labels=list(plot_df.columns), alpha=0.9)
    ax.set_title("Sportsbook operator concentration over time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Share of monthly handle")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(loc="upper left", ncol=2)
    ax.grid(alpha=0.2)

    outputs = save_figure(fig, "fig_operator_concentration")
    caption = build_caption(
        title="Figure 3 — Operator concentration (consumer layer)",
        what_it_shows=(
            "Monthly handle share by sportsbook operator (top operators plus 'other'). "
            "Alias-normalization is enforced before plotting."
        ),
        source_df=sb,
        volume_basis_text=f"Single basis enforced before share calculation: {basis}",
        output_paths=outputs,
    )
    return ExhibitResult("fig_operator_concentration", "Figure 3", outputs, caption)


def table_ux_comparison(ux_df: pd.DataFrame, ux_evidence_df: pd.DataFrame, logger) -> ExhibitResult:
    require_columns(ux_df, ["platform"], "ux_feature_matrix")
    table_df = ux_df.copy()

    unresolved_cells: List[str] = []
    for row_idx, row in table_df.iterrows():
        for col in table_df.columns:
            if col == "platform":
                continue
            value = "" if pd.isna(row[col]) else str(row[col])
            if value.strip() == "" or "MANUAL" in value.upper():
                unresolved_cells.append(f"row={row_idx},platform={row['platform']},column={col}")

    if unresolved_cells:
        logger.warning(
            "UX matrix has unresolved cells count=%s sample=%s",
            len(unresolved_cells),
            unresolved_cells[:10],
        )

    outputs = write_table_dual(table_df, "table_ux_comparison")
    caption_sources = ux_evidence_df if not ux_evidence_df.empty else pd.DataFrame(columns=PROVENANCE_COLUMNS)
    caption = build_caption(
        title="Table 1 — UX feature comparison",
        what_it_shows="Platform-by-feature categorical matrix from the hand-coded UX sheet.",
        source_df=caption_sources,
        volume_basis_text="N/A (categorical UX feature matrix, no volume aggregation)",
        output_paths=outputs,
    )
    return ExhibitResult("table_ux_comparison", "Table 1", outputs, caption)


def table_summary_metrics(metrics_df: pd.DataFrame) -> ExhibitResult:
    require_columns(
        metrics_df,
        ["platform", "platform_type", "metric_date", "volume", "volume_basis", "revenue", "liquidity_or_oi"],
        "metrics_combined",
    )

    records: List[Dict[str, object]] = []
    trace_rows: List[pd.DataFrame] = []
    grouped = metrics_df.groupby(["platform", "platform_type", "volume_basis"], dropna=False)

    for (platform, platform_type, volume_basis), group in grouped:
        valid_dates = group["metric_date"].dropna()
        if valid_dates.empty:
            continue
        latest_date = valid_dates.max()
        latest_rows = group[group["metric_date"] == latest_date]
        trace_rows.append(latest_rows)

        records.append(
            {
                "platform": platform,
                "platform_type": platform_type,
                "latest_metric_date": latest_date.date().isoformat(),
                "volume_basis": volume_basis,
                "latest_volume_or_handle": latest_rows["volume"].sum(min_count=1),
                "latest_revenue": latest_rows["revenue"].sum(min_count=1),
                "latest_liquidity_or_oi": latest_rows["liquidity_or_oi"].sum(min_count=1),
            }
        )

    if not records:
        raise ValueError("Table 2 could not be built because no rows had valid metric_date values")

    summary_df = pd.DataFrame(records).sort_values(["platform_type", "platform", "volume_basis"]).reset_index(drop=True)

    display_df = summary_df.copy()
    for numeric_col in ["latest_volume_or_handle", "latest_revenue", "latest_liquidity_or_oi"]:
        display_df[numeric_col] = display_df[numeric_col].round(2)

    outputs = write_table_dual(display_df, "table_summary_metrics")
    caption_source = pd.concat(trace_rows, ignore_index=True) if trace_rows else metrics_df.head(0)
    caption = build_caption(
        title="Table 2 — Summary metrics by platform",
        what_it_shows=(
            "Latest available period by platform and volume_basis, including volume/handle, revenue, and liquidity/OI."
        ),
        source_df=caption_source,
        volume_basis_text="Shown directly in the table (no cross-basis aggregation)",
        output_paths=outputs,
    )
    return ExhibitResult("table_summary_metrics", "Table 2", outputs, caption)


def validate(metrics_df: pd.DataFrame, logger) -> None:
    logger.info("VALIDATION start")

    assert_provenance_not_null(metrics_df, "metrics_combined")
    logger.info("VALIDATION provenance columns are non-null")

    pm = metrics_df[
        (metrics_df["platform_type"] == "prediction_market")
        & metrics_df["metric_month"].notna()
        & metrics_df["volume"].notna()
    ]
    sb = metrics_df[
        (metrics_df["platform_type"] == "sportsbook")
        & metrics_df["metric_month"].notna()
        & metrics_df["volume"].notna()
    ]

    assert_grouping_respects_volume_basis(pm, ["metric_month", "volume_basis"], "validate fig1 prediction-market")
    assert_grouping_respects_volume_basis(sb, ["metric_month", "volume_basis"], "validate fig1 sportsbook")
    logger.info("VALIDATION volume_basis guards passed for planned aggregates")

    if "operator" in metrics_df.columns:
        sb_operator = metrics_df[
            (metrics_df["platform_type"] == "sportsbook")
            & metrics_df["operator"].notna()
            & metrics_df["volume"].notna()
        ]
        if not sb_operator.empty:
            assert_operator_normalized(sb_operator["operator"])
            assert_single_volume_basis(sb_operator, "validate fig3 sportsbook operator")
            logger.info("VALIDATION operator normalization and single basis checks passed")

    sample_n = min(3, len(metrics_df))
    if sample_n > 0:
        sample = metrics_df.sample(sample_n, random_state=42)
        for _, row in sample.iterrows():
            logger.info(
                "TRACE platform=%s platform_type=%s metric_date=%s volume=%s volume_basis=%s source_url=%s",
                row.get("platform"),
                row.get("platform_type"),
                row.get("metric_date"),
                row.get("volume"),
                row.get("volume_basis"),
                row.get("source_url"),
            )
    else:
        logger.warning("TRACE skipped because metrics dataframe is empty")

    needs_review_dir = Path("raw/needs_review")
    if needs_review_dir.exists():
        flagged = sorted(path for path in needs_review_dir.iterdir() if path.is_file())
        if flagged:
            logger.warning("Manual follow-up: raw/needs_review is not empty count=%s", len(flagged))
        else:
            logger.info("raw/needs_review is empty")

    logger.info("VALIDATION complete")


def write_captions(results: Sequence[ExhibitResult]) -> None:
    lines = ["# Auto-generated exhibit captions", ""]
    for result in results:
        lines.append(result.caption)
        lines.append("")
    CAPTIONS_PATH.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")



def main() -> None:
    random.seed(42)
    ensure_analysis_dirs()
    logger, log_path = configure_logger("analyze")
    logger.info("Run started log_path=%s", log_path)

    metrics_df_raw, ux_df, ux_evidence_df = load_processed_inputs()
    metrics_df = prepare_metrics(metrics_df_raw)

    validate(metrics_df, logger)

    results: List[ExhibitResult] = []
    results.append(fig_volume_vs_handle(metrics_df))
    results.append(fig_data_granularity(metrics_df))
    results.append(fig_operator_concentration(metrics_df))
    results.append(table_ux_comparison(ux_df, ux_evidence_df, logger))
    results.append(table_summary_metrics(metrics_df))

    write_captions(results)
    logger.info("Wrote captions to %s", CAPTIONS_PATH)


if __name__ == "__main__":
    main()
