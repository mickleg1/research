from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from collector_common import configure_logger, ensure_project_dirs


OUTPUT_PATH = "processed/metrics_combined.csv"
INPUTS = [
    "processed/prediction_markets.csv",
    "processed/sportsbook_records.csv",
]


def main() -> None:
    ensure_project_dirs()
    logger, log_path = configure_logger("merge_metrics")
    logger.info("Run started log_path=%s", log_path)

    frames: List[pd.DataFrame] = []
    for path in INPUTS:
        p = Path(path)
        if not p.exists():
            logger.warning("Skipping missing input %s", path)
            continue
        df = pd.read_csv(p)
        frames.append(df)
        logger.info("Loaded %s rows=%s", path, len(df))

    if not frames:
        logger.warning("No inputs found; writing empty combined output")
        pd.DataFrame().to_csv(OUTPUT_PATH, index=False)
        return

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(OUTPUT_PATH, index=False)
    logger.info("Wrote combined metrics rows=%s path=%s", len(combined), OUTPUT_PATH)


if __name__ == "__main__":
    main()
