"""
scripts/inspect_dataset.py
==========================
Loads the MQ2007 (LETOR 4.0) dataset from data/raw/ and prints a concise
factual report.

Usage
-----
    python scripts/inspect_dataset.py

If the dataset is not present in data/raw/, the script clearly reports that
and exits without fabricating any statistics.

Expected directory layout inside data/raw/
------------------------------------------
    MQ2007/
      Fold1/
        train.txt
        test.txt
        vali.txt
      Fold2/ … Fold5/

or a flat layout where .txt files sit directly under data/raw/MQ2007/.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

# Make src importable from the project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    MQ2007_EXPECTED_FEATURE_COUNT,
    MQ2007_VALID_LABELS,
    load_letor_file,
    validate_dataframe,
)

import pandas as pd

logging.basicConfig(level=logging.WARNING)

RAW_DIR = PROJECT_ROOT / "data" / "raw"

# ---------------------------------------------------------------------------
# Locate LETOR files
# ---------------------------------------------------------------------------

def find_letor_files(root: Path) -> list[Path]:
    """Recursively find all .txt files under *root*.

    Returns an empty list if none are found.
    """
    return sorted(root.rglob("*.txt"))


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def main() -> None:
    sep = "=" * 60

    print(sep)
    print("  Search Ranking Model — Dataset Inspection Report")
    print(sep)
    print(f"  Looking in: {RAW_DIR}\n")

    letor_files = find_letor_files(RAW_DIR)

    if not letor_files:
        print("  [WARNING] NO DATASET FOUND")
        print()
        print("  The MQ2007 dataset is not present in data/raw/.")
        print("  The parser and validation code have been implemented and")
        print("  tested (see tests/test_data_loader.py), but real dataset")
        print("  statistics cannot be computed until the data is available.")
        print()
        print("  HOW TO OBTAIN THE DATASET")
        print("  ─────────────────────────")
        print("  1. Visit the official Microsoft Research LETOR 4.0 page:")
        print("     https://www.microsoft.com/en-us/research/project/")
        print("     letor-learning-rank-information-retrieval/letor-4-0/")
        print()
        print("  2. Request / download the MQ2007 package.")
        print("     The recommended version is 'QueryLevelNorm'.")
        print()
        print("  3. Extract the archive so the layout is:")
        print("     data/raw/MQ2007/Fold1/train.txt")
        print("     data/raw/MQ2007/Fold1/test.txt")
        print("     data/raw/MQ2007/Fold1/vali.txt")
        print("     ... (Fold2 through Fold5)")
        print()
        print("  4. Re-run this script:")
        print("     python scripts/inspect_dataset.py")
        print()
        print(sep)
        sys.exit(0)

    # ---------------------------------------------------------------- parse all
    all_dfs: list[pd.DataFrame] = []
    all_malformed: list[tuple] = []
    files_processed: list[str] = []

    for fpath in letor_files:
        rel = fpath.relative_to(RAW_DIR)
        print(f"  Parsing: {rel} ...", end=" ", flush=True)
        df, malformed = load_letor_file(fpath)
        all_dfs.append(df)
        all_malformed.extend(malformed)
        files_processed.append(str(rel))
        print(f"{len(df):,} rows")

    print()

    combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    # ----------------------------------------------------------- run validation
    report = validate_dataframe(
        combined,
        malformed_lines=all_malformed,
        expected_labels=MQ2007_VALID_LABELS,
        expected_feature_count=MQ2007_EXPECTED_FEATURE_COUNT,
    )

    feature_cols = [c for c in combined.columns if c.startswith("feature_")]

    # ------------------------------------------------------------------ report
    print(sep)
    print("  DATASET SUMMARY")
    print(sep)

    print(f"\n  Files processed       : {len(files_processed)}")
    for fp in files_processed:
        print(f"    • {fp}")

    print(f"\n  Total rows            : {len(combined):,}")
    print(f"  Unique queries (qid)  : {combined['qid'].nunique():,}")

    if "docid" in combined.columns:
        print(f"  Unique documents      : {combined['docid'].nunique():,}")
    else:
        print("  Unique documents      : docid column not present")

    print(f"  Number of features    : {len(feature_cols)}")
    print(f"  Expected features     : {MQ2007_EXPECTED_FEATURE_COUNT}")

    print(f"\n  Label distribution:")
    if report.label_distribution:
        for lbl, cnt in sorted(report.label_distribution.items()):
            pct = 100.0 * cnt / len(combined)
            print(f"    label {lbl} : {cnt:>8,}  ({pct:.1f}%)")
    else:
        print("    (no data)")

    print(f"\n  Missing values:")
    if report.missing_feature_summary:
        for col, cnt in sorted(
            report.missing_feature_summary.items(),
            key=lambda x: -x[1],
        )[:10]:
            print(f"    {col:<20}: {cnt:,} missing")
        if len(report.missing_feature_summary) > 10:
            print(f"    ... and {len(report.missing_feature_summary) - 10} more columns")
    else:
        print("    None detected")

    print(f"\n  Duplicate (qid, docid) pairs:")
    dup_count = (
        0
        if report.duplicate_qid_docid is None
        else len(report.duplicate_qid_docid)
    )
    print(f"    {dup_count:,} rows involved in duplicates")

    print(f"\n  Malformed lines (parse errors): {report.malformed_count}")
    if report.malformed_count > 0:
        print("    First 5:")
        for ln, raw, reason in all_malformed[:5]:
            print(f"      Line {ln}: {reason!r}")
            print(f"        → {raw[:80]!r}")

    inv_count = (
        0 if report.invalid_labels is None else len(report.invalid_labels)
    )
    print(f"\n  Invalid labels (outside {{0,1,2}}): {inv_count}")

    print(f"\n  Unexpected feature IDs        : {report.unexpected_feature_ids or 'none'}")

    print(f"\n{sep}")
    print("  Inspection complete. No data was fabricated.")
    print(sep)


if __name__ == "__main__":
    main()
