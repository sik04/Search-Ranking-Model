"""
src/data_loader.py
==================
Parser and validator for the LETOR 4.0 / MQ2007 dataset.

LETOR SVMLight format (one line = one query-document pair):

    <label> qid:<qid> <fid>:<val> <fid>:<val> ... # <docid> <comment>

Fields
------
- label   : integer relevance grade (MQ2007 uses 0, 1, 2)
- qid     : integer query identifier
- fid:val : sparse feature pairs, feature IDs start at 1
- #       : separator before optional inline comment
- docid   : string document identifier (first token after #)

MQ2007 specifics (QueryLevelNorm version)
-----------------------------------------
- 46 features  (feature IDs 1–46)
- 5 folds      (Fold1 … Fold5, each with train.txt / test.txt / vali.txt)
- ~1,700 unique queries across the full dataset
- Relevance labels: {0, 1, 2}
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path


import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Expected relevance labels for MQ2007 (0 = not relevant, 1 = relevant,
#: 2 = highly relevant).
MQ2007_VALID_LABELS: frozenset[int] = frozenset({0, 1, 2})

#: MQ2007 QueryLevelNorm has exactly 46 features.
MQ2007_EXPECTED_FEATURE_COUNT: int = 46

# Compiled regex for a single feature token "fid:value"
_FEATURE_RE = re.compile(r"^(\d+):([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ParsedRecord:
    """One parsed query-document pair."""

    label: int
    qid: int
    docid: str | None
    features: dict[int, float]  # {feature_id: value}
    raw_line: str
    line_number: int


@dataclass
class ParseResult:
    """Aggregated outcome of parsing one LETOR file."""

    records: list[ParsedRecord] = field(default_factory=list)
    malformed_lines: list[tuple[int, str, str]] = field(
        default_factory=list
    )  # (line_no, raw, reason)


# ---------------------------------------------------------------------------
# Low-level line parser
# ---------------------------------------------------------------------------


def _parse_line(raw: str, line_no: int) -> ParsedRecord | None:
    """Parse a single LETOR SVMLight line.

    Returns a :class:`ParsedRecord` on success, or ``None`` if the line
    should be silently skipped (blank / comment-only).  Raises
    ``ValueError`` for malformed records that should be captured by the
    caller.

    Parameters
    ----------
    raw:
        The raw string from the file (may contain a trailing newline).
    line_no:
        1-based line number, used for error messages.

    Raises
    ------
    ValueError
        When the line is non-empty and non-comment but cannot be parsed.
    """
    line = raw.strip()

    # Skip blank lines and pure comment lines
    if not line or line.startswith("#"):
        return None

    # Split off inline comment: everything after the first " # "
    comment_part = ""
    if " # " in line:
        line, comment_part = line.split(" # ", 1)
        line = line.strip()

    tokens = line.split()
    if len(tokens) < 2:
        raise ValueError("Too few tokens (need at least label and qid)")

    # --- relevance label ---
    try:
        label = int(tokens[0])
    except ValueError:
        raise ValueError(f"Non-integer label: {tokens[0]!r}")

    # --- qid ---
    qid_token = tokens[1]
    if not qid_token.startswith("qid:"):
        raise ValueError(f"Expected 'qid:<N>', got: {qid_token!r}")
    try:
        qid = int(qid_token[4:])
    except ValueError:
        raise ValueError(f"Non-integer qid value: {qid_token!r}")

    # --- features ---
    features: dict[int, float] = {}
    for token in tokens[2:]:
        m = _FEATURE_RE.match(token)
        if m is None:
            raise ValueError(f"Malformed feature token: {token!r}")
        fid = int(m.group(1))
        fval = float(m.group(2))
        if fid in features:
            raise ValueError(f"Duplicate feature ID {fid}")
        features[fid] = fval

    # --- docid (first token of the comment) ---
    docid: str | None = None
    if comment_part:
        docid_candidate = comment_part.split()[0] if comment_part.split() else None
        docid = docid_candidate

    return ParsedRecord(
        label=label,
        qid=qid,
        docid=docid,
        features=features,
        raw_line=raw.rstrip("\n"),
        line_number=line_no,
    )


# ---------------------------------------------------------------------------
# File parser
# ---------------------------------------------------------------------------


def parse_letor_file(path: str | Path) -> ParseResult:
    """Parse a LETOR-format file and return all records and malformed lines.

    The raw files are **never modified**.  Suspicious records are collected
    into ``ParseResult.malformed_lines`` and reported to the caller; they
    are NOT silently discarded.

    Parameters
    ----------
    path:
        Absolute or relative path to a LETOR .txt file.

    Returns
    -------
    ParseResult
        Contains a list of successfully parsed records and a list of
        (line_no, raw_line, reason) tuples for malformed lines.
    """
    path = Path(path)
    result = ParseResult()

    with path.open(encoding="utf-8", errors="replace") as fh:
        for line_no, raw in enumerate(fh, start=1):
            try:
                record = _parse_line(raw, line_no)
                if record is not None:
                    result.records.append(record)
            except ValueError as exc:
                logger.warning("Line %d malformed: %s", line_no, exc)
                result.malformed_lines.append((line_no, raw.rstrip("\n"), str(exc)))

    logger.info(
        "Parsed %s → %d records, %d malformed lines",
        path.name,
        len(result.records),
        len(result.malformed_lines),
    )
    return result


def records_to_dataframe(records: list[ParsedRecord]) -> pd.DataFrame:
    """Convert a list of :class:`ParsedRecord` objects to a tidy DataFrame.

    Columns: ``label``, ``qid``, ``docid``, ``feature_1``, …, ``feature_N``.
    Missing feature values for a given record are stored as ``NaN``.

    Parameters
    ----------
    records:
        List of parsed records (from :func:`parse_letor_file`).

    Returns
    -------
    pd.DataFrame
        One row per query-document pair.
    """
    if not records:
        return pd.DataFrame()

    rows = []
    for rec in records:
        row: dict = {
            "label": rec.label,
            "qid": rec.qid,
            "docid": rec.docid,
        }
        for fid, fval in rec.features.items():
            row[f"feature_{fid}"] = fval
        rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure label and qid are proper integer types (even if features have NaN)
    df["label"] = df["label"].astype(int)
    df["qid"] = df["qid"].astype(int)

    # Sort feature columns numerically
    feature_cols = sorted(
        [c for c in df.columns if c.startswith("feature_")],
        key=lambda c: int(c.split("_")[1]),
    )
    ordered_cols = ["label", "qid", "docid"] + feature_cols
    df = df[[c for c in ordered_cols if c in df.columns]]

    return df


def load_letor_file(path: str | Path) -> tuple[pd.DataFrame, list[tuple]]:
    """High-level convenience function: parse → DataFrame.

    Parameters
    ----------
    path:
        Path to a LETOR-format file.

    Returns
    -------
    df:
        Tidy DataFrame (see :func:`records_to_dataframe`).
    malformed:
        List of ``(line_no, raw_line, reason)`` for malformed lines.
    """
    result = parse_letor_file(path)
    df = records_to_dataframe(result.records)
    return df, result.malformed_lines


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Structured report produced by :func:`validate_dataframe`."""

    # Labels
    invalid_labels: pd.Series | None = None  # rows with unexpected labels
    label_distribution: dict | None = None

    # QIDs
    missing_qid_rows: int = 0

    # Features
    feature_count: int = 0
    missing_feature_mask: pd.DataFrame | None = None  # True where NaN
    missing_feature_summary: dict | None = None  # {col: count}
    unexpected_feature_ids: list[int] = field(default_factory=list)

    # Records
    malformed_count: int = 0
    duplicate_qid_docid: pd.DataFrame | None = None  # duplicated pairs
    empty_record_count: int = 0

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = ["=== Validation Report ==="]

        # Labels
        lines.append(f"\nLabel distribution: {self.label_distribution}")
        inv = 0 if self.invalid_labels is None else len(self.invalid_labels)
        lines.append(f"Invalid labels: {inv} rows")

        # QIDs
        lines.append(f"Missing qid rows: {self.missing_qid_rows}")

        # Features
        lines.append(f"Feature count in data: {self.feature_count}")
        lines.append(
            f"Expected feature count (MQ2007): {MQ2007_EXPECTED_FEATURE_COUNT}"
        )
        lines.append(f"Unexpected feature IDs: {self.unexpected_feature_ids}")
        if self.missing_feature_summary:
            total_missing = sum(self.missing_feature_summary.values())
            lines.append(f"Total missing feature values: {total_missing}")

        # Records
        lines.append(f"Malformed lines (parse phase): {self.malformed_count}")
        dup_count = (
            0
            if self.duplicate_qid_docid is None
            else len(self.duplicate_qid_docid)
        )
        lines.append(f"Duplicate (qid, docid) pairs: {dup_count}")
        lines.append(f"Empty feature rows: {self.empty_record_count}")

        return "\n".join(lines)


def validate_dataframe(
    df: pd.DataFrame,
    malformed_lines: list[tuple] | None = None,
    expected_labels: frozenset[int] = MQ2007_VALID_LABELS,
    expected_feature_count: int = MQ2007_EXPECTED_FEATURE_COUNT,
) -> ValidationReport:
    """Run all validation checks on a parsed LETOR DataFrame.

    This function **never modifies** the DataFrame.  It only inspects and
    reports.  Suspicious records are surfaced in the returned report; the
    caller decides how to act on them.

    Parameters
    ----------
    df:
        DataFrame produced by :func:`load_letor_file`.
    malformed_lines:
        List of malformed lines from the parse phase (may be empty).
    expected_labels:
        Set of valid relevance labels (default: {0, 1, 2} for MQ2007).
    expected_feature_count:
        Expected number of feature columns (default: 46 for MQ2007).

    Returns
    -------
    ValidationReport
    """
    report = ValidationReport()
    report.malformed_count = len(malformed_lines) if malformed_lines else 0

    if df.empty:
        logger.warning("DataFrame is empty — nothing to validate.")
        return report

    # ------------------------------------------------------------------ labels
    report.label_distribution = df["label"].value_counts().sort_index().to_dict()
    invalid_mask = ~df["label"].isin(expected_labels)
    if invalid_mask.any():
        report.invalid_labels = df[invalid_mask]
        logger.warning("%d rows have unexpected labels.", invalid_mask.sum())

    # ------------------------------------------------------------------- qids
    missing_qid = df["qid"].isna() | (df["qid"] == 0)
    report.missing_qid_rows = int(missing_qid.sum())
    if report.missing_qid_rows:
        logger.warning("%d rows have missing/zero qid.", report.missing_qid_rows)

    # ---------------------------------------------------------------- features
    feature_cols = sorted(
        [c for c in df.columns if c.startswith("feature_")],
        key=lambda c: int(c.split("_")[1]),
    )
    report.feature_count = len(feature_cols)

    # Check for unexpected feature IDs
    actual_ids = {int(c.split("_")[1]) for c in feature_cols}
    expected_ids = set(range(1, expected_feature_count + 1))
    report.unexpected_feature_ids = sorted(actual_ids - expected_ids)
    missing_ids = sorted(expected_ids - actual_ids)
    if missing_ids:
        logger.warning(
            "%d expected feature IDs not found in data: %s",
            len(missing_ids),
            missing_ids[:10],
        )
    if report.unexpected_feature_ids:
        logger.warning(
            "Unexpected feature IDs present: %s", report.unexpected_feature_ids[:10]
        )

    # Missing values inside feature columns
    if feature_cols:
        missing_mask = df[feature_cols].isna()
        if missing_mask.any().any():
            report.missing_feature_mask = missing_mask
            report.missing_feature_summary = (
                missing_mask.sum()[missing_mask.sum() > 0].to_dict()
            )
            logger.warning(
                "Missing feature values found in %d columns.",
                len(report.missing_feature_summary),
            )

    # Empty feature rows (all features NaN for a row)
    if feature_cols:
        all_nan_rows = df[feature_cols].isna().all(axis=1)
        report.empty_record_count = int(all_nan_rows.sum())
        if report.empty_record_count:
            logger.warning(
                "%d rows have all-NaN features.", report.empty_record_count
            )

    # --------------------------------------------------------------- duplicates
    if "docid" in df.columns and df["docid"].notna().any():
        dup_mask = df.duplicated(subset=["qid", "docid"], keep=False)
        if dup_mask.any():
            report.duplicate_qid_docid = df[dup_mask].copy()
            logger.warning(
                "%d rows are duplicate (qid, docid) pairs.",
                dup_mask.sum(),
            )

    return report
