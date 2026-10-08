"""
tests/test_data_loader.py
=========================
Unit tests for src/data_loader.py.

All tests use tiny in-memory fixtures — the full MQ2007 dataset is NOT
required to be present for any test to pass.
"""

from __future__ import annotations

import textwrap
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Make the src package importable when running pytest from the project root.
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import (
    MQ2007_EXPECTED_FEATURE_COUNT,
    MQ2007_VALID_LABELS,
    ParseResult,
    _parse_line,
    load_letor_file,
    parse_letor_file,
    records_to_dataframe,
    validate_dataframe,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

#: A minimal but realistic LETOR file that exercises most code paths.
VALID_LETOR_CONTENT = textwrap.dedent(
    """\
    2 qid:1 1:0.500 2:0.333 3:1.000 # GX010-70-1234567 docno=GX010-70-1234567
    1 qid:1 1:0.250 2:0.111 3:0.500 # GX010-70-7654321 docno=GX010-70-7654321
    0 qid:1 1:0.000 2:0.000 3:0.000 # GX010-70-9999999 docno=GX010-70-9999999
    2 qid:2 1:0.800 2:0.900 3:0.700 # GX020-10-1111111 docno=GX020-10-1111111
    0 qid:2 1:0.100 2:0.050 3:0.200 # GX020-10-2222222 docno=GX020-10-2222222
    """
)

#: File with a mix of good lines, a blank line, a pure comment, and one bad line.
MIXED_LETOR_CONTENT = textwrap.dedent(
    """\
    2 qid:10 1:0.5 2:0.3 # doc_A

    # This is a pure comment line — should be skipped silently
    BAD_LINE_NO_QID
    1 qid:10 1:0.2 2:0.1 # doc_B
    """
)


def _write_temp_file(content: str, suffix: str = ".txt") -> Path:
    """Write *content* to a NamedTemporaryFile and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    )
    tmp.write(content)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# 1. Valid LETOR line parsing
# ---------------------------------------------------------------------------

class TestParseValidLine:
    """_parse_line correctly handles well-formed LETOR lines."""

    def test_label_is_integer(self):
        rec = _parse_line("2 qid:1 1:0.5 2:0.3 # docA", line_no=1)
        assert rec is not None
        assert rec.label == 2

    def test_label_zero(self):
        rec = _parse_line("0 qid:5 1:0.0 # docX", line_no=1)
        assert rec.label == 0

    def test_label_max(self):
        rec = _parse_line("2 qid:5 1:1.0 # docY", line_no=1)
        assert rec.label == 2

    def test_returns_parse_record(self):
        from src.data_loader import ParsedRecord
        rec = _parse_line("1 qid:3 1:0.1 2:0.2 # docZ", line_no=42)
        assert isinstance(rec, ParsedRecord)
        assert rec.line_number == 42

    def test_line_with_scientific_notation(self):
        rec = _parse_line("0 qid:7 1:1.23e-4 2:5.67E+2 # docS", line_no=1)
        assert rec is not None
        assert abs(rec.features[1] - 1.23e-4) < 1e-10
        assert abs(rec.features[2] - 567.0) < 1e-6


# ---------------------------------------------------------------------------
# 2. QID extraction
# ---------------------------------------------------------------------------

class TestQIDExtraction:
    def test_qid_integer(self):
        rec = _parse_line("1 qid:42 1:0.5 # doc1", line_no=1)
        assert rec.qid == 42

    def test_qid_large_number(self):
        rec = _parse_line("0 qid:999999 1:0.1 # doc2", line_no=1)
        assert rec.qid == 999_999

    def test_missing_qid_prefix_raises(self):
        with pytest.raises(ValueError, match="qid"):
            _parse_line("1 42 1:0.5 # doc", line_no=1)

    def test_non_integer_qid_raises(self):
        with pytest.raises(ValueError):
            _parse_line("1 qid:abc 1:0.5 # doc", line_no=1)

    def test_qid_in_dataframe(self):
        tmp = _write_temp_file(VALID_LETOR_CONTENT)
        df, _ = load_letor_file(tmp)
        assert set(df["qid"].unique()) == {1, 2}


# ---------------------------------------------------------------------------
# 3. Relevance label extraction
# ---------------------------------------------------------------------------

class TestRelevanceLabelExtraction:
    def test_label_0(self):
        rec = _parse_line("0 qid:1 1:0.0 # doc", line_no=1)
        assert rec.label == 0

    def test_label_1(self):
        rec = _parse_line("1 qid:1 1:0.5 # doc", line_no=1)
        assert rec.label == 1

    def test_label_2(self):
        rec = _parse_line("2 qid:1 1:1.0 # doc", line_no=1)
        assert rec.label == 2

    def test_non_integer_label_raises(self):
        with pytest.raises(ValueError, match="label"):
            _parse_line("NOTANINT qid:1 1:0.5 # doc", line_no=1)

    def test_label_distribution_in_validation(self):
        tmp = _write_temp_file(VALID_LETOR_CONTENT)
        df, mal = load_letor_file(tmp)
        report = validate_dataframe(df, mal)
        dist = report.label_distribution
        assert dist[0] == 2  # two "not relevant" rows
        assert dist[1] == 1
        assert dist[2] == 2


# ---------------------------------------------------------------------------
# 4. Feature extraction
# ---------------------------------------------------------------------------

class TestFeatureExtraction:
    def test_feature_ids_and_values(self):
        rec = _parse_line("2 qid:1 1:0.5 3:0.75 10:1.0 # doc", line_no=1)
        assert rec.features == {1: 0.5, 3: 0.75, 10: 1.0}

    def test_feature_zero_value(self):
        rec = _parse_line("0 qid:2 1:0.0 2:0.0 # doc", line_no=1)
        assert rec.features[1] == pytest.approx(0.0)

    def test_features_in_dataframe(self):
        tmp = _write_temp_file(VALID_LETOR_CONTENT)
        df, _ = load_letor_file(tmp)
        assert "feature_1" in df.columns
        assert "feature_2" in df.columns
        assert "feature_3" in df.columns

    def test_feature_column_ordering(self):
        """Feature columns must be sorted numerically, not lexicographically."""
        content = "1 qid:1 1:0.1 2:0.2 10:1.0 # doc\n"
        tmp = _write_temp_file(content)
        df, _ = load_letor_file(tmp)
        feat_cols = [c for c in df.columns if c.startswith("feature_")]
        ids = [int(c.split("_")[1]) for c in feat_cols]
        assert ids == sorted(ids)

    def test_duplicate_feature_id_raises(self):
        with pytest.raises(ValueError, match="Duplicate feature ID"):
            _parse_line("1 qid:1 1:0.5 1:0.9 # doc", line_no=1)

    def test_records_to_dataframe_missing_features_are_nan(self):
        """If record A has feature_10 but record B does not, B's feature_10 is NaN."""
        content = textwrap.dedent(
            """\
            1 qid:1 1:0.5 10:0.8 # docA
            0 qid:1 1:0.3 # docB
            """
        )
        tmp = _write_temp_file(content)
        df, _ = load_letor_file(tmp)
        assert "feature_10" in df.columns
        assert pd.isna(df.loc[df["docid"] == "docB", "feature_10"].iloc[0])


# ---------------------------------------------------------------------------
# 5. Comment handling
# ---------------------------------------------------------------------------

class TestCommentHandling:
    def test_docid_extracted_from_comment(self):
        rec = _parse_line("2 qid:1 1:0.5 # GX010-70-1234567 extra info", line_no=1)
        assert rec.docid == "GX010-70-1234567"

    def test_no_comment_docid_is_none(self):
        rec = _parse_line("1 qid:1 1:0.5", line_no=1)
        assert rec.docid is None

    def test_pure_comment_line_returns_none(self):
        result = _parse_line("# this entire line is a comment", line_no=5)
        assert result is None

    def test_blank_line_returns_none(self):
        assert _parse_line("", line_no=10) is None
        assert _parse_line("   \n", line_no=11) is None

    def test_comment_not_treated_as_feature(self):
        rec = _parse_line("0 qid:2 1:0.1 # docX some note 1:fake", line_no=1)
        assert rec is not None
        # The "1:fake" inside the comment must NOT be parsed as a feature
        assert len(rec.features) == 1


# ---------------------------------------------------------------------------
# 6. Malformed line detection
# ---------------------------------------------------------------------------

class TestMalformedLineDetection:
    def test_bad_line_captured_not_raised(self):
        """parse_letor_file must capture malformed lines without crashing."""
        tmp = _write_temp_file(MIXED_LETOR_CONTENT)
        result = parse_letor_file(tmp)
        assert result.malformed_lines, "Expected at least one malformed line"

    def test_malformed_line_has_line_number(self):
        tmp = _write_temp_file(MIXED_LETOR_CONTENT)
        result = parse_letor_file(tmp)
        for line_no, raw, reason in result.malformed_lines:
            assert isinstance(line_no, int)
            assert line_no >= 1

    def test_malformed_line_has_reason(self):
        tmp = _write_temp_file(MIXED_LETOR_CONTENT)
        result = parse_letor_file(tmp)
        for _ln, _raw, reason in result.malformed_lines:
            assert isinstance(reason, str) and reason

    def test_good_lines_still_parsed_despite_bad_line(self):
        tmp = _write_temp_file(MIXED_LETOR_CONTENT)
        result = parse_letor_file(tmp)
        # MIXED_LETOR_CONTENT has 2 valid records (qid:10 lines)
        assert len(result.records) == 2

    def test_too_few_tokens_raises(self):
        with pytest.raises(ValueError):
            _parse_line("1", line_no=1)

    def test_bad_feature_token_raises(self):
        with pytest.raises(ValueError, match="Malformed feature token"):
            _parse_line("1 qid:1 BADFEAT # doc", line_no=1)


# ---------------------------------------------------------------------------
# 7. Missing feature handling
# ---------------------------------------------------------------------------

class TestMissingFeatureHandling:
    def test_missing_feature_reported_in_validation(self):
        """When records have different feature sets, NaN gaps are reported."""
        content = textwrap.dedent(
            """\
            1 qid:1 1:0.5 2:0.3 # docA
            0 qid:1 1:0.2 # docB
            """
        )
        tmp = _write_temp_file(content)
        df, mal = load_letor_file(tmp)
        report = validate_dataframe(df, mal, expected_feature_count=2)
        assert report.missing_feature_summary is not None
        assert "feature_2" in report.missing_feature_summary

    def test_missing_feature_count_correct(self):
        content = textwrap.dedent(
            """\
            1 qid:1 1:0.5 2:0.3 3:0.1 # docA
            0 qid:1 1:0.2 # docB
            0 qid:1 1:0.4 # docC
            """
        )
        tmp = _write_temp_file(content)
        df, mal = load_letor_file(tmp)
        report = validate_dataframe(df, mal, expected_feature_count=3)
        # docB and docC are each missing feature_2 and feature_3 → 4 missing cells
        total = sum(report.missing_feature_summary.values())
        assert total == 4


# ---------------------------------------------------------------------------
# 8. Duplicate detection
# ---------------------------------------------------------------------------

class TestDuplicateDetection:
    def test_no_duplicates_by_default(self):
        tmp = _write_temp_file(VALID_LETOR_CONTENT)
        df, mal = load_letor_file(tmp)
        report = validate_dataframe(df, mal)
        assert report.duplicate_qid_docid is None or len(report.duplicate_qid_docid) == 0

    def test_duplicate_pair_detected(self):
        content = textwrap.dedent(
            """\
            2 qid:1 1:0.5 # docA
            1 qid:1 1:0.5 # docA
            0 qid:1 1:0.2 # docB
            """
        )
        tmp = _write_temp_file(content)
        df, mal = load_letor_file(tmp)
        report = validate_dataframe(df, mal)
        assert report.duplicate_qid_docid is not None
        assert len(report.duplicate_qid_docid) == 2  # both rows flagged

    def test_duplicate_has_same_qid_and_docid(self):
        content = textwrap.dedent(
            """\
            2 qid:5 1:0.5 # docX
            0 qid:5 1:0.1 # docX
            1 qid:5 1:0.9 # docY
            """
        )
        tmp = _write_temp_file(content)
        df, mal = load_letor_file(tmp)
        report = validate_dataframe(df, mal)
        dups = report.duplicate_qid_docid
        assert dups is not None
        assert all(dups["qid"] == 5)
        assert all(dups["docid"] == "docX")


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_file_returns_empty_dataframe(self):
        tmp = _write_temp_file("")
        df, mal = load_letor_file(tmp)
        assert df.empty
        assert mal == []

    def test_comment_only_file_returns_empty_dataframe(self):
        content = "# just a comment\n# another\n"
        tmp = _write_temp_file(content)
        df, mal = load_letor_file(tmp)
        assert df.empty

    def test_validation_on_empty_dataframe_does_not_crash(self):
        report = validate_dataframe(pd.DataFrame(), [])
        assert report.feature_count == 0
