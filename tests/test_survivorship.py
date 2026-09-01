"""Tests for the deduplication and survivorship rule.

The rule is: drop exact duplicate rows, then for records sharing a business key
keep the most complete one, breaking ties on the most recent admission. These
tests pin that behaviour down so it cannot regress into "whatever row came first".
"""

import pandas as pd

from clean_data import drop_duplicates


def make_frame(rows: list[dict]) -> pd.DataFrame:
    """Build a small patient frame with the columns the rule depends on."""
    columns = ["patient_id", "first_name", "last_name", "admission_date", "email"]
    frame = pd.DataFrame(rows, columns=columns)
    frame["admission_date"] = pd.to_datetime(frame["admission_date"])
    return frame


def dedupe(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report: dict = {}
    result = drop_duplicates(frame, "patient_id", report)
    return result, report["duplicates"]


class TestExactDuplicates:
    def test_identical_rows_collapse_to_one(self):
        row = {"patient_id": "P1", "first_name": "Aisha", "last_name": "Okafor",
               "admission_date": "2025-01-01", "email": "a@example.com"}
        result, stats = dedupe(make_frame([row, dict(row), dict(row)]))

        assert len(result) == 1
        assert stats["exact_duplicate_rows_removed"] == 2
        assert stats["same_patient_rows_removed"] == 0

    def test_distinct_patients_are_all_kept(self):
        result, stats = dedupe(make_frame([
            {"patient_id": "P1", "first_name": "Aisha", "last_name": "Okafor",
             "admission_date": "2025-01-01", "email": "a@example.com"},
            {"patient_id": "P2", "first_name": "Daniel", "last_name": "Tan",
             "admission_date": "2025-01-02", "email": "d@example.com"},
        ]))

        assert len(result) == 2
        assert stats["exact_duplicate_rows_removed"] == 0
        assert stats["same_patient_rows_removed"] == 0


class TestSurvivorship:
    def test_most_complete_record_survives(self):
        """The fuller row wins even though the sparse one came first in the file."""
        result, stats = dedupe(make_frame([
            {"patient_id": "P1", "first_name": "Aisha", "last_name": None,
             "admission_date": "2025-01-01", "email": None},
            {"patient_id": "P1", "first_name": "Aisha", "last_name": "Okafor",
             "admission_date": "2025-01-01", "email": "a@example.com"},
        ]))

        assert len(result) == 1
        assert result.iloc[0]["last_name"] == "Okafor"
        assert result.iloc[0]["email"] == "a@example.com"
        assert stats["same_patient_rows_removed"] == 1

    def test_completeness_beats_recency(self):
        """A more complete older record is preferred over a sparse newer one."""
        result, _ = dedupe(make_frame([
            {"patient_id": "P1", "first_name": "Aisha", "last_name": None,
             "admission_date": "2025-06-01", "email": None},
            {"patient_id": "P1", "first_name": "Aisha", "last_name": "Okafor",
             "admission_date": "2025-01-01", "email": "a@example.com"},
        ]))

        assert result.iloc[0]["admission_date"] == pd.Timestamp("2025-01-01")
        assert result.iloc[0]["last_name"] == "Okafor"

    def test_recency_breaks_a_completeness_tie(self):
        result, _ = dedupe(make_frame([
            {"patient_id": "P1", "first_name": "Aisha", "last_name": "Okafor",
             "admission_date": "2025-01-01", "email": "old@example.com"},
            {"patient_id": "P1", "first_name": "Aisha", "last_name": "Okafor",
             "admission_date": "2025-06-01", "email": "new@example.com"},
        ]))

        assert len(result) == 1
        assert result.iloc[0]["email"] == "new@example.com"

    def test_missing_admission_date_loses_a_tie(self):
        """A record with no admission date sorts last rather than winning by accident."""
        result, _ = dedupe(make_frame([
            {"patient_id": "P1", "first_name": "Aisha", "last_name": "Okafor",
             "admission_date": None, "email": "a@example.com"},
            {"patient_id": "P1", "first_name": "Aisha", "last_name": "Okafor",
             "admission_date": "2025-06-01", "email": "a@example.com"},
        ]))

        assert len(result) == 1
        assert result.iloc[0]["admission_date"] == pd.Timestamp("2025-06-01")

    def test_survivorship_applied_per_patient_independently(self):
        result, stats = dedupe(make_frame([
            {"patient_id": "P1", "first_name": "Aisha", "last_name": None,
             "admission_date": "2025-01-01", "email": None},
            {"patient_id": "P1", "first_name": "Aisha", "last_name": "Okafor",
             "admission_date": "2025-01-01", "email": "a@example.com"},
            {"patient_id": "P2", "first_name": "Daniel", "last_name": "Tan",
             "admission_date": "2025-02-01", "email": "d@example.com"},
            {"patient_id": "P2", "first_name": "Daniel", "last_name": "Tan",
             "admission_date": "2025-03-01", "email": None},
        ]))

        assert len(result) == 2
        assert result.set_index("patient_id").loc["P1", "email"] == "a@example.com"
        assert result.set_index("patient_id").loc["P2", "email"] == "d@example.com"
        assert stats["same_patient_rows_removed"] == 2


class TestOutputShape:
    def test_result_is_sorted_by_key_with_a_clean_index(self):
        result, _ = dedupe(make_frame([
            {"patient_id": "P3", "first_name": "Mei Ling", "last_name": "Lim",
             "admission_date": "2025-01-01", "email": "m@example.com"},
            {"patient_id": "P1", "first_name": "Aisha", "last_name": "Okafor",
             "admission_date": "2025-01-01", "email": "a@example.com"},
            {"patient_id": "P2", "first_name": "Daniel", "last_name": "Tan",
             "admission_date": "2025-01-01", "email": "d@example.com"},
        ]))

        assert result["patient_id"].tolist() == ["P1", "P2", "P3"]
        assert result.index.tolist() == [0, 1, 2]

    def test_helper_column_is_not_leaked_into_the_output(self):
        result, _ = dedupe(make_frame([
            {"patient_id": "P1", "first_name": "Aisha", "last_name": "Okafor",
             "admission_date": "2025-01-01", "email": "a@example.com"},
        ]))

        assert "_completeness" not in result.columns

    def test_report_row_counts_reconcile(self):
        frame = make_frame([
            {"patient_id": "P1", "first_name": "Aisha", "last_name": "Okafor",
             "admission_date": "2025-01-01", "email": "a@example.com"},
            {"patient_id": "P1", "first_name": "Aisha", "last_name": "Okafor",
             "admission_date": "2025-01-01", "email": "a@example.com"},
            {"patient_id": "P1", "first_name": "Aisha", "last_name": None,
             "admission_date": "2025-01-01", "email": None},
            {"patient_id": "P2", "first_name": "Daniel", "last_name": "Tan",
             "admission_date": "2025-02-01", "email": "d@example.com"},
        ])
        result, stats = dedupe(frame)

        removed = stats["exact_duplicate_rows_removed"] + stats["same_patient_rows_removed"]
        assert stats["rows_in"] - removed == stats["rows_out"] == len(result)

    def test_works_without_an_admission_date_column(self):
        """The rule degrades to completeness-only rather than raising."""
        frame = pd.DataFrame([
            {"patient_id": "P1", "first_name": "Aisha", "last_name": None},
            {"patient_id": "P1", "first_name": "Aisha", "last_name": "Okafor"},
        ])
        report: dict = {}
        result = drop_duplicates(frame, "patient_id", report)

        assert len(result) == 1
        assert result.iloc[0]["last_name"] == "Okafor"
