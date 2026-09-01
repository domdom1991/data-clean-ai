"""Tests for fuzzy duplicate detection on name and date of birth.

The risk in a fuzzy matcher runs both ways. Too strict and it misses the
re-registrations it exists to catch; too loose and it proposes merging two
different patients. These tests pin down both edges.
"""

import pandas as pd
import pytest

from fuzzy_match import (
    DEFAULT_THRESHOLD,
    NO_DOB_THRESHOLD,
    annotate,
    dob_relation,
    find_fuzzy_duplicates,
    letter_signature,
    name_similarity,
    normalize_name,
    soundex,
)


def make_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["patient_id", "first_name", "last_name",
                                        "date_of_birth"])
    frame["date_of_birth"] = pd.to_datetime(frame["date_of_birth"])
    return frame


class TestNormalizeName:
    @pytest.mark.parametrize("raw, expected", [
        ("Okafor", "okafor"),
        ("  Okafor  ", "okafor"),
        ("O'Souza", "o souza"),
        ("Mei  Ling", "mei ling"),
        ("D'ANGELO", "d angelo"),
        ("Bergström", "bergstrom"),
        ("Núñez", "nunez"),
        ("Smith-Jones", "smith jones"),
    ])
    def test_normalizes_to_comparable_form(self, raw, expected):
        assert normalize_name(raw) == expected

    @pytest.mark.parametrize("empty", [None, pd.NA, float("nan"), ""])
    def test_missing_values_become_empty_string(self, empty):
        assert normalize_name(empty) == ""


class TestSoundex:
    """Checked against the published NARA reference values, not against itself."""

    @pytest.mark.parametrize("word, expected", [
        ("Robert", "R163"),
        ("Rupert", "R163"),
        ("Rubin", "R150"),
        ("Ashcraft", "A261"),   # the 'h' is transparent between two 2s
        ("Tymczak", "T522"),
        ("Pfister", "P236"),    # leading double consonant coded once
        ("Honeyman", "H555"),
    ])
    def test_matches_published_reference_values(self, word, expected):
        assert soundex(word) == expected

    def test_phonetic_variants_share_a_code(self):
        assert soundex("Smith") == soundex("Smyth")

    def test_insertion_typo_survives(self):
        assert soundex("Okafor") == soundex("Okaafor")

    def test_transposition_changes_the_code(self):
        """The documented blind spot -- letter_signature covers this case."""
        assert soundex("Wong") != soundex("Wogn")

    def test_empty_input_returns_empty(self):
        assert soundex("") == ""


class TestLetterSignature:
    def test_transposition_produces_the_same_signature(self):
        assert letter_signature("wong") == letter_signature("wogn")

    def test_swapped_given_and_family_name_matches(self):
        assert letter_signature("dominic", "ng") == letter_signature("ng", "dominic")

    def test_insertion_changes_the_signature(self):
        """The mirror blind spot -- soundex covers this case."""
        assert letter_signature("okafor") != letter_signature("okaafor")

    def test_different_names_differ(self):
        assert letter_signature("aisha", "okafor") != letter_signature("daniel", "tan")


class TestNameSimilarity:
    def test_identical_names_score_one(self):
        assert name_similarity("aisha", "okafor", "aisha", "okafor") == 1.0

    def test_single_typo_scores_high_but_not_perfect(self):
        score = name_similarity("hana", "okafor", "hana", "okaafor")
        assert 0.9 < score < 1.0

    def test_swapped_given_and_family_name_still_matches(self):
        """"Ng Dominic" and "Dominic Ng" are the same person."""
        assert name_similarity("dominic", "ng", "ng", "dominic") == 1.0

    def test_different_people_score_low(self):
        assert name_similarity("aisha", "okafor", "daniel", "tan") < 0.5

    @pytest.mark.parametrize("first, last", [("", ""), ("", "")])
    def test_empty_name_scores_zero(self, first, last):
        assert name_similarity(first, last, "aisha", "okafor") == 0.0


class TestDobRelation:
    def test_identical_dates_are_exact(self):
        d = pd.Timestamp("1985-04-12")
        assert dob_relation(d, d) == "exact"

    def test_day_month_swap_is_transposed(self):
        assert dob_relation(
            pd.Timestamp("1985-04-12"), pd.Timestamp("1985-12-04")
        ) == "transposed"

    def test_same_year_different_date(self):
        assert dob_relation(
            pd.Timestamp("1985-04-12"), pd.Timestamp("1985-07-30")
        ) == "same_year"

    def test_different_years_are_not_related(self):
        assert dob_relation(
            pd.Timestamp("1985-04-12"), pd.Timestamp("1990-04-12")
        ) is None

    @pytest.mark.parametrize("a, b", [
        (pd.NaT, pd.Timestamp("1985-04-12")),
        (pd.Timestamp("1985-04-12"), pd.NaT),
        (pd.NaT, pd.NaT),
    ])
    def test_missing_dates_are_unknown_not_disagreement(self, a, b):
        """Absence of evidence is not evidence of difference.

        A missing date must not be treated the same as a conflicting one: the
        first leaves the question open, the second answers it.
        """
        assert dob_relation(a, b) == "unknown"


class TestFindFuzzyDuplicates:
    def test_finds_a_typo_under_the_same_date_of_birth(self):
        pairs, stats = find_fuzzy_duplicates(make_frame([
            {"patient_id": "P1", "first_name": "Hana", "last_name": "Okafor",
             "date_of_birth": "1940-08-15"},
            {"patient_id": "P2", "first_name": "Hana", "last_name": "Okaafor",
             "date_of_birth": "1940-08-15"},
        ]))

        assert len(pairs) == 1
        assert set(pairs.loc[0, ["patient_id_a", "patient_id_b"]]) == {"P1", "P2"}
        assert pairs.loc[0, "dob_relation"] == "exact"
        assert pairs.loc[0, "confidence"] == "high"
        assert stats["candidate_pairs"] == 1
        assert stats["records_involved"] == 2

    def test_does_not_match_different_people_sharing_a_birthday(self):
        """The expensive error. Same date of birth is not evidence on its own."""
        pairs, _ = find_fuzzy_duplicates(make_frame([
            {"patient_id": "P1", "first_name": "Aisha", "last_name": "Okafor",
             "date_of_birth": "1985-04-12"},
            {"patient_id": "P2", "first_name": "Daniel", "last_name": "Tan",
             "date_of_birth": "1985-04-12"},
        ]))

        assert pairs.empty

    def test_matches_across_a_transposed_date_of_birth(self):
        pairs, _ = find_fuzzy_duplicates(make_frame([
            {"patient_id": "P1", "first_name": "Hana", "last_name": "Adeyemi",
             "date_of_birth": "1952-10-04"},
            {"patient_id": "P2", "first_name": "Hana", "last_name": "Adeyemi",
             "date_of_birth": "1952-04-10"},
        ]))

        assert len(pairs) == 1
        assert pairs.loc[0, "dob_relation"] == "transposed"
        assert pairs.loc[0, "confidence"] == "medium"

    def test_weaker_date_evidence_demands_stronger_name_evidence(self):
        """A name pair good enough under an exact date is rejected under a
        transposed one, because the strict threshold applies."""
        borderline = [
            {"patient_id": "P1", "first_name": "Mei Ling", "last_name": "Nguyen"},
            {"patient_id": "P2", "first_name": "Meil Ing", "last_name": "Nguyen"},
        ]
        score = name_similarity("mei ling", "nguyen", "meil ing", "nguyen")
        assert DEFAULT_THRESHOLD <= score < 0.95, "test fixture no longer sits in the band"

        exact = make_frame([{**borderline[0], "date_of_birth": "1942-12-30"},
                            {**borderline[1], "date_of_birth": "1942-12-30"}])
        transposed = make_frame([{**borderline[0], "date_of_birth": "1942-12-06"},
                                 {**borderline[1], "date_of_birth": "1942-06-12"}])

        assert len(find_fuzzy_duplicates(exact)[0]) == 1
        assert find_fuzzy_duplicates(transposed)[0].empty

    def test_records_without_a_date_of_birth_match_via_name_only_blocking(self):
        """They used to be unmatchable. Phonetic and signature blocking give
        them a route in, at the strictest threshold and lowest confidence."""
        pairs, stats = find_fuzzy_duplicates(make_frame([
            {"patient_id": "P1", "first_name": "Lucia", "last_name": "Wong",
             "date_of_birth": None},
            {"patient_id": "P2", "first_name": "Lucia", "last_name": "Wong",
             "date_of_birth": None},
        ]))

        assert len(pairs) == 1
        assert pairs.loc[0, "dob_relation"] == "unknown"
        assert pairs.loc[0, "confidence"] == "low"
        assert stats["records_without_dob"] == 2
        assert stats["pairs_from_name_only_blocks"] == 1

    def test_a_dateless_record_can_match_one_that_has_a_date(self):
        """The valuable case: the record missing a date of birth is usually the
        incomplete re-registration, not the original."""
        pairs, _ = find_fuzzy_duplicates(make_frame([
            {"patient_id": "P1", "first_name": "Aisha", "last_name": "Okafor",
             "date_of_birth": "1985-04-12"},
            {"patient_id": "P2", "first_name": "Aisha", "last_name": "Okafor",
             "date_of_birth": None},
        ]))

        assert len(pairs) == 1
        assert pairs.loc[0, "dob_relation"] == "unknown"

    def test_name_only_threshold_is_stricter_than_the_dated_one(self):
        """Name-only matching sits above the default, because the name is the
        only evidence there is."""
        assert NO_DOB_THRESHOLD > DEFAULT_THRESHOLD

    def test_name_only_threshold_is_configurable_and_binding(self):
        """The same pair is accepted or rejected purely by where the bar sits.

        The pair scores ~0.93, which is why the sweep in the README matters:
        the default was moved from 0.95 to 0.92 on measured evidence, and this
        pair is exactly the kind that decision governs.
        """
        frame = make_frame([
            {"patient_id": "P1", "first_name": "Mei Ling", "last_name": "Nguyen",
             "date_of_birth": None},
            {"patient_id": "P2", "first_name": "Meil Ing", "last_name": "Nguyen",
             "date_of_birth": None},
        ])
        score = name_similarity("mei ling", "nguyen", "meil ing", "nguyen")
        assert 0.92 <= score < 0.95, "test fixture no longer straddles the band"

        assert find_fuzzy_duplicates(frame, no_dob_threshold=0.95)[0].empty
        assert len(find_fuzzy_duplicates(frame, no_dob_threshold=0.92)[0]) == 1


    def test_dateless_records_with_different_names_still_do_not_match(self):
        pairs, _ = find_fuzzy_duplicates(make_frame([
            {"patient_id": "P1", "first_name": "Aisha", "last_name": "Okafor",
             "date_of_birth": None},
            {"patient_id": "P2", "first_name": "Daniel", "last_name": "Tan",
             "date_of_birth": None},
        ]))

        assert pairs.empty

    def test_records_with_neither_date_nor_name_are_counted_unmatchable(self):
        _, stats = find_fuzzy_duplicates(make_frame([
            {"patient_id": "P1", "first_name": None, "last_name": None,
             "date_of_birth": None},
            {"patient_id": "P2", "first_name": "Aisha", "last_name": "Okafor",
             "date_of_birth": "1985-04-12"},
        ]))

        assert stats["unmatchable_records"] == 1

    def test_different_birth_years_never_match(self):
        pairs, _ = find_fuzzy_duplicates(make_frame([
            {"patient_id": "P1", "first_name": "Aisha", "last_name": "Okafor",
             "date_of_birth": "1985-04-12"},
            {"patient_id": "P2", "first_name": "Aisha", "last_name": "Okafor",
             "date_of_birth": "1990-04-12"},
        ]))

        assert pairs.empty

    def test_threshold_is_respected(self):
        frame = make_frame([
            {"patient_id": "P1", "first_name": "Lucia", "last_name": "Wong",
             "date_of_birth": "1960-01-01"},
            {"patient_id": "P2", "first_name": "Ulcia", "last_name": "Wogn",
             "date_of_birth": "1960-01-01"},
        ])

        assert find_fuzzy_duplicates(frame, threshold=0.88)[0].empty
        assert len(find_fuzzy_duplicates(frame, threshold=0.75)[0]) == 1

    def test_empty_result_still_has_the_expected_columns(self):
        pairs, _ = find_fuzzy_duplicates(make_frame([
            {"patient_id": "P1", "first_name": "Aisha", "last_name": "Okafor",
             "date_of_birth": "1985-04-12"},
        ]))

        assert pairs.empty
        assert "patient_id_a" in pairs.columns
        assert "confidence" in pairs.columns

    def test_blocking_avoids_comparing_every_pair(self):
        """30 records with distinct birth years must not produce 435 comparisons."""
        frame = make_frame([
            {"patient_id": f"P{i}", "first_name": "Aisha", "last_name": "Okafor",
             "date_of_birth": f"{1950 + i}-04-12"}
            for i in range(30)
        ])
        _, stats = find_fuzzy_duplicates(frame)

        assert stats["pairs_compared"] == 0  # no two records share a blocking key


class TestGenderVeto:
    """A known gender conflict rules out a name-only match, and only those."""

    def _dateless_pair(self, gender_a, gender_b):
        frame = make_frame([
            {"patient_id": "P1", "first_name": "Elena", "last_name": "Kumar",
             "date_of_birth": None},
            {"patient_id": "P2", "first_name": "Elena", "last_name": "Kumar",
             "date_of_birth": None},
        ])
        frame["gender"] = [gender_a, gender_b]
        return frame

    def test_conflicting_known_genders_veto_a_name_only_match(self):
        pairs, stats = find_fuzzy_duplicates(self._dateless_pair("M", "F"))

        assert pairs.empty
        assert stats["vetoed_on_gender_conflict"] == 1

    def test_matching_genders_do_not_veto(self):
        assert len(find_fuzzy_duplicates(self._dateless_pair("F", "F"))[0]) == 1

    def test_unknown_gender_never_vetoes(self):
        """That value is imputed, so it is not evidence of anything."""
        assert len(find_fuzzy_duplicates(self._dateless_pair("Unknown", "F"))[0]) == 1

    def test_veto_does_not_apply_when_the_date_of_birth_agrees(self):
        """With a confirmed date, a gender mismatch is more likely a data-entry
        error than proof of two different people."""
        frame = make_frame([
            {"patient_id": "P1", "first_name": "Elena", "last_name": "Kumar",
             "date_of_birth": "1985-04-12"},
            {"patient_id": "P2", "first_name": "Elena", "last_name": "Kumar",
             "date_of_birth": "1985-04-12"},
        ])
        frame["gender"] = ["M", "F"]

        assert len(find_fuzzy_duplicates(frame)[0]) == 1


class TestAnnotate:
    def test_adds_partner_ids_and_leaves_unmatched_rows_null(self):
        frame = make_frame([
            {"patient_id": "P1", "first_name": "Hana", "last_name": "Okafor",
             "date_of_birth": "1940-08-15"},
            {"patient_id": "P2", "first_name": "Hana", "last_name": "Okaafor",
             "date_of_birth": "1940-08-15"},
            {"patient_id": "P3", "first_name": "Daniel", "last_name": "Tan",
             "date_of_birth": "1975-02-02"},
        ])
        pairs, _ = find_fuzzy_duplicates(frame)
        result = annotate(frame, pairs)

        lookup = result.set_index("patient_id")["possible_duplicate_of"]
        assert lookup["P1"] == "P2"
        assert lookup["P2"] == "P1"
        assert pd.isna(lookup["P3"])

    def test_handles_no_matches(self):
        frame = make_frame([
            {"patient_id": "P1", "first_name": "Aisha", "last_name": "Okafor",
             "date_of_birth": "1985-04-12"},
        ])
        result = annotate(frame, find_fuzzy_duplicates(frame)[0])

        assert result["possible_duplicate_of"].isna().all()
