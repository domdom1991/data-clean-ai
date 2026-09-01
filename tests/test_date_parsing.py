"""Tests for the date parsing stage.

These cover the decisions that are easy to get wrong and impossible to notice
afterwards: a date that parses cleanly under the wrong convention produces no
error, no null, and a silently wrong answer.
"""

import pandas as pd
import pytest

from clean_data import fix_two_digit_century, infer_day_first, parse_date_series


class TestParseDateSeries:
    """Every format the source systems emit must land on the same date."""

    # Each of these is 12 April 1985 written a different way.
    @pytest.mark.parametrize("raw", [
        "1985-04-12",
        "1985/04/12",
        "12/04/1985",
        "12.04.1985",
        "12-Apr-1985",
        "12 Apr 1985",
        "April 12, 1985",
        "19850412",
        "12/04/85",
    ])
    def test_every_known_format_resolves_to_the_same_date(self, raw):
        result = parse_date_series(pd.Series([raw]), day_first=True)
        assert result.iloc[0] == pd.Timestamp("1985-04-12")

    @pytest.mark.parametrize("raw", ["1985-04-12 09:30:00", "1985-04-12T09:30:00"])
    def test_timestamps_keep_their_time_component(self, raw):
        """Parsing preserves the time; dropping it to day grain is a separate,
        deliberate step in standardize_dates, not a side effect of parsing."""
        result = parse_date_series(pd.Series([raw]), day_first=True)

        assert result.iloc[0] == pd.Timestamp("1985-04-12 09:30:00")
        assert result.dt.normalize().iloc[0] == pd.Timestamp("1985-04-12")

    def test_mixed_formats_in_one_column_all_parse(self):
        """The real case: one column, many formats, no nulls in the output."""
        series = pd.Series([
            "1985-04-12", "12/04/1985", "12-Apr-1985", "April 12, 1985",
            "12.04.1985", "19850412",
        ])
        result = parse_date_series(series, day_first=True)

        assert result.notna().all()
        assert (result == pd.Timestamp("1985-04-12")).all()

    def test_unparseable_values_become_nat_rather_than_raising(self):
        series = pd.Series(["1985-04-12", "not a date", "31/31/1985", ""])
        result = parse_date_series(series, day_first=True)

        assert result.iloc[0] == pd.Timestamp("1985-04-12")
        assert result.iloc[1:].isna().all()

    def test_missing_values_stay_missing(self):
        series = pd.Series([None, "1985-04-12", None])
        result = parse_date_series(series, day_first=True)

        assert result.isna().tolist() == [True, False, True]

    def test_ambiguous_value_follows_the_chosen_convention(self):
        """04/12/1985 is a real date under both readings -- the convention decides.

        This is the whole reason the convention has to be inferred and reported
        rather than assumed: neither reading fails, so nothing warns you.
        """
        series = pd.Series(["04/12/1985"])

        assert parse_date_series(series, day_first=True).iloc[0] == pd.Timestamp("1985-12-04")
        assert parse_date_series(series, day_first=False).iloc[0] == pd.Timestamp("1985-04-12")

    def test_unambiguous_value_parses_under_either_convention(self):
        """25 cannot be a month, so the day-first reading must win regardless."""
        series = pd.Series(["25/04/1985"])

        for day_first in (True, False):
            result = parse_date_series(series, day_first=day_first)
            assert result.iloc[0] == pd.Timestamp("1985-04-25")

    def test_index_is_preserved(self):
        series = pd.Series(["1985-04-12", "1990-01-01"], index=[7, 42])
        result = parse_date_series(series, day_first=True)

        assert result.index.tolist() == [7, 42]


class TestInferDayFirst:
    """The convention is inferred from values that resolve themselves."""

    def test_day_first_evidence_wins(self):
        # 25 and 31 can only be days, and they sit in the first position.
        series = pd.Series(["25/04/1985", "31/01/1990", "04/12/1985"])
        day_first, evidence = infer_day_first(series)

        assert day_first is True
        assert evidence["day_first_evidence"] == 2
        assert evidence["month_first_evidence"] == 0

    def test_month_first_evidence_wins(self):
        # 25 and 31 can only be days, and here they sit in the second position.
        series = pd.Series(["04/25/1985", "01/31/1990", "04/12/1985"])
        day_first, evidence = infer_day_first(series)

        assert day_first is False
        assert evidence["day_first_evidence"] == 0
        assert evidence["month_first_evidence"] == 2

    def test_defaults_to_day_first_when_no_evidence_exists(self):
        """Every value is ambiguous, so the documented default applies."""
        series = pd.Series(["04/12/1985", "01/02/1990"])
        day_first, evidence = infer_day_first(series)

        assert day_first is True
        assert evidence == {"day_first_evidence": 0, "month_first_evidence": 0}

    def test_handles_an_all_missing_column_without_raising(self):
        day_first, evidence = infer_day_first(pd.Series([None, None], dtype="object"))

        assert day_first is True
        assert evidence["day_first_evidence"] == 0

    def test_ignores_non_slash_formats(self):
        """ISO and month-name dates carry no evidence about slash ordering."""
        series = pd.Series(["1985-04-12", "April 12, 1985", "25/04/1985"])
        _, evidence = infer_day_first(series)

        assert evidence["day_first_evidence"] == 1

    def test_dot_separated_dates_count_as_evidence(self):
        series = pd.Series(["25.04.1985", "31.01.1990"])
        day_first, evidence = infer_day_first(series)

        assert day_first is True
        assert evidence["day_first_evidence"] == 2


class TestFixTwoDigitCentury:
    """%y maps 55 to 2055, so a 1955 birth date lands in the future."""

    AS_OF = pd.Timestamp("2026-09-01")

    def test_future_birth_date_is_pulled_back_a_century(self):
        dates = pd.Series([pd.Timestamp("2055-04-12")])
        fixed, count = fix_two_digit_century(dates, self.AS_OF)

        assert fixed.iloc[0] == pd.Timestamp("1955-04-12")
        assert count == 1

    def test_past_dates_are_left_alone(self):
        dates = pd.Series([pd.Timestamp("1985-04-12"), pd.Timestamp("2020-01-01")])
        fixed, count = fix_two_digit_century(dates, self.AS_OF)

        assert fixed.tolist() == dates.tolist()
        assert count == 0

    def test_missing_dates_are_untouched(self):
        dates = pd.Series([pd.NaT, pd.Timestamp("2055-01-01")])
        fixed, count = fix_two_digit_century(dates, self.AS_OF)

        assert pd.isna(fixed.iloc[0])
        assert fixed.iloc[1] == pd.Timestamp("1955-01-01")
        assert count == 1

    def test_counts_every_correction(self):
        dates = pd.Series([
            pd.Timestamp("2055-01-01"), pd.Timestamp("2040-06-15"),
            pd.Timestamp("1985-04-12"),
        ])
        _, count = fix_two_digit_century(dates, self.AS_OF)

        assert count == 2

    def test_two_digit_year_round_trip_through_the_parser(self):
        """End to end: '12/04/55' must come out as 1955, not 2055."""
        parsed = parse_date_series(pd.Series(["12/04/55"]), day_first=True)
        fixed, count = fix_two_digit_century(parsed, self.AS_OF)

        assert fixed.iloc[0] == pd.Timestamp("1955-04-12")
        assert count == 1
