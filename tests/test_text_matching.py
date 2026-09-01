"""Tests for the string similarity and phonetic encoding primitives.

These are pure functions with published reference values, so wherever such
values exist the tests assert against *them* rather than against whatever this
implementation happens to produce. A test that only checks the code agrees with
itself would pass just as happily on a broken algorithm.

Metaphone is the exception: it has no single canonical implementation and
published libraries disagree on specific codes, so those tests assert
equivalence properties instead -- which is what blocking actually depends on.
"""

import pandas as pd
import pytest

from text_matching import (
    jaro,
    jaro_winkler,
    letter_signature,
    metaphone,
    normalize_name,
    soundex,
)


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


class TestJaro:
    """Reference values from Winkler's published worked examples."""

    @pytest.mark.parametrize("a, b, expected", [
        ("MARTHA", "MARHTA", 0.944),
        ("DIXON", "DICKSONX", 0.767),
        ("DWAYNE", "DUANE", 0.822),
        ("JELLYFISH", "SMELLYFISH", 0.896),
        ("CRATE", "TRACE", 0.733),
        ("ABCVWXYZ", "CABVWXYZ", 0.958),
    ])
    def test_matches_published_values(self, a, b, expected):
        assert jaro(a, b) == pytest.approx(expected, abs=0.001)

    def test_match_window_excludes_distant_characters(self):
        """Pins down the window as floor(max(len)/2) - 1, which no published
        example above happens to discriminate.

        For two 4-character strings the window is 1, so a character may only
        match one at most one position away. Here s1's 'b' at index 3 cannot
        reach s2's 'b' at index 1 -- two apart, outside the window -- so only
        the leading 'a' matches: (1/4 + 1/4 + 1/1) / 3 = 0.5 exactly. A window
        one wider would pair the 'b's and return 0.667.
        """
        assert jaro("aaab", "abcd") == pytest.approx(0.5)

    def test_identical_strings_score_one(self):
        assert jaro("okafor", "okafor") == 1.0

    def test_nothing_in_common_scores_zero(self):
        assert jaro("abc", "xyz") == 0.0

    @pytest.mark.parametrize("a, b", [("", "abc"), ("abc", ""), ("", "")])
    def test_empty_input(self, a, b):
        expected = 1.0 if a == b else 0.0
        assert jaro(a, b) == expected

    def test_is_symmetric(self):
        assert jaro("martha", "marhta") == pytest.approx(jaro("marhta", "martha"))


class TestJaroWinkler:
    @pytest.mark.parametrize("a, b, expected", [
        ("MARTHA", "MARHTA", 0.961),
        ("DIXON", "DICKSONX", 0.813),
        ("DWAYNE", "DUANE", 0.840),
        ("JELLYFISH", "SMELLYFISH", 0.896),  # no shared prefix, so no boost
    ])
    def test_matches_published_values(self, a, b, expected):
        assert jaro_winkler(a, b) == pytest.approx(expected, abs=0.001)

    def test_shared_prefix_raises_the_score_above_plain_jaro(self):
        assert jaro_winkler("martha", "marhta") > jaro("martha", "marhta")

    def test_no_shared_prefix_leaves_the_score_untouched(self):
        assert jaro_winkler("jellyfish", "smellyfish") == jaro("jellyfish", "smellyfish")

    def test_prefix_bonus_caps_at_four_characters(self):
        """Winkler's definition considers at most the first four characters."""
        long_prefix = jaro_winkler("abcdefgh", "abcdefXX")
        assert long_prefix == pytest.approx(
            jaro("abcdefgh", "abcdefXX")
            + 4 * 0.1 * (1 - jaro("abcdefgh", "abcdefXX"))
        )

    def test_poor_matches_get_no_boost(self):
        """Below the boost threshold a shared prefix must not rescue a bad match."""
        assert jaro_winkler("abcde", "abzzzzzzzz") == jaro("abcde", "abzzzzzzzz")

    def test_beats_difflib_on_short_names(self):
        """The reason for the switch: a subsequence ratio punishes one dropped
        letter far too harshly when the string is only four characters long."""
        from difflib import SequenceMatcher

        assert SequenceMatcher(None, "rosa", "roa").ratio() < 0.88
        assert jaro_winkler("rosa", "roa") > 0.92


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


class TestMetaphone:
    """Asserts equivalence properties rather than exact codes.

    Metaphone has no canonical implementation -- published libraries disagree
    on specific outputs -- so pinning exact strings would be testing this
    implementation's quirks. What blocking actually needs is that names
    pronounced alike collide, and names pronounced differently do not.
    """

    @pytest.mark.parametrize("a, b", [
        ("Smith", "Smyth"),      # y for i
        ("Knight", "Night"),     # silent initial k
        ("Wright", "Rite"),      # silent initial w, gh
        ("Phillip", "Fillip"),   # ph -> f
        ("Gnome", "Nome"),       # silent initial g
        ("Wrigley", "Rigley"),   # silent initial w
    ])
    def test_names_pronounced_alike_collide(self, a, b):
        assert metaphone(a) == metaphone(b)

    def test_gh_is_silent_unless_a_vowel_follows(self):
        assert metaphone("Night") == metaphone("Nite")
        assert "K" in metaphone("Ghost")

    @pytest.mark.parametrize("a, b", [
        ("Smith", "Jones"),
        ("Okafor", "Adeyemi"),
        ("Chen", "Tan"),
    ])
    def test_names_pronounced_differently_do_not_collide(self, a, b):
        assert metaphone(a) != metaphone(b)

    def test_is_more_selective_than_soundex(self):
        """The whole reason to prefer it: fewer accidental collisions, so
        fewer wasted comparisons per block."""
        names = ["Smith", "Smyth", "Sneed", "Snead", "Schmidt", "Sandy",
                 "Swanson", "Simon", "Symon", "Samson"]

        assert len({metaphone(n) for n in names}) > len({soundex(n) for n in names})

    def test_vowels_survive_only_in_first_position(self):
        assert metaphone("Amara").startswith("A")
        assert "A" not in metaphone("Kumar")

    def test_empty_input_returns_empty(self):
        assert metaphone("") == ""

    def test_ignores_case_and_punctuation(self):
        assert metaphone("O'Brien") == metaphone("obrien")


class TestLetterSignature:
    def test_transposition_produces_the_same_signature(self):
        assert letter_signature("wong") == letter_signature("wogn")

    def test_swapped_given_and_family_name_matches(self):
        assert letter_signature("dominic", "ng") == letter_signature("ng", "dominic")

    def test_insertion_changes_the_signature(self):
        """The mirror blind spot -- the phonetic codes cover this case."""
        assert letter_signature("okafor") != letter_signature("okaafor")

    def test_different_names_differ(self):
        assert letter_signature("aisha", "okafor") != letter_signature("daniel", "tan")
