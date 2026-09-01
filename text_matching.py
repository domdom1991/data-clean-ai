"""String similarity and phonetic encoding primitives.

Pure functions over strings, with no knowledge of patients or DataFrames, so
they can be tested against published reference values rather than against the
pipeline's own behaviour. `fuzzy_match.py` holds the matching policy that uses
them.

Two families:

  Similarity  jaro, jaro_winkler -- how alike are two strings?
  Phonetic    soundex, metaphone, letter_signature -- do two strings collide
              under some equivalence, cheap enough to bucket a million records?

Standard library only, so the runtime dependencies stay pandas and numpy.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

_NON_LETTERS = re.compile(r"[^a-z\s]")
_WHITESPACE = re.compile(r"\s+")

VOWELS = "AEIOU"


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

def normalize_name(value: object) -> str:
    """Casefold, strip accents and punctuation, collapse whitespace.

    "D'Souza", "DSouza" and "d souza" all have to reduce to the same string
    before any similarity score means anything.
    """
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value)).lower()
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return _WHITESPACE.sub(" ", _NON_LETTERS.sub(" ", text)).strip()


# --------------------------------------------------------------------------
# Similarity
# --------------------------------------------------------------------------

def jaro(s1: str, s2: str) -> float:
    """Jaro similarity: matching characters within a sliding window, minus
    half a penalty for each transposition.

    Unlike a longest-common-subsequence ratio, it normalises by the length of
    *both* strings, which is why it stays sensible on short inputs where one
    dropped letter is a large fraction of the whole word.
    """
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    window = max(max(len(s1), len(s2)) // 2 - 1, 0)

    s1_matched = [False] * len(s1)
    s2_matched = [False] * len(s2)
    matches = 0

    for i, ch in enumerate(s1):
        for j in range(max(0, i - window), min(i + window + 1, len(s2))):
            if s2_matched[j] or s2[j] != ch:
                continue
            s1_matched[i] = s2_matched[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    # Count matched characters that appear in a different order in each string.
    transpositions = 0
    k = 0
    for i in range(len(s1)):
        if not s1_matched[i]:
            continue
        while not s2_matched[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    transpositions //= 2

    return (
        matches / len(s1)
        + matches / len(s2)
        + (matches - transpositions) / matches
    ) / 3


def jaro_winkler(s1: str, s2: str, prefix_weight: float = 0.1,
                 max_prefix: int = 4, boost_threshold: float = 0.7) -> float:
    """Jaro, with a bonus for a shared prefix.

    Winkler's observation was that people mistype the ends of names far more
    often than the beginnings, so agreement on the first few characters is
    stronger evidence than agreement anywhere else. The bonus only applies
    above `boost_threshold` -- boosting an already-poor match would be noise.
    """
    score = jaro(s1, s2)
    if score < boost_threshold:
        return score

    prefix = 0
    for a, b in zip(s1[:max_prefix], s2[:max_prefix]):
        if a != b:
            break
        prefix += 1

    return score + prefix * prefix_weight * (1 - score)


# --------------------------------------------------------------------------
# Phonetic and structural keys
# --------------------------------------------------------------------------

_SOUNDEX_CODES = {
    letter: digit
    for letters, digit in [("bfpv", "1"), ("cgjkqsxz", "2"), ("dt", "3"),
                           ("l", "4"), ("mn", "5"), ("r", "6")]
    for letter in letters
}


def soundex(word: str) -> str:
    """Classic NARA Soundex: first letter plus three consonant-group digits.

    Groups letters that sound alike, so "Smith" and "Smyth" share a code. Note
    what it does NOT do: it is anchored on the first letter and it is order
    sensitive, so a transposition ("Wong" -> "Wogn") produces a different code.
    That blind spot is why letter_signature exists alongside it.
    """
    word = "".join(ch for ch in word.lower() if ch.isalpha())
    if not word:
        return ""

    codes = [_SOUNDEX_CODES.get(ch, "") for ch in word]
    digits: list[str] = []
    previous = codes[0]
    for ch, code in zip(word[1:], codes[1:]):
        if code and code != previous:
            digits.append(code)
        # h and w are transparent: letters either side of them still count as
        # adjacent, so they do not reset the previous code.
        if ch not in "hw":
            previous = code

    return (word[0].upper() + "".join(digits) + "000")[:4]


def metaphone(word: str) -> str:
    """Lawrence Philips' Metaphone, encoding English pronunciation.

    Finer-grained than Soundex: it models digraphs (PH -> F, TH -> 0, SH -> X)
    and silent letters rather than lumping consonants into six buckets, so it
    collides far less often on names that merely look similar.

    Unlike Soundex there is no single canonical implementation -- published
    libraries disagree on specific codes -- so the tests assert equivalence
    properties ("Smith" and "Smyth" must collide) rather than exact strings.
    """
    word = "".join(ch for ch in word.upper() if ch.isalpha())
    if not word:
        return ""

    # Silent leading letters, and initial digraphs that are pronounced oddly.
    if word[:2] in ("AE", "GN", "KN", "PN", "WR"):
        word = word[1:]
    elif word[0] == "X":
        word = "S" + word[1:]
    elif word[:2] == "WH":
        word = "W" + word[2:]

    out: list[str] = []
    length = len(word)
    i = 0

    while i < length:
        ch = word[i]
        prev = word[i - 1] if i else ""
        nxt = word[i + 1] if i + 1 < length else ""
        nxt2 = word[i + 2] if i + 2 < length else ""

        # Doubled letters are pronounced once. CC is the exception (accent).
        if ch == prev and ch != "C":
            i += 1
            continue

        if ch in VOWELS:
            if i == 0:  # vowels only survive in first position
                out.append(ch)
        elif ch == "B":
            if not (i == length - 1 and prev == "M"):  # silent in "lamb"
                out.append("B")
        elif ch == "C":
            if nxt == "I" and nxt2 == "A":              # "-cia-"
                out.append("X")
            elif nxt == "H":                            # "-ch-"
                out.append("K" if prev == "S" else "X")  # "school" vs "chair"
            elif nxt in "IEY":
                if prev != "S":                          # silent in "-sci-"
                    out.append("S")
            else:
                out.append("K")
        elif ch == "D":
            if nxt == "G" and nxt2 in "EYI":            # "-dge-"
                out.append("J")
                i += 1
            else:
                out.append("T")
        elif ch == "G":
            if nxt == "H":
                # GH is silent unless a vowel follows it: "night", "through"
                # and a trailing "-ough" all drop it, "ghost" keeps it.
                if nxt2 in VOWELS:
                    out.append("K")
            elif nxt == "N":                             # silent in "sign"
                pass
            elif nxt in "IEY":
                out.append("J")
            else:
                out.append("K")
        elif ch == "H":
            # Silent after a vowel with no vowel following, and after the
            # consonants whose digraph already consumed it.
            if prev in VOWELS and nxt not in VOWELS:
                pass
            elif prev in "CSPTG":
                pass
            else:
                out.append("H")
        elif ch in "FJKLMNR":
            out.append(ch)
        elif ch == "P":
            out.append("F" if nxt == "H" else "P")
        elif ch == "Q":
            out.append("K")
        elif ch == "S":
            if nxt == "H" or (nxt == "I" and nxt2 in "OA"):
                out.append("X")
            else:
                out.append("S")
        elif ch == "T":
            if nxt == "I" and nxt2 in "OA":
                out.append("X")
            elif nxt == "H":
                out.append("0")  # theta
            else:
                out.append("T")
        elif ch == "V":
            out.append("F")
        elif ch == "W":
            if nxt in VOWELS:
                out.append("W")
        elif ch == "X":
            out.append("KS")
        elif ch == "Y":
            if nxt in VOWELS:
                out.append("Y")
        elif ch == "Z":
            out.append("S")

        i += 1

    return "".join(out)


def letter_signature(*parts: str) -> str:
    """The name's letters, sorted -- identical for any reordering of them.

    Complements the phonetic codes: this catches transpositions ("Wong"/"Wogn",
    and even a swapped given/family name) but is blind to insertions and
    deletions, which is exactly the case they handle.
    """
    return "".join(sorted("".join(parts).replace(" ", "")))
