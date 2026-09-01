"""Fuzzy duplicate detection on patient name and date of birth.

The deduplication in clean_data.py resolves records that share a patient_id.
It cannot see the harder case: the same person registered twice under two
different IDs, with a name typo, a transposed date of birth, or both. That is
the duplicate an exact key match will never catch and a human always spots.

Three design decisions worth being able to defend:

1. Candidates are FLAGGED, never merged automatically. Merging two records that
   turn out to be different people is a clinical safety incident; missing a
   duplicate is an inconvenience. Those costs are not symmetric, so this
   produces a review queue rather than a decision.

2. Comparison is blocked, not all-pairs. Comparing every record against every
   other is O(n^2) -- fine for 400 rows, ruinous at a million. Records are
   bucketed by date of birth and by (birth year + last name prefix), and only
   records sharing a bucket are ever compared.

3. Weaker date evidence demands stronger name evidence. An exact date of birth
   match clears at the normal threshold; a transposed or same-year date has to
   clear a stricter one.

Uses only difflib from the standard library, so the runtime dependencies stay
pandas and numpy.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from itertools import combinations

import pandas as pd

DEFAULT_THRESHOLD = 0.88
# Applied when the two dates of birth are not identical.
STRICT_THRESHOLD = 0.95

# A bucket this large means the blocking key is not discriminating (every
# record sharing one placeholder date, say). Comparing it would cost more than
# it is worth and would swamp the review queue, so it is skipped and reported.
MAX_BLOCK_SIZE = 100

_NON_LETTERS = re.compile(r"[^a-z\s]")
_WHITESPACE = re.compile(r"\s+")


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


def name_similarity(first_a: str, last_a: str, first_b: str, last_b: str) -> float:
    """Similarity of two full names, tolerant of given/family name being swapped.

    Registration forms routinely capture "Ng Dominic" in one system and
    "Dominic Ng" in another. Both orderings are scored and the better taken,
    so a swap costs nothing while a genuine mismatch still scores low.
    """
    left = f"{first_a} {last_a}".strip()
    right = f"{first_b} {last_b}".strip()
    swapped = f"{last_b} {first_b}".strip()

    if not left or not right:
        return 0.0

    return max(
        SequenceMatcher(None, left, right).ratio(),
        SequenceMatcher(None, left, swapped).ratio(),
    )


def dob_relation(a: pd.Timestamp, b: pd.Timestamp) -> str | None:
    """How two dates of birth relate, or None if they are simply different.

    "transposed" is the day/month swap that the date-convention ambiguity
    produces -- 04/12 read one way in one system and the other way in another.
    Those records describe the same person and must still be comparable.
    """
    if pd.isna(a) or pd.isna(b):
        return None
    if a == b:
        return "exact"
    if a.year == b.year and a.day == b.month and a.month == b.day:
        return "transposed"
    if a.year == b.year:
        return "same_year"
    return None


def _confidence(relation: str, score: float) -> str:
    if relation == "exact":
        return "high" if score >= STRICT_THRESHOLD else "medium"
    if relation == "transposed":
        return "medium"
    return "low"


def _build_blocks(dobs: list, last_keys: list[str]) -> dict:
    """Bucket record positions by the keys a duplicate is likely to share.

    Two keys, because either one alone has a blind spot: blocking on date of
    birth misses records whose date was mistyped, and blocking on name prefix
    misses records whose name was mistyped.
    """
    blocks: dict = {}
    for position, (dob, last_key) in enumerate(zip(dobs, last_keys)):
        if pd.isna(dob):
            continue
        blocks.setdefault(("dob", dob), []).append(position)
        if last_key:
            blocks.setdefault(("year_name", dob.year, last_key), []).append(position)
    return blocks


def find_fuzzy_duplicates(
    df: pd.DataFrame,
    key: str = "patient_id",
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[pd.DataFrame, dict]:
    """Return candidate duplicate pairs and the stats describing the search."""
    columns = [f"{key}_a", f"{key}_b", "name_a", "name_b",
               "date_of_birth_a", "date_of_birth_b",
               "dob_relation", "name_score", "confidence"]

    keys = df[key].tolist()
    firsts = [normalize_name(v) for v in df.get("first_name", pd.Series([None] * len(df)))]
    lasts = [normalize_name(v) for v in df.get("last_name", pd.Series([None] * len(df)))]
    if "date_of_birth" in df.columns:
        dobs = df["date_of_birth"].tolist()
    else:
        dobs = [pd.NaT] * len(df)

    last_keys = [last[:3] for last in lasts]

    blocks = _build_blocks(dobs, last_keys)

    pairs: set[tuple[int, int]] = set()
    oversized = 0
    for members in blocks.values():
        if len(members) < 2:
            continue
        if len(members) > MAX_BLOCK_SIZE:
            oversized += 1
            continue
        pairs.update(combinations(sorted(members), 2))

    matches = []
    for a, b in pairs:
        relation = dob_relation(dobs[a], dobs[b])
        if relation is None:
            continue

        score = name_similarity(firsts[a], lasts[a], firsts[b], lasts[b])
        required = threshold if relation == "exact" else STRICT_THRESHOLD
        if score < required:
            continue

        matches.append({
            f"{key}_a": keys[a],
            f"{key}_b": keys[b],
            "name_a": f"{firsts[a]} {lasts[a]}".strip(),
            "name_b": f"{firsts[b]} {lasts[b]}".strip(),
            "date_of_birth_a": dobs[a],
            "date_of_birth_b": dobs[b],
            "dob_relation": relation,
            "name_score": round(score, 3),
            "confidence": _confidence(relation, score),
        })

    result = pd.DataFrame(matches, columns=columns)
    if not result.empty:
        result = result.sort_values(
            ["confidence", "name_score"], ascending=[True, False]
        ).reset_index(drop=True)

    # Records that could not participate at all -- worth surfacing, because a
    # clean-looking zero-match result is meaningless if half the file was
    # ineligible for comparison.
    no_dob = int(sum(pd.isna(d) for d in dobs))
    no_name = int(sum(1 for f, l in zip(firsts, lasts) if not f and not l))

    stats = {
        "threshold": threshold,
        "strict_threshold": STRICT_THRESHOLD,
        "blocks_built": len(blocks),
        "pairs_compared": len(pairs),
        "candidate_pairs": len(result),
        "records_involved": int(
            pd.concat([result[f"{key}_a"], result[f"{key}_b"]]).nunique()
        ) if not result.empty else 0,
        "by_confidence": (
            result["confidence"].value_counts().to_dict() if not result.empty else {}
        ),
        "skipped_no_dob": no_dob,
        "skipped_no_name": no_name,
        "oversized_blocks_skipped": oversized,
    }
    return result, stats


def annotate(df: pd.DataFrame, pairs: pd.DataFrame, key: str = "patient_id") -> pd.DataFrame:
    """Add a possible_duplicate_of column listing each record's matched partners."""
    partners: dict = {}
    if not pairs.empty:
        for a, b in zip(pairs[f"{key}_a"], pairs[f"{key}_b"]):
            partners.setdefault(a, set()).add(b)
            partners.setdefault(b, set()).add(a)

    df["possible_duplicate_of"] = [
        ";".join(sorted(partners[k])) if k in partners else pd.NA for k in df[key]
    ]
    return df



