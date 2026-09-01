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
   bucketed by date of birth, by (birth year + last name prefix), and -- for
   records with no date at all -- by phonetic and letter-signature keys. Only
   records sharing a bucket are ever compared.

3. Weaker date evidence demands stronger name evidence. An exact date of birth
   match clears at the normal threshold; a transposed or same-year date has to
   clear a stricter one; a missing date is stricter still.

String algorithms live in text_matching.py. This module holds only the policy
that uses them.
"""

from __future__ import annotations

from itertools import combinations

import pandas as pd

from text_matching import (
    jaro_winkler,
    letter_signature,
    metaphone,
    normalize_name,
    soundex,
)

# Thresholds are Jaro-Winkler scores, re-tuned when the scorer moved off
# difflib -- Jaro-Winkler is systematically more generous, so the old values
# would have been far too permissive. Each sits below the lowest observed true
# match in its tier, with margin. See the README for the score distribution.
DEFAULT_THRESHOLD = 0.90       # lowest true match with an exact date: 0.915
# Applied when the two dates of birth are not identical.
STRICT_THRESHOLD = 0.96        # lowest true match with a transposed date: 0.986
# Applied when at least one record has no date of birth at all, so the name is
# the only evidence there is. Note that no threshold makes this tier clean: its
# worst false positive is two different patients with an identical name, which
# scores a perfect 1.0 -- higher than any true match. That is why these land in
# a separate low-confidence tier for review rather than being acted on.
NO_DOB_THRESHOLD = 0.95        # lowest true match with no date: 0.983

# A bucket this large means the blocking key is not discriminating (every
# record sharing one placeholder date, say). Comparing it would cost more than
# it is worth and would swamp the review queue, so it is skipped and reported.
MAX_BLOCK_SIZE = 100


def name_similarity(first_a: str, last_a: str, first_b: str, last_b: str) -> float:
    """Similarity of two full names, tolerant of given/family name being swapped.

    Jaro-Winkler rather than difflib's ratio. Names are short, and a
    subsequence ratio punishes a single dropped letter far too harshly at that
    length: "rosa" vs "roa" scores 0.857 under difflib but 0.933 under
    Jaro-Winkler, which is the difference between missing a real duplicate and
    catching it. Winkler's prefix bonus also matches how names are actually
    mistyped -- people fumble the end of a name far more often than the start.

    Registration forms routinely capture "Ng Dominic" in one system and
    "Dominic Ng" in another. Both orderings are scored and the better taken,
    so a swap costs nothing while a genuine mismatch still scores low.
    """
    left = f"{first_a} {last_a}".strip()
    right = f"{first_b} {last_b}".strip()
    swapped = f"{last_b} {first_b}".strip()

    if not left or not right:
        return 0.0

    return max(jaro_winkler(left, right), jaro_winkler(left, swapped))


def dob_relation(a: pd.Timestamp, b: pd.Timestamp) -> str | None:
    """How two dates of birth relate. None means they positively disagree.

    "transposed" is the day/month swap that the date-convention ambiguity
    produces -- 04/12 read one way in one system and the other way in another.
    Those records describe the same person and must still be comparable.

    A missing date returns "unknown", not None. Absence of evidence is not
    evidence of difference: two records can still be the same person when one
    of them never captured a date of birth. Two records with *different* dates
    of birth are a different matter -- that is real evidence they are two
    different people, so it returns None and the pair is dropped.
    """
    if pd.isna(a) or pd.isna(b):
        return "unknown"
    if a == b:
        return "exact"
    if a.year == b.year and a.day == b.month and a.month == b.day:
        return "transposed"
    if a.year == b.year:
        return "same_year"
    return None


def _required_score(relation: str, threshold: float, no_dob_threshold: float) -> float:
    """The name score a pair must reach, given how strong the date evidence is."""
    if relation == "exact":
        return threshold
    if relation == "unknown":
        return no_dob_threshold
    return STRICT_THRESHOLD  # transposed or same_year


def _genders_conflict(a: object, b: object) -> bool:
    """True only when both genders are known and they disagree.

    Used to veto name-only matches. Two records sharing a common name with no
    date of birth between them are weak evidence at the best of times; a known
    gender conflict is enough to rule them out. "Unknown" never vetoes, because
    that value is itself imputed.
    """
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return False
    if "Unknown" in (a, b):
        return False
    return a != b


def _confidence(relation: str, score: float) -> str:
    if relation == "exact":
        return "high" if score >= STRICT_THRESHOLD else "medium"
    if relation == "transposed":
        return "medium"
    return "low"


def _build_blocks(dobs: list, last_keys: list[str],
                  firsts: list[str], lasts: list[str]) -> tuple[dict, dict]:
    """Bucket record positions by the keys a duplicate is likely to share.

    Returns two families of block, because they serve different purposes.

    Dated blocks key on the date of birth, which is the strongest evidence
    available. Two keys, because either alone has a blind spot: blocking on the
    exact date misses a mistyped date, blocking on name prefix misses a
    mistyped name.

    Name-only blocks exist solely so that records with NO date of birth can be
    compared at all -- otherwise they could never enter a block and would be
    silently unmatchable. Three keys, covering different failure modes:

      soundex    coarse phonetic grouping; high recall, collides often
      metaphone  finer phonetic grouping; models digraphs and silent letters
      signature  sorted letters; the only one that survives a transposition

    Soundex and Metaphone overlap heavily by design. Metaphone is the more
    precise of the two, but Soundex's very coarseness occasionally rescues a
    pair Metaphone splits, and a redundant blocking key costs only comparisons
    -- the scoring threshold still decides what actually matches.
    """
    dated: dict = {}
    name_only: dict = {}

    for position, (dob, last_key) in enumerate(zip(dobs, last_keys)):
        if not pd.isna(dob):
            dated.setdefault(("dob", dob), []).append(position)
            if last_key:
                dated.setdefault(("year_name", dob.year, last_key), []).append(position)

        first, last = firsts[position], lasts[position]
        if not first and not last:
            continue
        name_only.setdefault(
            ("soundex", soundex(first), soundex(last)), []
        ).append(position)
        name_only.setdefault(
            ("metaphone", metaphone(first), metaphone(last)), []
        ).append(position)
        name_only.setdefault(
            ("signature", letter_signature(first, last)), []
        ).append(position)

    return dated, name_only


def find_fuzzy_duplicates(
    df: pd.DataFrame,
    key: str = "patient_id",
    threshold: float = DEFAULT_THRESHOLD,
    no_dob_threshold: float = NO_DOB_THRESHOLD,
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

    dated_blocks, name_blocks = _build_blocks(dobs, last_keys, firsts, lasts)

    pairs: set[tuple[int, int]] = set()
    oversized = 0

    for members in dated_blocks.values():
        if len(members) < 2:
            continue
        if len(members) > MAX_BLOCK_SIZE:
            oversized += 1
            continue
        pairs.update(combinations(sorted(members), 2))

    # Name-only blocks are used exclusively to rescue records with no date of
    # birth. Two records that both have dates are already covered above, and
    # pairing them on name alone would only add noise.
    no_dob_pairs = 0
    for members in name_blocks.values():
        if len(members) < 2:
            continue
        if len(members) > MAX_BLOCK_SIZE:
            oversized += 1
            continue
        for a, b in combinations(sorted(members), 2):
            if (pd.isna(dobs[a]) or pd.isna(dobs[b])) and (a, b) not in pairs:
                pairs.add((a, b))
                no_dob_pairs += 1

    genders = (df["gender"].tolist() if "gender" in df.columns
               else [None] * len(df))

    matches = []
    vetoed_on_gender = 0
    for a, b in pairs:
        relation = dob_relation(dobs[a], dobs[b])
        if relation is None:
            continue

        score = name_similarity(firsts[a], lasts[a], firsts[b], lasts[b])
        if score < _required_score(relation, threshold, no_dob_threshold):
            continue

        # With no date of birth, the name is the only evidence -- so a known
        # gender conflict is allowed to veto. Never applied when a date agrees,
        # where the date is the stronger signal and a gender mismatch is more
        # likely a data-entry error than two different people.
        if relation == "unknown" and _genders_conflict(genders[a], genders[b]):
            vetoed_on_gender += 1
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

    # A record with neither a date of birth nor a name cannot be matched by any
    # route. Worth surfacing, because a clean-looking zero-match result is
    # meaningless if part of the file was never eligible for comparison.
    no_dob = int(sum(pd.isna(d) for d in dobs))
    no_name = int(sum(1 for f, l in zip(firsts, lasts) if not f and not l))
    unmatchable = int(sum(
        1 for d, f, l in zip(dobs, firsts, lasts) if pd.isna(d) and not f and not l
    ))

    stats = {
        "threshold": threshold,
        "strict_threshold": STRICT_THRESHOLD,
        "no_dob_threshold": no_dob_threshold,
        "vetoed_on_gender_conflict": vetoed_on_gender,
        "blocks_built": len(dated_blocks) + len(name_blocks),
        "pairs_compared": len(pairs),
        "pairs_from_name_only_blocks": no_dob_pairs,
        "candidate_pairs": len(result),
        "records_involved": int(
            pd.concat([result[f"{key}_a"], result[f"{key}_b"]]).nunique()
        ) if not result.empty else 0,
        "by_confidence": (
            result["confidence"].value_counts().to_dict() if not result.empty else {}
        ),
        "records_without_dob": no_dob,
        "records_without_name": no_name,
        "unmatchable_records": unmatchable,
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



