"""Data-Clean-AI -- a command-line cleaning pipeline for messy healthcare extracts.

The pipeline runs in a fixed order, because each stage depends on the one before it:

    1. load          read everything as text, so pandas cannot silently guess a dtype
    2. standardise   trim whitespace, collapse the many "null" tokens into one real NA
    3. dates         parse every date format found into a single ISO date
    4. duplicates    drop exact copies, then resolve same-patient records by survivorship
    5. categories    map casing/abbreviation variants back to a controlled vocabulary
    6. numerics      strip currency formatting, quarantine physically impossible values
    7. impute        fill the gaps that CAN be filled defensibly, and record which ones
    8. derive        add age, BMI and length of stay
    9. validate      flag rows that are still suspect instead of deleting them
   10. report        print (and optionally save) a data-quality summary

Usage:
    python clean_data.py --input data/raw/messy_patients.csv \
                         --output data/clean/patients_clean.csv \
                         --report-json data/clean/quality_report.json
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Everything a source system might use to mean "no value". Read as literal text
# and normalised here, so we control the definition of missing rather than
# inheriting whatever pandas guesses.
NULL_TOKENS = {
    "", "na", "n/a", "n.a.", "null", "none", "nil", "-", "--", "?", "unknown",
    "unk", "not available", "not applicable", "#n/a", "nan",
}

# Date formats whose meaning cannot be misread.
UNAMBIGUOUS_DATE_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%b-%Y",
    "%d %b %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%d-%B-%Y",
    "%Y%m%d",
]

# Formats where 04/12 could be 4 December or 12 April. Which list we try first
# is decided by `infer_day_first` below.
DAY_FIRST_FORMATS = ["%d/%m/%Y %H:%M:%S", "%d/%m/%Y", "%d.%m.%Y", "%d/%m/%y", "%d.%m.%y"]
MONTH_FIRST_FORMATS = ["%m/%d/%Y %H:%M:%S", "%m/%d/%Y", "%m.%d.%Y", "%m/%d/%y", "%m.%d.%y"]

DATE_COLUMNS = ["date_of_birth", "admission_date", "discharge_date"]

# Messy spelling -> canonical value. Keys are compared lower-cased and stripped.
DEPARTMENT_MAP = {
    "cardiology": "Cardiology", "cardio": "Cardiology",
    "oncology": "Oncology", "onc": "Oncology", "onco": "Oncology",
    "orthopaedics": "Orthopaedics", "orthopedics": "Orthopaedics",
    "ortho": "Orthopaedics",
    "paediatrics": "Paediatrics", "pediatrics": "Paediatrics",
    "peds": "Paediatrics", "paeds": "Paediatrics",
    "neurology": "Neurology", "neuro": "Neurology",
    "emergency": "Emergency", "er": "Emergency", "a&e": "Emergency",
    "general medicine": "General Medicine", "gen med": "General Medicine",
    "genmed": "General Medicine",
}

GENDER_MAP = {
    "m": "M", "male": "M", "man": "M",
    "f": "F", "female": "F", "woman": "F",
}

SMOKER_MAP = {
    "y": "Y", "yes": "Y", "true": "Y", "1": "Y", "smoker": "Y",
    "n": "N", "no": "N", "false": "N", "0": "N", "non-smoker": "N",
}

VALID_BLOOD_TYPES = {"A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"}

# Values outside these ranges are not "outliers" -- they are impossible, so they
# are treated as data-entry errors and quarantined BEFORE any median is taken.
PLAUSIBLE_RANGES = {
    "height_cm": (30, 250),
    "weight_kg": (1, 400),
    "systolic_bp": (60, 260),
    "diastolic_bp": (30, 160),
    "total_charges": (0, 1_000_000),
}

NUMERIC_COLUMNS = list(PLAUSIBLE_RANGES)

# Numeric columns filled with a group median, and the columns defining the group.
# Body measurements are grouped by age band as well as gender: filling a child's
# height with an adult median produces a plausible-looking number and a nonsense BMI.
MEDIAN_IMPUTE_BY = {
    "height_cm": ["age_band", "gender"],
    "weight_kg": ["age_band", "gender"],
    "systolic_bp": ["age_band", "gender"],
    "diastolic_bp": ["age_band", "gender"],
    "total_charges": ["department"],
}

# Upper bound (exclusive) -> label. Ages above the last bound fall into AGE_BAND_TOP.
AGE_BANDS = [(13, "0-12"), (18, "13-17"), (40, "18-39"), (65, "40-64")]
AGE_BAND_TOP = "65+"

# Categorical columns filled with a fixed, meaningful default.
CONSTANT_IMPUTE = {
    "gender": "Unknown",
    "department": "Unknown",
    "insurance_provider": "Self-Pay",
    "smoker": "Unknown",
}

# Deliberately NOT imputed. See the README -- we do not invent clinical facts,
# identity attributes, or events that may not have happened yet.
NEVER_IMPUTED = ["date_of_birth", "blood_type", "discharge_date", "diagnosis_code",
                 "first_name", "last_name", "email", "phone"]


# --------------------------------------------------------------------------
# 1. Load
# --------------------------------------------------------------------------

def load_raw(path: Path) -> pd.DataFrame:
    """Read the CSV as pure text.

    `dtype=str` plus `keep_default_na=False` stops pandas from inferring types or
    deciding what counts as missing. Inference on a dirty file is where silent
    data loss happens -- a column with one "N/A" becomes object, a zip code
    becomes an int and loses its leading zero. We decide instead.
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


# --------------------------------------------------------------------------
# 2. Standardise text and null tokens
# --------------------------------------------------------------------------

def standardize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace, collapse internal runs of spaces, unify the null tokens."""
    for col in df.columns:
        cleaned = (
            df[col]
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)  # "Gen   Med" -> "Gen Med"
            .str.strip()
        )
        # Compare case-insensitively against the null vocabulary.
        is_null = cleaned.str.lower().isin(NULL_TOKENS)
        df[col] = cleaned.mask(is_null, pd.NA)
    return df


# --------------------------------------------------------------------------
# 3. Dates
# --------------------------------------------------------------------------

def infer_day_first(series: pd.Series) -> tuple[bool, dict]:
    """Work out whether slash dates in this column are day-first or month-first.

    We look only at the values that resolve the ambiguity by themselves:
    in "25/04/1985" the 25 can only be a day, and in "04/25/1985" the 25 can only
    be a day in second position. Counting those gives evidence for the whole
    column, which we then apply consistently.
    """
    present = series.dropna().astype(str)
    parts = present.str.extract(r"^(\d{1,2})[/.](\d{1,2})[/.](\d{2,4})$")
    if parts.empty:
        return True, {"day_first_evidence": 0, "month_first_evidence": 0}

    first = pd.to_numeric(parts[0], errors="coerce")
    second = pd.to_numeric(parts[1], errors="coerce")

    day_first_evidence = int((first > 12).sum())    # first slot must be a day
    month_first_evidence = int((second > 12).sum())  # second slot must be a day

    evidence = {
        "day_first_evidence": day_first_evidence,
        "month_first_evidence": month_first_evidence,
    }
    # Tie or no evidence -> default to day-first (ISO-adjacent, and the more
    # common convention outside the US). This is a documented assumption, not a
    # guess buried in the code.
    return day_first_evidence >= month_first_evidence, evidence


def parse_date_series(series: pd.Series, day_first: bool) -> pd.Series:
    """Try each known format in turn, keeping whatever each pass manages to parse.

    Explicit formats are used rather than letting dateutil guess per row: guessing
    is ~100x slower on a big frame and, worse, it can read two rows in the same
    column with two different conventions.
    """
    formats = list(UNAMBIGUOUS_DATE_FORMATS)
    formats += DAY_FIRST_FORMATS + MONTH_FIRST_FORMATS if day_first else \
               MONTH_FIRST_FORMATS + DAY_FIRST_FORMATS

    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    unparsed = series.notna()

    for fmt in formats:
        if not unparsed.any():
            break
        attempt = pd.to_datetime(series[unparsed], format=fmt, errors="coerce")
        parsed = attempt.dropna()
        if parsed.empty:
            continue
        result.loc[parsed.index] = parsed
        unparsed.loc[parsed.index] = False

    return result


def fix_two_digit_century(dates: pd.Series, as_of: pd.Timestamp) -> tuple[pd.Series, int]:
    """Pull impossible future birth dates back a century.

    "%y" resolves 55 to 2055, so a patient born in 1955 lands in the future.
    Any birth date after today is therefore a century error, not a real date.
    """
    future = dates.notna() & (dates > as_of)
    corrected = int(future.sum())
    if corrected:
        dates = dates.mask(future, dates - pd.DateOffset(years=100))
    return dates, corrected


def standardize_dates(df: pd.DataFrame, convention: str, as_of: pd.Timestamp,
                      report: dict) -> pd.DataFrame:
    for col in DATE_COLUMNS:
        if col not in df.columns:
            continue

        raw_present = int(df[col].notna().sum())

        if convention == "auto":
            day_first, evidence = infer_day_first(df[col])
        else:
            day_first = convention == "dayfirst"
            evidence = {"day_first_evidence": None, "month_first_evidence": None}

        parsed = parse_date_series(df[col], day_first)

        century_fixes = 0
        if col == "date_of_birth":
            parsed, century_fixes = fix_two_digit_century(parsed, as_of)

        unparseable = raw_present - int(parsed.notna().sum())
        df[col] = parsed.dt.normalize()  # every date column is at day grain

        report["dates"][col] = {
            "values_present": raw_present,
            "parsed": int(parsed.notna().sum()),
            "unparseable": unparseable,
            "convention_used": "day-first" if day_first else "month-first",
            "century_corrections": century_fixes,
            **evidence,
        }
    return df


# --------------------------------------------------------------------------
# 4. Duplicates
# --------------------------------------------------------------------------

def drop_duplicates(df: pd.DataFrame, key: str, report: dict) -> pd.DataFrame:
    """Two passes, because there are two different kinds of duplicate.

    Pass 1 -- exact copies. Every field matches, so nothing is lost by keeping one.

    Pass 2 -- same patient, different row. A re-registration or a second source
    system. Here we must choose a survivor, so we apply an explicit rule:
    keep the most complete record, breaking ties on the most recent admission.
    "Keep the first row" would be an arbitrary decision dressed up as a default.
    """
    before = len(df)
    df = df.drop_duplicates()
    exact_removed = before - len(df)

    df = df.assign(_completeness=df.notna().sum(axis=1))
    sort_by = ["_completeness"]
    if "admission_date" in df.columns:
        sort_by.append("admission_date")
    df = df.sort_values(
        by=sort_by,
        ascending=[False] * len(sort_by),
        na_position="last",
        kind="mergesort",  # stable: equal rows keep their original file order
    )
    before_key = len(df)
    df = df.drop_duplicates(subset=[key], keep="first")
    key_removed = before_key - len(df)

    df = df.drop(columns="_completeness").sort_values(key).reset_index(drop=True)

    report["duplicates"] = {
        "rows_in": before,
        "exact_duplicate_rows_removed": exact_removed,
        "same_patient_rows_removed": key_removed,
        "rows_out": len(df),
    }
    return df


# --------------------------------------------------------------------------
# 5. Categorical standardisation
# --------------------------------------------------------------------------

def map_to_vocabulary(series: pd.Series, mapping: dict) -> pd.Series:
    """Lower-case, strip, then look up. Anything unrecognised becomes NA."""
    lookup = series.str.lower().str.strip()
    return lookup.map(mapping).astype("object")


def standardize_categories(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    stats = {}

    for col, mapping in [("gender", GENDER_MAP), ("department", DEPARTMENT_MAP),
                         ("smoker", SMOKER_MAP)]:
        if col not in df.columns:
            continue
        before_distinct = int(df[col].nunique(dropna=True))
        mapped = map_to_vocabulary(df[col], mapping)
        # Values present in the raw data but not in our vocabulary -- worth
        # surfacing, because it usually means the source added a new category.
        unmapped = int((df[col].notna() & mapped.isna()).sum())
        df[col] = mapped
        stats[col] = {
            "distinct_before": before_distinct,
            "distinct_after": int(mapped.nunique(dropna=True)),
            "unmapped_values": unmapped,
        }

    if "blood_type" in df.columns:
        cleaned = df["blood_type"].str.upper().str.replace(r"\s+", "", regex=True)
        invalid = int((cleaned.notna() & ~cleaned.isin(VALID_BLOOD_TYPES)).sum())
        df["blood_type"] = cleaned.where(cleaned.isin(VALID_BLOOD_TYPES), pd.NA)
        stats["blood_type"] = {"invalid_values_nulled": invalid}

    if "diagnosis_code" in df.columns:
        df["diagnosis_code"] = df["diagnosis_code"].str.upper().str.strip()

    for col in ["first_name", "last_name"]:
        if col in df.columns:
            df[col] = df[col].str.title()

    if "email" in df.columns:
        df["email"] = df["email"].str.lower().str.strip()

    if "phone" in df.columns:
        # Keep digits only, then re-format. Storing a canonical form means a join
        # on phone number actually works.
        digits = df["phone"].str.replace(r"\D", "", regex=True)
        df["phone"] = digits.where(digits.str.len() == 10, pd.NA)
        df["phone"] = df["phone"].str.replace(r"^(\d{3})(\d{3})(\d{4})$", r"\1-\2-\3", regex=True)

    report["categories"] = stats
    return df


# --------------------------------------------------------------------------
# 6. Numerics
# --------------------------------------------------------------------------

CURRENCY_NOISE = re.compile(r"[^0-9.\-]")


def standardize_numerics(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    stats = {}
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        was_missing = df[col].isna()

        # "$4,120.50" and "USD 4120.50" both become 4120.50.
        stripped = df[col].str.replace(CURRENCY_NOISE, "", regex=True)
        numeric = pd.to_numeric(stripped, errors="coerce")
        # Present in the file but not a number at all, e.g. "twelve".
        unparseable = int((~was_missing & numeric.isna()).sum())

        low, high = PLAUSIBLE_RANGES[col]
        implausible = numeric.notna() & ((numeric < low) | (numeric > high))

        df[col] = numeric.mask(implausible, np.nan)
        stats[col] = {
            "non_numeric_coerced": unparseable,
            "implausible_quarantined": int(implausible.sum()),
            "plausible_range": [low, high],
        }
    report["numerics"] = stats
    return df


# --------------------------------------------------------------------------
# 7. Imputation
# --------------------------------------------------------------------------

def impute(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Fill only what can be filled defensibly, and leave a paper trail.

    Group medians are used rather than a single global mean: median resists the
    outliers we could not catch, and grouping by gender/department keeps the
    filled value close to the population it came from. Every filled cell is
    recorded in `imputed_fields` so downstream analysis can exclude them.
    """
    stats = {}
    # One list per row, recording which of that row's fields we filled in.
    imputation_log = [[] for _ in range(len(df))]

    def log_filled(column: str, mask: pd.Series) -> None:
        for position in np.flatnonzero(mask.to_numpy()):
            imputation_log[position].append(column)

    # Categoricals go first: the numeric imputation below groups by gender and
    # department, so those columns must have no gaps by the time we get there.
    for col, default in CONSTANT_IMPUTE.items():
        if col not in df.columns:
            continue
        missing = df[col].isna()
        if missing.any():
            df.loc[missing, col] = default
            log_filled(col, missing)
        stats[col] = {"strategy": f"constant '{default}'", "filled": int(missing.sum())}

    for col, group_cols in MEDIAN_IMPUTE_BY.items():
        if col not in df.columns:
            continue
        group_cols = [c for c in group_cols if c in df.columns]
        missing = df[col].isna()
        if missing.any() and group_cols:
            group_median = df.groupby(group_cols, observed=True)[col].transform("median")
            df[col] = df[col].fillna(group_median)
            # A group can be empty for this column; fall back to the column median.
            df[col] = df[col].fillna(df[col].median()).round(1)
            log_filled(col, missing)
        stats[col] = {
            "strategy": f"median by {' + '.join(group_cols)} (global median fallback)",
            "filled": int(missing.sum()),
        }

    for col in NEVER_IMPUTED:
        if col in df.columns:
            stats[col] = {
                "strategy": "left null by design",
                "still_missing": int(df[col].isna().sum()),
            }

    df["imputed_fields"] = [";".join(fields) if fields else pd.NA for fields in imputation_log]
    report["imputation"] = stats
    return df


# --------------------------------------------------------------------------
# 8. Derived columns
# --------------------------------------------------------------------------

def add_age(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Age and age band. Computed before imputation, because the body-measurement
    imputation groups by age band."""
    if "date_of_birth" not in df.columns:
        return df

    age_days = (as_of - df["date_of_birth"]).dt.days
    df["age"] = np.floor(age_days / 365.25).astype("Float64").astype("Int64")

    band = pd.Series(AGE_BAND_TOP, index=df.index, dtype="object")
    for upper, label in reversed(AGE_BANDS):
        band = band.mask(df["age"].notna() & (df["age"] < upper), label)
    # Age unknown is its own band, so those rows are not silently pooled with adults.
    df["age_band"] = band.mask(df["age"].isna(), "Unknown")
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Measures that depend on imputed values, so they run after imputation."""
    if {"height_cm", "weight_kg"} <= set(df.columns):
        height_m = df["height_cm"] / 100
        df["bmi"] = (df["weight_kg"] / height_m.pow(2)).round(1)

    if {"admission_date", "discharge_date"} <= set(df.columns):
        df["length_of_stay_days"] = (
            (df["discharge_date"] - df["admission_date"]).dt.days.astype("Float64").astype("Int64")
        )

    return df


# --------------------------------------------------------------------------
# 9. Validation
# --------------------------------------------------------------------------

# Blocking failures make a row unfit for analysis. Informational flags describe a
# real, legitimate state that an analyst should simply be aware of -- a patient
# who is still admitted is not a broken record.
BLOCKING_CHECKS = {
    "dob_missing", "age_out_of_range", "admission_missing",
    "discharge_before_admission", "bp_inverted", "bmi_implausible",
}


def validate(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Flag rows rather than delete them.

    Deleting a suspect row hides a source-system problem and quietly biases every
    downstream number. A flag column lets an analyst filter on purpose, and gives
    someone the list of records to go fix at source.
    """
    checks = {
        "dob_missing": df["date_of_birth"].isna(),
        "age_out_of_range": df["age"].notna() & ((df["age"] < 0) | (df["age"] > 120)),
        "admission_missing": df["admission_date"].isna(),
        "discharge_before_admission": (
            df["length_of_stay_days"].notna() & (df["length_of_stay_days"] < 0)
        ),
        "still_admitted": df["admission_date"].notna() & df["discharge_date"].isna(),
        "bp_inverted": (
            df["systolic_bp"].notna() & df["diastolic_bp"].notna()
            & (df["systolic_bp"] <= df["diastolic_bp"])
        ),
        # Catches height/weight pairs that are each individually plausible but
        # nonsensical together -- a check on a single column cannot see this.
        "bmi_implausible": df["bmi"].notna() & ((df["bmi"] < 10) | (df["bmi"] > 60)),
        "no_contact_details": df["email"].isna() & df["phone"].isna(),
    }
    checks = {name: mask.fillna(False).astype(bool) for name, mask in checks.items()}

    flags = [[] for _ in range(len(df))]
    for name, mask in checks.items():
        for position in np.flatnonzero(mask.to_numpy()):
            flags[position].append(name)

    df["dq_flags"] = [";".join(names) if names else pd.NA for names in flags]

    blocking = np.zeros(len(df), dtype=bool)
    for name in BLOCKING_CHECKS:
        blocking |= checks[name].to_numpy()
    df["is_valid"] = ~blocking

    report["validation"] = {name: int(mask.sum()) for name, mask in checks.items()}
    report["validation"]["rows_passing_blocking_checks"] = int(df["is_valid"].sum())
    return df


# --------------------------------------------------------------------------
# 10. Report
# --------------------------------------------------------------------------

def print_report(report: dict) -> None:
    def header(text: str) -> None:
        print(f"\n{text}\n" + "-" * len(text))

    print("=" * 62)
    print("  DATA-CLEAN-AI  ::  data quality report")
    print("=" * 62)

    header("Volume")
    v = report["volume"]
    print(f"  rows in                 : {v['rows_in']}")
    print(f"  rows out                : {v['rows_out']}")
    print(f"  columns in / out        : {v['columns_in']} / {v['columns_out']}")

    header("Duplicates")
    d = report["duplicates"]
    print(f"  exact duplicate rows    : {d['exact_duplicate_rows_removed']}")
    print(f"  same-patient rows       : {d['same_patient_rows_removed']}  (survivorship applied)")

    header("Dates")
    for col, s in report["dates"].items():
        print(f"  {col:<16} parsed {s['parsed']}/{s['values_present']}"
              f"  unparseable={s['unparseable']}"
              f"  convention={s['convention_used']}"
              f"  century_fixes={s['century_corrections']}")

    header("Missing values, after deduplication (before -> after)")
    imputed_cols = {
        col for col, s in report["imputation"].items() if s.get("filled", 0) > 0
    }
    for col, before in report["missing_before"].items():
        after = report["missing_after"][col]
        if col in imputed_cols:
            note = "  <- imputed"
        elif after > before:
            note = "  <- values failed validation and were nulled"
        else:
            note = ""
        print(f"  {col:<20} {before:>4} -> {after:<4}{note}")

    header("Numeric quarantine")
    for col, s in report["numerics"].items():
        print(f"  {col:<16} implausible values nulled: {s['implausible_quarantined']}"
              f"   (kept range {s['plausible_range']})")

    header("Validation flags")
    for name, count in report["validation"].items():
        print(f"  {name:<28} {count}")

    print()


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def clean(df: pd.DataFrame, convention: str, as_of: pd.Timestamp,
          key: str = "patient_id") -> tuple[pd.DataFrame, dict]:
    report: dict = {"dates": {}, "generated_at": datetime.now().isoformat(timespec="seconds")}
    report["volume"] = {"rows_in": len(df), "columns_in": len(df.columns)}

    df = standardize_text(df)
    report["missing_in_source"] = {c: int(df[c].isna().sum()) for c in df.columns}

    df = standardize_dates(df, convention, as_of, report)
    df = drop_duplicates(df, key, report)

    # Measured after deduplication so that before/after compare the same set of
    # rows. Comparing against the source counts would credit dropped duplicates
    # as if they had been imputed.
    report["missing_before"] = {c: int(df[c].isna().sum()) for c in df.columns}

    df = standardize_categories(df, report)
    df = standardize_numerics(df, report)
    df = add_age(df, as_of)
    df = impute(df, report)
    df = add_derived_columns(df)
    df = validate(df, report)

    report["missing_after"] = {
        c: int(df[c].isna().sum()) for c in report["missing_before"] if c in df.columns
    }
    report["volume"]["rows_out"] = len(df)
    report["volume"]["columns_out"] = len(df.columns)
    return df, report


def write_output(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    # Serialise dates as plain ISO strings so the CSV is unambiguous to any reader.
    for col in DATE_COLUMNS:
        if col in out.columns:
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean a messy healthcare CSV into an analysis-ready dataset."
    )
    parser.add_argument("--input", "-i", default="data/raw/messy_patients.csv")
    parser.add_argument("--output", "-o", default="data/clean/patients_clean.csv")
    parser.add_argument("--report-json", default=None,
                        help="Optional path to save the quality report as JSON.")
    parser.add_argument("--key", default="patient_id",
                        help="Business key used to resolve duplicate records.")
    parser.add_argument("--date-convention", choices=["auto", "dayfirst", "monthfirst"],
                        default="auto",
                        help="How to read ambiguous slash dates. 'auto' infers it from the data.")
    parser.add_argument("--as-of", default=None,
                        help="Reference date (YYYY-MM-DD) for age. Defaults to today.")
    parser.add_argument("--valid-only", action="store_true",
                        help="Write only rows that passed every blocking validation check.")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}\nRun generate_messy_data.py first.")

    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.today().normalize()

    raw = load_raw(in_path)
    cleaned, report = clean(raw, args.date_convention, as_of, args.key)

    if args.valid_only:
        kept = cleaned[cleaned["is_valid"]]
        report["volume"]["rows_written"] = len(kept)
        cleaned_out = kept
    else:
        cleaned_out = cleaned

    write_output(cleaned_out, Path(args.output))
    print_report(report)
    print(f"Clean dataset written to: {args.output}  ({len(cleaned_out)} rows)")

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Quality report written to: {report_path}")


if __name__ == "__main__":
    main()
