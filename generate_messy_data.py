"""Generate a realistically messy healthcare CSV for the Data-Clean-AI demo.

The output deliberately contains the three problems the cleaner is built to fix:

  1. Missing values, encoded as several different "null" tokens ("", "N/A",
     "unknown", "-", "?") rather than a single consistent sentinel.
  2. Duplicate patients -- both byte-identical rows and "near duplicates" that
     share a patient_id but differ in formatting and completeness.
  3. Inconsistent date formats in date_of_birth (and the admission/discharge
     columns), including two-digit years.

It also throws in the everyday grime you get from real source systems:
inconsistent casing, stray whitespace, currency symbols in numeric columns,
and physically impossible measurements.

Usage:
    python generate_messy_data.py
    python generate_messy_data.py --rows 400 --output data/raw/messy_patients.csv
"""

from __future__ import annotations

import argparse
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

SEED = 42
DEFAULT_ROWS = 400
DUPLICATE_SHARE = 0.09  # ~9% of rows will be duplicates of an earlier patient
# Same person, new patient_id -- the duplicate a key-based rule cannot catch.
REREGISTRATION_SHARE = 0.02

# Tokens a source system might use to mean "no value". A real extract almost
# never uses just one of these, which is why the cleaner normalises them all.
MISSING_TOKENS = ["", "N/A", "n/a", "NA", "null", "NULL", "-", "unknown", "?", "   "]

FIRST_NAMES = [
    "Aisha", "Daniel", "Mei Ling", "Rajesh", "Sofia", "Marcus", "Priya", "Tomas",
    "Grace", "Hakim", "Elena", "Jonas", "Nadia", "Owen", "Farah", "Victor",
    "Amara", "Liam", "Yusuf", "Clara", "Diego", "Hana", "Noah", "Ines",
    "Kwame", "Lucia", "Arjun", "Beatriz", "Sean", "Talia",
]

LAST_NAMES = [
    "Okafor", "Tan", "Kumar", "Silva", "Nguyen", "Rossi", "Haddad", "Bergstrom",
    "Mwangi", "Fischer", "Delgado", "Okonkwo", "Lim", "Petrov", "Costa", "Abadi",
    "Novak", "Chen", "Farouk", "Larsen", "Reyes", "Adeyemi", "Wong", "Mensah",
]

# Canonical value -> the messy spellings that show up in the raw extract.
DEPARTMENT_VARIANTS = {
    "Cardiology": ["Cardiology", "cardiology", "CARDIOLOGY", " Cardiology", "Cardio"],
    "Oncology": ["Oncology", "oncology", "ONCOLOGY", "Onc ", "Onco"],
    "Orthopaedics": ["Orthopaedics", "orthopaedics", "Orthopedics", "Ortho", "ORTHO"],
    "Paediatrics": ["Paediatrics", "paediatrics", "Pediatrics", "Peds", "PAEDS"],
    "Neurology": ["Neurology", "neurology", "NEURO", "Neuro ", "neuro"],
    "Emergency": ["Emergency", "emergency", "ER", "A&E", "EMERGENCY"],
    "General Medicine": ["General Medicine", "general medicine", "Gen Med", "GENMED"],
}

GENDER_VARIANTS = {
    "M": ["M", "m", "Male", "MALE", "male", "M "],
    "F": ["F", "f", "Female", "FEMALE", "female", " F"],
}

SMOKER_VARIANTS = {
    "Y": ["Yes", "Y", "yes", "1", "TRUE", "True", "y"],
    "N": ["No", "N", "no", "0", "FALSE", "False", "n"],
}

BLOOD_TYPES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

INSURERS = [
    "Great Eastern", "AIA", "Allianz", "Prudential", "Etiqa", "Medicare",
    "BlueCross", "Aetna",
]

# ICD-10-style codes, mixed case on purpose.
DIAGNOSIS_CODES = [
    "I10", "E11.9", "J45.909", "M54.5", "K21.9", "N39.0", "F32.9", "I25.10",
    "J44.9", "E78.5", "R07.9", "S72.001A", "G43.909", "L40.0", "A09",
]

# Every one of these renders the SAME date differently. The cleaner has to
# collapse them back to one ISO format.
DOB_FORMATS = [
    "%Y-%m-%d",     # 1985-04-12  (ISO, the target)
    "%d/%m/%Y",     # 12/04/1985  (day-first slash)
    "%d-%b-%Y",     # 12-Apr-1985
    "%B %d, %Y",    # April 12, 1985
    "%d.%m.%Y",     # 12.04.1985
    "%Y%m%d",       # 19850412    (compact, no separators)
    "%d/%m/%y",     # 12/04/85    (two-digit year -- century is ambiguous)
]

ADMISSION_FORMATS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%Y-%m-%d %H:%M:%S",
    "%d-%b-%Y",
]


def maybe_missing(value: str, rate: float, rng: random.Random) -> str:
    """Replace `value` with a random null-token `rate` of the time."""
    if rng.random() < rate:
        return rng.choice(MISSING_TOKENS)
    return value


def random_dob(rng: random.Random) -> date:
    """A birth date between roughly 1 and 95 years ago."""
    days_old = rng.randint(365, 95 * 365)
    return date.today() - timedelta(days=days_old)


def format_phone(digits: str, rng: random.Random) -> str:
    style = rng.randint(0, 3)
    if style == 0:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    if style == 1:
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    if style == 2:
        return f"{digits[:3]}.{digits[3:6]}.{digits[6:]}"
    return digits


def format_charges(amount: float, rng: random.Random) -> str:
    style = rng.randint(0, 3)
    if style == 0:
        return f"${amount:,.2f}"
    if style == 1:
        return f"{amount:,.2f}"
    if style == 2:
        return f"USD {amount:.2f}"
    return f"{amount:.2f}"


def build_patient(patient_id: str, rng: random.Random) -> dict:
    """One patient record, already messed up."""
    canonical_gender = rng.choice(["M", "F"])
    canonical_dept = rng.choice(list(DEPARTMENT_VARIANTS))
    canonical_smoker = rng.choices(["Y", "N"], weights=[0.25, 0.75])[0]

    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    dob = random_dob(rng)

    admission = date.today() - timedelta(days=rng.randint(1, 900))
    stay_days = rng.choices([0, 1, 2, 3, 5, 8, 14, 30], weights=[10, 20, 20, 15, 15, 10, 7, 3])[0]
    discharge = admission + timedelta(days=stay_days)

    # Physical measurements are generated coherently rather than independently:
    # height follows age, and weight is derived from a plausible BMI for that age.
    # Drawing them independently would produce 200 cm toddlers.
    age_years = (date.today() - dob).days / 365.25
    if age_years < 18:
        height = round(min(50 + age_years * 7 + rng.gauss(0, 4), 178), 1)
        bmi = rng.gauss(17.5, 3)
    else:
        height = round(rng.gauss(168, 11), 1)
        bmi = rng.gauss(25.5, 5)
    weight = round(max(2.5, bmi * (height / 100) ** 2), 1)

    # Blood pressure also rises with age.
    systolic = int(rng.gauss(100 + min(age_years, 80) * 0.45, 14))
    diastolic = int(rng.gauss(62 + min(age_years, 80) * 0.25, 9))
    if rng.random() < 0.02:
        height = rng.choice([0, 999, -170])
    if rng.random() < 0.02:
        weight = rng.choice([0, -65, 700])
    if rng.random() < 0.02:
        systolic, diastolic = rng.choice([(0, 0), (60, 190)])  # second one is inverted

    email_local = f"{first.split()[0]}.{last}".lower().replace(" ", "")
    email = rng.choice([
        f"{email_local}@example.com",
        f"{email_local.upper()}@EXAMPLE.COM",
        f" {email_local}@example.com ",
    ])

    charges = max(50.0, rng.gauss(4200, 2600))

    blood = rng.choice(BLOOD_TYPES)
    blood = rng.choice([blood, blood.lower(), blood.replace("+", " +")])

    code = rng.choice(DIAGNOSIS_CODES)
    code = rng.choice([code, code.lower(), f" {code}"])

    return {
        "patient_id": patient_id,
        "first_name": maybe_missing(first, 0.02, rng),
        "last_name": maybe_missing(last, 0.02, rng),
        "gender": maybe_missing(rng.choice(GENDER_VARIANTS[canonical_gender]), 0.06, rng),
        "date_of_birth": maybe_missing(dob.strftime(rng.choice(DOB_FORMATS)), 0.05, rng),
        "blood_type": maybe_missing(blood, 0.12, rng),
        "department": maybe_missing(rng.choice(DEPARTMENT_VARIANTS[canonical_dept]), 0.04, rng),
        "admission_date": maybe_missing(
            admission.strftime(rng.choice(ADMISSION_FORMATS)), 0.03, rng
        ),
        # Left blank more often -- some patients are genuinely still admitted.
        "discharge_date": maybe_missing(
            discharge.strftime(rng.choice(ADMISSION_FORMATS)), 0.10, rng
        ),
        "diagnosis_code": maybe_missing(code, 0.05, rng),
        "height_cm": maybe_missing(str(height), 0.09, rng),
        "weight_kg": maybe_missing(str(weight), 0.11, rng),
        "systolic_bp": maybe_missing(str(systolic), 0.08, rng),
        "diastolic_bp": maybe_missing(str(diastolic), 0.08, rng),
        "smoker": maybe_missing(rng.choice(SMOKER_VARIANTS[canonical_smoker]), 0.14, rng),
        "insurance_provider": maybe_missing(rng.choice(INSURERS), 0.15, rng),
        "total_charges": maybe_missing(format_charges(charges, rng), 0.07, rng),
        "email": maybe_missing(email, 0.10, rng),
        "phone": maybe_missing(
            format_phone("".join(str(rng.randint(0, 9)) for _ in range(10)), rng), 0.12, rng
        ),
    }


def parse_any_dob(value: str):
    """Read back a date of birth written in any of the formats above."""
    if value in MISSING_TOKENS:
        return None
    for fmt in DOB_FORMATS:
        try:
            return pd.to_datetime(value, format=fmt)
        except (ValueError, TypeError):
            continue
    return None


def introduce_typo(text: str, rng: random.Random) -> str:
    """One realistic keyboard slip: a transposition, a drop, or a doubled letter."""
    if len(text) < 3:
        return text
    position = rng.randrange(len(text) - 1)
    style = rng.randint(0, 2)
    if style == 0:  # swap two adjacent characters
        return text[:position] + text[position + 1] + text[position] + text[position + 2:]
    if style == 1:  # drop a character
        return text[:position] + text[position + 1:]
    return text[:position] + text[position] + text[position:]  # double a character


def make_near_duplicate(row: dict, rng: random.Random) -> dict:
    """A second row for the same patient_id that is NOT byte-identical.

    This is what a re-submitted registration form or a second source system
    looks like: same person, different formatting, different gaps. Exact-match
    deduplication will not catch it -- you need a key-based rule.
    """
    dup = dict(row)

    # Re-render the date of birth in a different format, if it is present.
    parsed = parse_any_dob(dup["date_of_birth"])
    if parsed is not None:
        dup["date_of_birth"] = parsed.strftime(rng.choice(DOB_FORMATS))

    # Different casing / whitespace on the free-text fields.
    if dup["last_name"] not in MISSING_TOKENS:
        dup["last_name"] = rng.choice([
            dup["last_name"].upper(), dup["last_name"].lower(), f" {dup['last_name']} "
        ])

    # Blank out one or two extra fields, so this copy is less complete than the
    # original. The cleaner's survivorship rule should therefore keep the original.
    for col in rng.sample(
        ["weight_kg", "insurance_provider", "phone", "email", "blood_type"], k=rng.randint(1, 3)
    ):
        dup[col] = rng.choice(MISSING_TOKENS)

    return dup


def make_reregistration(row: dict, new_id: str, rng: random.Random) -> dict:
    """The same human, registered again under a BRAND NEW patient_id.

    This is the duplicate that key-based deduplication can never catch, because
    the key itself differs. It happens when someone re-registers at a different
    site, or arrives via a second source system that mints its own IDs.

    The name picks up a typo and the date of birth is sometimes day/month
    transposed -- so matching has to tolerate both without becoming so loose
    that it merges two genuinely different patients.
    """
    dup = dict(row)
    dup["patient_id"] = new_id

    for name_field in ("first_name", "last_name"):
        if dup[name_field] not in MISSING_TOKENS and rng.random() < 0.6:
            dup[name_field] = introduce_typo(dup[name_field], rng)

    parsed = parse_any_dob(dup["date_of_birth"])
    if parsed is not None:
        # A third of the time the day and month get swapped, which is exactly
        # what the date-convention ambiguity produces upstream.
        if rng.random() < 0.33 and parsed.day <= 12:
            parsed = parsed.replace(month=parsed.day, day=parsed.month)
        dup["date_of_birth"] = parsed.strftime(rng.choice(DOB_FORMATS))

    # A fresh registration means fresh contact details and a new episode of care.
    dup["email"] = rng.choice(MISSING_TOKENS)
    dup["phone"] = format_phone(
        "".join(str(rng.randint(0, 9)) for _ in range(10)), rng
    )
    admission = date.today() - timedelta(days=rng.randint(1, 400))
    dup["admission_date"] = admission.strftime(rng.choice(ADMISSION_FORMATS))
    dup["discharge_date"] = (
        admission + timedelta(days=rng.randint(0, 10))
    ).strftime(rng.choice(ADMISSION_FORMATS))
    dup["department"] = rng.choice(DEPARTMENT_VARIANTS[rng.choice(list(DEPARTMENT_VARIANTS))])

    return dup


def generate(n_rows: int, seed: int = SEED) -> pd.DataFrame:
    rng = random.Random(seed)

    n_duplicates = int(n_rows * DUPLICATE_SHARE)
    n_reregistrations = int(n_rows * REREGISTRATION_SHARE)
    n_unique = n_rows - n_duplicates - n_reregistrations

    rows = [build_patient(f"P{1000 + i}", rng) for i in range(n_unique)]

    # Roughly half the duplicates are byte-identical copies, half are near
    # duplicates. Both kinds occur in real extracts and they need different fixes.
    for _ in range(n_duplicates):
        source = rng.choice(rows[:n_unique])
        rows.append(dict(source) if rng.random() < 0.5 else make_near_duplicate(source, rng))

    # Re-registrations get their own IDs, continuing the sequence so they look
    # like ordinary new patients. Only name and date of birth betray them.
    for offset in range(n_reregistrations):
        source = rng.choice(rows[:n_unique])
        rows.append(make_reregistration(source, f"P{1000 + n_unique + offset}", rng))

    rng.shuffle(rows)  # duplicates should not all sit at the bottom of the file
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a messy healthcare CSV.")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="Total rows to write.")
    parser.add_argument(
        "--output", default="data/raw/messy_patients.csv", help="Destination CSV path."
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility.")
    args = parser.parse_args()

    df = generate(args.rows, args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} rows x {len(df.columns)} columns -> {out_path}")
    print(f"  unique patient_ids : {df['patient_id'].nunique()}")
    print(f"  duplicate rows     : {len(df) - df['patient_id'].nunique()}")


if __name__ == "__main__":
    main()
