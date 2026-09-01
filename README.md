# Data-Clean-AI

A command-line data cleaning pipeline for messy healthcare extracts, built with pandas.

It takes a raw patient CSV full of the problems real source systems produce — duplicate
registrations, five different date formats, missing values encoded five different ways,
impossible measurements — and produces an analysis-ready dataset plus a data quality
report that says exactly what was changed and why.

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Create the messy source file (400 rows)
python generate_messy_data.py

# 2. Clean it
python clean_data.py --report-json data/clean/quality_report.json
```

Output:

```
data/raw/messy_patients.csv     400 rows, deliberately dirty
data/clean/patients_clean.csv   deduplicated, standardised, validated
data/clean/quality_report.json  machine-readable audit of the run
```

Useful flags:

| Flag | Purpose |
|---|---|
| `--input` / `--output` | Point at different files |
| `--key patient_id` | The business key used to resolve duplicates |
| `--date-convention auto\|dayfirst\|monthfirst` | How to read ambiguous slash dates |
| `--as-of 2026-01-01` | Reference date for age, so results are reproducible |
| `--valid-only` | Write only rows that pass every blocking check |

---

## The problems in the source data

`generate_messy_data.py` builds a 400-row patient extract with the failure modes
deliberately planted, so the cleaner has something real to fix:

- **Duplicates, two kinds.** Byte-identical rows *and* near-duplicates that share a
  `patient_id` but differ in casing, date formatting, and which fields are filled in.
  These need different fixes.
- **Inconsistent dates.** `date_of_birth` appears as `1985-04-12`, `12/04/1985`,
  `12-Apr-1985`, `April 12, 1985`, `12.04.1985`, `19850412`, and `12/04/85`.
- **Missing values with no single sentinel.** Blanks, `N/A`, `null`, `-`, `?`,
  `unknown`, and whitespace-only cells all mean "no value".
- **Inconsistent categories.** `Cardiology` / `cardiology` / `CARDIO` / ` Cardiology`.
- **Numbers stored as formatted text.** `$4,120.50`, `USD 4120.50`, `4,120.50`.
- **Physically impossible values.** Height of `999` cm, weight of `-65` kg, blood
  pressure of `0/0`.

Everything else about the mock data is generated *coherently* — height follows age,
weight follows a plausible BMI for that age, blood pressure rises with age. Drawing those
independently would produce 200 cm toddlers, and the cleaning decisions below would be
untestable against them.

---

## How the cleaner works

The pipeline runs in a fixed order, because each stage depends on the one before it.
That ordering is the part worth understanding.

### 1. Load everything as text

```python
pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
```

Type inference on a dirty file is where silent data loss happens. Let pandas guess and
a `patient_id` of `0012` becomes the integer `12`, and a column with one stray `N/A`
gets a dtype you didn't ask for. Reading as text means **we** decide what every value
means, not the parser.

### 2. Standardise text and null tokens

Trim whitespace, collapse internal runs of spaces, then map the whole vocabulary of
"nothing here" strings to a single real `NA`. Everything downstream — `.isna()`,
`groupby`, joins — depends on missing being one thing, not seven.

### 3. Parse dates *before* deduplicating

Two known formats can be the same date, so dates have to be normalised before you can
compare records.

**Explicit formats, not `dateutil` guessing.** The parser tries a list of known formats
in order and keeps whatever each pass resolves. Letting `pd.to_datetime` infer per row
is far slower on a large frame, and — worse — it can read two rows in the same column
using two different conventions, which corrupts data without raising an error.

**The ambiguity problem.** In `04/12/1985`, is that 4 December or 12 April? You cannot
tell from the value alone. The pipeline infers it from the column: in `25/04/1985` the
`25` can only be a day, so every such row is evidence for the whole column. It counts
that evidence, picks a convention, and applies it consistently — and reports which
convention it chose so the decision is visible rather than buried.

This is not a nitpick. Running the same 400-row file with `--date-convention monthfirst`
instead of the inferred `dayfirst` changes **66 of 364 birth dates** — every one of them
a silent day/month swap that parses cleanly, raises no error, and quietly corrupts every
age, cohort, and trend built on top of it:

```
patient_id   dayfirst     monthfirst
P1002        2025-02-01   2025-01-02
P1004        1938-05-07   1938-07-05
P1010        2005-02-12   2005-12-02
```

> If a column genuinely mixes both conventions, it is unrecoverable — no amount of
> cleaning can tell you what `04/12/1985` meant. That is a data contract problem to
> raise with the source system, not something to quietly guess at. Knowing where
> cleaning stops and escalation starts is the point.

**Two-digit years.** `%y` resolves `55` to **2055**, so a patient born in 1955 lands in
the future. Any birth date after today is therefore a century error, and gets pulled
back 100 years.

### 4. Deduplicate in two passes

**Pass 1 — exact duplicates.** Every field matches, so nothing is lost by keeping one.

**Pass 2 — same patient, different row.** A re-registration, or the same person arriving
from a second source system. Here you have to *choose* a survivor, so the rule is
explicit: **keep the most complete record, breaking ties on the most recent admission.**

This is the bit most people skip. `drop_duplicates()` on its own keeps whichever row
happened to be first in the file — an arbitrary decision dressed up as a default. Naming
a survivorship rule means the result is reproducible and defensible.

### 5. Map categories to a controlled vocabulary

Lower-case, strip, look up in a mapping dict. Anything not in the vocabulary becomes
`NA` **and gets counted in the report** — an unmapped value usually means the source
system added a category nobody told you about, which is exactly the kind of drift you
want to hear about early.

### 6. Clean numerics, then quarantine the impossible

Strip currency symbols and separators, coerce to numeric, then null out anything outside
a plausible clinical range.

**This has to happen before imputation.** A height of `999` cm is not an outlier, it is
a typo — and if you leave it in, it poisons the median you are about to impute with.

### 7. Impute only what can be defended

| Field | Strategy | Reasoning |
|---|---|---|
| `height_cm`, `weight_kg`, `systolic_bp`, `diastolic_bp` | Median by **age band + gender** | Median resists outliers; grouping keeps the value near the right population |
| `total_charges` | Median by department | Cost varies far more by department than overall |
| `insurance_provider` | `"Self-Pay"` | A domain-meaningful default, not a statistical one |
| `gender`, `department`, `smoker` | `"Unknown"` | An honest category beats a guessed one |
| `date_of_birth`, `blood_type`, `diagnosis_code`, names, contact details | **Left null** | Never invent a clinical fact or an identity attribute |
| `discharge_date` | **Left null** | It may not have happened yet — see `still_admitted` |

Three things worth saying out loud:

- **Group by age band, not just gender.** Fill a 6-year-old's missing height with the
  adult median and you get 168 cm — a number that passes every range check and produces
  a nonsense BMI. In the run above, the median height for the `0-12` band is 106 cm
  against 168 cm for adults. Ages with no date of birth get their own `Unknown` band
  rather than being quietly pooled with adults.
- **Order matters.** Categoricals are filled first, because the numeric imputation groups
  by `gender` and `department` — those columns cannot have gaps by the time it runs. For
  the same reason `age` and `age_band` are derived *before* imputation, while `bmi` and
  `length_of_stay_days` are derived *after*, since they depend on imputed values.
- **Every filled cell is logged.** The `imputed_fields` column lists which of that row's
  values were synthesised, so downstream analysis can exclude them. An imputed value that
  looks identical to a measured one is a trap; this makes it traceable.

### 8. Derive the fields analysts actually want

`age`, `age_band`, `bmi`, and `length_of_stay_days` — computed once, here, so every
consumer gets the same definition instead of six slightly different ones in six
dashboards.

### 9. Validate by flagging, not deleting

Every row gets a `dq_flags` column: `dob_missing`, `age_out_of_range`,
`admission_missing`, `discharge_before_admission`, `bp_inverted`, `bmi_implausible`,
`still_admitted`, `no_contact_details`.

Deleting a suspect row hides a source-system problem and quietly biases every number
downstream. Flagging lets an analyst filter on purpose, and gives someone a worklist of
records to fix at source.

Flags are split by severity. **Blocking** failures make a row unfit for analysis and
drive the `is_valid` column. **Informational** flags describe a legitimate state — a
patient who is still admitted is not a broken record — so they never mark a row invalid.

**Cross-field checks catch what column checks cannot.** The range check in step 6 looks at
one column at a time, so it cannot see a combination that is individually fine and jointly
absurd. The sample run flags a 10-year-old recorded at 124 cm and 11.7 kg: both values sit
comfortably inside their own plausible ranges, but together they give a BMI of 7.6. Same
idea behind `bp_inverted` and `discharge_before_admission`.

`bp_inverted` is flagged rather than auto-corrected. A systolic below diastolic is almost
certainly two swapped columns, but silently repairing a clinical value on a guess is not
a call a pipeline should make on its own.

### 10. Report

Every run prints a summary — rows in/out, duplicates removed by type, dates parsed per
column and under which convention, missing before → after per column, values quarantined,
and validation flag counts. `--report-json` writes the same thing machine-readably, which
is what you would assert against in CI or track over time to catch source drift.

One detail that matters: the missing-value before/after counts are both measured **after
deduplication**, so they describe the same set of rows. Comparing the raw file's counts
against the output would credit dropped duplicate rows as though they had been imputed —
a report that flatters itself is worse than no report.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

39 tests covering the two stages where a bug would be invisible in the output:

- **`tests/test_date_parsing.py`** — every source format resolves to the same date;
  mixed formats in one column all parse; garbage becomes `NaT` instead of raising;
  day-first vs month-first inference under each kind of evidence; the documented default
  when a column offers no evidence at all; and two-digit-year century correction,
  including the `12/04/55` → `1955-04-12` round trip.
- **`tests/test_survivorship.py`** — exact duplicates collapse; the most complete record
  wins; completeness beats recency; recency breaks a completeness tie; a record with no
  admission date sorts last rather than winning by accident; the rule degrades gracefully
  when there is no admission date column at all; and the report's row counts reconcile.

The ambiguous-date test is the one worth reading. `04/12/1985` parses cleanly under both
conventions, so no error and no null ever tells you it went wrong — the test asserts that
day-first gives 4 December and month-first gives 12 April, pinning down which reading the
pipeline commits to.

These were checked by mutation: reversing the survivorship sort order fails 6 tests,
flipping the no-evidence default fails 1, and disabling the century correction fails 4.
A test suite that cannot fail is not evidence of anything.

---

## Talking points

Things worth being ready for:

**"Why median and not mean?"** Median resists the outliers you did not catch. Grouping by
gender/department keeps the filled value close to the population it came from rather than
dragging everyone toward one global number.

**"Doesn't imputation bias your analysis?"** Yes, and that is why every imputed cell is
logged in `imputed_fields` and clinical facts are never imputed at all. Imputation is a
convenience for aggregate analysis, not a claim about an individual patient.

**"Why not just drop the bad rows?"** Because missingness is rarely random. Drop everyone
without a discharge date and you have systematically removed the patients who are still
admitted — usually the longest, most expensive stays.

**"How would this scale?"** The transformations are vectorised, and there are no row-wise
`.apply()` calls in the hot path. Past a few million rows I would push the same logic into
SQL/dbt and keep this shape: staging (types and nulls) → deduplication with an explicit
survivorship rule → conformed dimensions → tested marts. The `dq_flags` column is the same
idea as a dbt test, just carried in the data.

**"What would you add next?"** Schema validation with Pandera or Great Expectations, fuzzy
matching on name plus date of birth to catch duplicates where the `patient_id` itself
differs, and tracking the JSON report over time so source-system drift shows up as a trend
rather than a surprise.
