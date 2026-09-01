# Data-Clean-AI

[![tests](https://github.com/domdom1991/data-clean-ai/actions/workflows/tests.yml/badge.svg)](https://github.com/domdom1991/data-clean-ai/actions/workflows/tests.yml)

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
| `--fuzzy-threshold 0.90` | Name similarity needed to propose a duplicate pair |
| `--fuzzy-report PATH` | Write the duplicate review queue to CSV |
| `--no-dob-threshold 0.95` | Name similarity needed when a record has no date of birth |
| `--no-fuzzy` | Skip fuzzy duplicate detection |

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

The name pools are deliberately large (70 given × 55 family names) for the same reason.
An earlier version used 30 × 24, which left **136 of 364 patients sharing a full name with
someone else** — and that alone made duplicate-matching precision unmeasurable, because
most "duplicates" found were just name collisions. Mock data has to be realistic in the
dimensions you intend to measure.

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

**Pass 3 — same person, different `patient_id`.** See below; this one needs fuzzy
matching, and it is handled separately because it cannot be resolved automatically.

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

### 10. Fuzzy duplicate detection (`fuzzy_match.py`)

Passes 1 and 2 both rely on `patient_id`. Neither can see the same human registered
twice under *two different IDs* — someone re-registering at another site, or arriving via
a second source system that mints its own keys. That record has a new key, a typo in the
name, and sometimes a transposed date of birth. It is the duplicate an exact match will
never catch and a human always spots.

**Candidates are flagged, never merged.** This is the important decision. Merging two
records that turn out to be different people is a clinical safety incident; missing a
duplicate is an inconvenience. Those costs are not symmetric, so the tool produces a
review queue (`--fuzzy-report`) and a `possible_duplicate_of` column, and stops there.
`possible_duplicate` is an informational flag — a hypothesis for a human to confirm is
not grounds for invalidating a record.

**Comparison is blocked, not all-pairs.** Comparing every record against every other is
O(n²) — fine for 400 rows, ruinous at a million. Records are bucketed and only records
sharing a bucket are compared, turning ~66,000 possible pairs into a few dozen. Four
blocking keys, each covering the others' blind spots:

| Key | Catches | Blind to |
|---|---|---|
| Exact date of birth | Name typos | A mistyped date |
| Birth year + last-name prefix | A mistyped day or month | A mistyped surname |
| **Soundex of first + last name** | Phonetic variants (`Smith`/`Smyth`), insertions (`Okafor`/`Okaafor`) | Transpositions — it is anchored on the first letter |
| **Metaphone of first + last name** | The same, but modelling digraphs (`PH`→`F`, `TH`→`0`) and silent letters, so it collides far less | Transpositions, likewise |
| **Sorted letters of the full name** | Transpositions (`Wong`/`Wogn`), swapped given/family name | Insertions and deletions |

The last three are **name-only** keys. They exist solely so records with *no date of birth*
can be compared at all — otherwise those could never enter a block and would be silently
unmatchable. They are used only when at least one record in the pair lacks a date;
pairing two dated records on name alone would add nothing but noise.

**Weaker date evidence demands stronger name evidence.** An exact date of birth clears at
the 0.90 default; a transposed or same-year date must clear 0.96; a *missing* date must
clear 0.95 with no conflicting gender.

#### Jaro-Winkler, not `difflib`

Names are short, and a subsequence ratio punishes a single dropped letter far too harshly
at that length. `"rosa"` vs `"roa"` scores **0.857 under difflib but 0.933 under
Jaro-Winkler** — which was exactly the difference between missing a real duplicate and
catching it. Jaro normalises by the length of *both* strings rather than counting a longest
common subsequence, and Winkler's prefix bonus matches how names are actually mistyped:
people fumble the end of a name far more often than the start.

Both are implemented from scratch in `text_matching.py` (standard library only, so the
runtime dependencies stay pandas and numpy) and verified against Winkler's published
worked examples — `MARTHA`/`MARHTA` = 0.961, `DIXON`/`DICKSONX` = 0.813 — rather than
against their own output.

A missing date returns `"unknown"`, not `None`. **Absence of evidence is not evidence of
difference:** two records can still be the same person when one never captured a date of
birth. Two records with *conflicting* dates are a different matter — that is real evidence
they are two different people, so the pair is dropped outright.

#### The thresholds were measured, not guessed

Every threshold was set by sweeping it against the generator's ground truth, and in both
rounds the measurement contradicted the intuition.

**Round one, under difflib.** The name-only threshold was first set to 0.95 on the
reasoning that weak evidence deserves a strict bar. It turned out to be *strictly worse
than not having the feature at all* — one false positive, zero true ones — because it also
rejected the genuine matches. 0.92 was where the tier began paying for itself.

**Round two, after switching to Jaro-Winkler.** Every threshold had to be re-tuned, since
Jaro-Winkler is systematically more generous than difflib and the old values would have
been far too permissive. Rather than sweep blindly, the scores of every candidate pair were
plotted against ground truth:

| tier | true-match scores | false-match scores | threshold chosen |
|---|---|---|---|
| exact date | 0.915 – 0.982 | *none at any threshold down to 0.70* | **0.90** |
| transposed date | 0.986 | none | **0.96** |
| no date (name only) | 0.983 | **1.000** | **0.95** |

Two things fall out of that table. In the dated tiers the classes separate cleanly, so the
threshold sits just below the lowest true match with margin. In the name-only tier **the
worst false positive scores higher than the best true match** — a perfect 1.000, being two
different patients who genuinely share a name. No threshold can separate those. That is not
a tuning problem to be solved but an irreducible property of matching on a name with no
date to corroborate it, and it is why those land in a separate low-confidence tier for
human review rather than being acted on.

Final performance, by confidence tier:

```
high      2/2      exact date of birth, near-identical name
medium    5/5      transposed date, or a weaker name match on an exact date
low       1/2      name-only, no date of birth to corroborate
overall   8/9 precision, 8/8 recall     (7/7 precision on high+medium alone)
```

Recall is now complete: every planted re-registration is found. The single false positive
is the irreducible one described above.

**A gender conflict vetoes a name-only match.** If two dateless records share a name but
have *known, different* genders, the pair is dropped. `"Unknown"` never vetoes, because
that value is itself imputed. This is applied only to the name-only tier — with a
confirmed date of birth, a gender mismatch is more likely a data-entry error than proof of
two different people.

This pair is what drove the move to Jaro-Winkler. `P1240 "Rosa"` vs `P1361 "Roa"` have no
surname recorded, so the comparison string is four characters and one dropped letter cost
0.14 of difflib similarity — 0.857 against a 0.88 bar, missed. Under Jaro-Winkler the same
pair scores 0.933 and is caught. Choosing a similarity metric suited to the *shape* of the
data beat trying to tune around the wrong one.

### 11. Report

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

135 tests covering the stages where a bug would be invisible in the output:

- **`tests/test_date_parsing.py`** — every source format resolves to the same date;
  mixed formats in one column all parse; garbage becomes `NaT` instead of raising;
  day-first vs month-first inference under each kind of evidence; the documented default
  when a column offers no evidence at all; and two-digit-year century correction,
  including the `12/04/55` → `1955-04-12` round trip.
- **`tests/test_survivorship.py`** — exact duplicates collapse; the most complete record
  wins; completeness beats recency; recency breaks a completeness tie; a record with no
  admission date sorts last rather than winning by accident; the rule degrades gracefully
  when there is no admission date column at all; and the report's row counts reconcile.
- **`tests/test_fuzzy_match.py`** — swapped given/family names still matching; two
  different people who share a birthday *not* matching; the strict threshold rejecting a
  name pair that an exact date would have accepted; dateless records matching via
  name-only blocking; the gender veto firing only where it should; and blocking actually
  reducing the comparison count.
- **`tests/test_text_matching.py`** — the string algorithms, asserted against *published
  reference values* rather than against their own output: Winkler's worked examples for
  Jaro and Jaro-Winkler, the NARA table for Soundex. Metaphone is the exception — it has
  no canonical implementation, so those tests assert equivalence properties (`Smith` and
  `Smyth` must collide) which is what blocking actually depends on.

The ambiguous-date test is the one worth reading. `04/12/1985` parses cleanly under both
conventions, so no error and no null ever tells you it went wrong — the test asserts that
day-first gives 4 December and month-first gives 12 April, pinning down which reading the
pipeline commits to.

These were checked by mutation. Reversing the survivorship sort order fails 6 tests;
flipping the no-evidence date default fails 1; disabling the century correction fails 4;
dropping Jaro's transposition penalty fails 8; and removing Winkler's prefix cap, ignoring
its boost threshold, breaking Metaphone's `PH` digraph, disabling the gender veto, and
dropping accent normalisation each fail their own test.

One mutation initially **survived**: changing Jaro's match window from
`floor(max(len)/2) - 1` to `floor(max(len)/2)` broke nothing, because none of the published
reference values happen to discriminate it. That is a coverage hole the published examples
cannot close, so a hand-derived case was added — `jaro("aaab", "abcd")` is exactly 0.5 with
the correct window and 0.667 with one too wide. Mutation testing is worth doing precisely
because it finds the assertions you did not think to write.

### CI

`.github/workflows/tests.yml` runs on every push and pull request to `main`, across
Python 3.11, 3.12 and 3.13. Three stages:

1. **Unit tests** — the 39 above.
2. **End-to-end run** — generate the messy file, then clean it. Proves the pipeline still
   works as a whole, which unit tests on individual functions cannot tell you.
3. **Output invariants** — asserts the things that must be true of any clean output
   regardless of the input: no duplicate business keys, every date ISO-formatted, no
   negative lengths of stay, no null keys.

That third stage is the interesting one. It is the same idea as a dbt test — a contract
asserted against the data itself rather than the code that produced it.

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

**"How do you catch duplicates when the ID itself is different?"** Fuzzy matching on name
plus date of birth, blocked across four keys to stay tractable — and flagged rather than
merged, because the cost of wrongly merging two patients is not symmetric with the cost of
missing a duplicate. See step 10; 8/9 precision and 8/8 recall on the sample, reported per
confidence tier rather than as one number.

**"How did you pick your thresholds?"** By measuring them, and the measurement changed my
answer. For name-only matching I first set the bar at 0.95, reasoning that weak evidence
deserves a strict threshold. Sweeping it against ground truth showed 0.95 was **strictly
worse than not having the feature at all** — it contributed one false positive and zero
true ones. 0.92 is where the tier starts recovering real duplicates, taking recall from
6/8 to 7/8.

The general lesson is that a round number which *sounds* cautious is not automatically the
safe one, and you cannot tell the difference by reasoning about it. Any threshold, join
condition, or matching rule that gets picked by intuition should be swept against labelled
data before it ships, and the sweep is cheap enough that there is no excuse not to.

**"How do you know your evaluation is trustworthy?"** I did not, at first. The initial
measurement showed precision collapsing from 7/7 to 7/14, which looked like the new feature
was badly broken. It was not: my generated population drew from 30 given and 24 family
names, so **136 of 364 patients shared a full name with someone else** and nearly every
"false positive" was an accidental collision rather than a matcher error.

Enlarging the pools to 70 × 55 brought collisions down to a realistic level and the true
picture emerged. The point worth making is that a number from a benchmark is only as good
as the data behind it — if a metric moves sharply, check whether the fixture changed before
concluding the code did.

**"How did you know which part to improve?"** By measuring which one was actually
limiting. Recall could be capped either by *blocking* (pairs never compared) or by
*scoring* (pairs compared and rejected). Those need completely different fixes, and I
could not tell which from looking at the code. Instrumenting it showed blocking recall was
already 8/8 — every true pair shared a key — so better phonetic blocking could not possibly
help, and the entire deficit sat in the similarity metric. That pointed straight at
Jaro-Winkler and took recall to 8/8. Metaphone still went in, because it makes the blocks
tighter and covers phonetic variation this generator does not produce, but the honest
statement is that it earned no measurable recall gain here.

**"What would you add next?"** Schema validation with Pandera or Great Expectations, an
active-learning loop so reviewer decisions on the queue feed back into the thresholds,
Double Metaphone for non-English names (plain Metaphone encodes English pronunciation and
will mis-group a lot of the surnames in this dataset), and tracking the JSON report over
time so source-system drift shows up as a trend rather than a surprise.

