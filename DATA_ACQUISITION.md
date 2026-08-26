# Obtaining the WVS-7 data

Everything needed for Phase 0. Verified against the live WVS site on 2026-08-11.

**TL;DR** — you need **`WVS Cross-National Wave 7 csv v6 0.zip`**, and *only* v6.0.
Anything earlier has no India rows at all.

---

## 0. The two things the old docs got wrong

The repo previously told you to save the file as
`data/raw/WVS_Cross_Wave_1981_2022_CSV_v5_0.csv`. Two problems:

**That filename does not exist.** It is a blend of two different WVS products —
the *Cross-National Wave 7* file and the *Time-Series 1981-2022* file. A GitHub
code search for it returns **0 results**, while the real name returns ~97
repositories. Nothing on the WVS site is called that.

**v5.0 has no India.** India's WVS-7 fieldwork was delayed by COVID and completed
**July 2023** — after v5.0 shipped. The WVS-7 page states outright: *"The last
included survey comes from India and was completed in July 2023."* v5.0 covers
64 countries and 94,728 respondents; **v6.0 covers 66**. This is confirmed
independently: the WorldValuesBench codebook — built on v5.0, and which this
project plans to reuse — lists **88 country codes with no India entry**.

Both are fixed in the code. `src/config.py:resolve_wvs_csv()` now looks for the
real filenames and raises an actionable error otherwise, and
`src/data/load_wvs.py` fails loudly with the v5.0 diagnostic if India is absent.

---

## 1. Download

**Page:** https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp

The site is a JSP frameset — the file list lives in an iframe and every download
link is a JS handler, not an `<a href>`. There is no static URL you can `wget`,
and the download is gated behind a per-file form. Scripting it is not worth the
effort for a one-time fetch.

1. Open the page above. Scroll to **Statistical Data Files** (below
   *Questionnaire* and *Documentation*).
2. Click **`WVS Cross-National Wave 7 csv v6 0.zip`**.
3. A registration form (`AJDownloadLicense.jsp`) appears. Fill in:

   | Field | Notes |
   |---|---|
   | Title (position) | e.g. *Undergraduate researcher* |
   | Full name | |
   | Company/Institution | your university |
   | E-mail | used for the download; no confirmation loop |
   | Project title | e.g. *Silicon Sampling for Indian Public Opinion* |
   | Intended use | dropdown — pick **Dissertation** or **Academic research project** |
   | Brief description of purpose | 1-2 sentences |
   | ☑ Conditions of use | must be ticked |

4. Submit. The `.zip` downloads immediately.

There is **no account, no password, no email verification, and no waiting
period** — it is a one-shot form per file. Fill it again for each file you want.

### Conditions of Use (verbatim, abridged)

Data are available without restriction provided: (a) non-profit use; (b) correct
citations provided and sent to the WVSA for each publication; (c) **the data
files themselves are not redistributed**; (d) proper citation in the reference
list.

(c) is why `data/raw/` is gitignored. Do not commit the CSV, and do not push it
to a public Kaggle Dataset — use a **private** Kaggle Dataset for the Track A/B
notebooks.

---

## 2. Install into this project

```bash
unzip ~/Downloads/WVS_Cross-National_Wave_7_csv_v6_0.zip -d data/raw/
ls -la data/raw/WVS_Cross-National_Wave_7_csv_v6_0.csv
```

Canonical path for this repo:

```
data/raw/WVS_Cross-National_Wave_7_csv_v6_0.csv
```

Then:

```bash
python -m src.data.load_wvs --country IND
```

Expected: `Filtered to 1692 respondents in IND`. `resolve_wvs_csv()` also accepts
a renamed file as long as it is a `.csv` in `data/raw/` containing
`Cross-National` in the name.

---

## 3. Expected file details

| | |
|---|---|
| Archive | `WVS_Cross-National_Wave_7_csv_v6_0.zip` |
| Extracted CSV | `WVS_Cross-National_Wave_7_csv_v6_0.csv` |
| Format | Comma-delimited, UTF-8, header row of variable names |
| Version | 6.0 (docs revision 6.0.2) |
| Coverage | 66 countries/territories, 2017–2022 (India completed July 2023) |
| Total rows | ≈97,000 respondents (v5.0 was 94,728 across 64 countries) |
| India rows | **1,692** — matches `WVS_INDIA_N` in `src/config.py` |
| Columns | ~600 variables |
| Zip size | ~40–70 MB; CSV expands to roughly 300–500 MB |
| DOI | [10.14281/18241.24](https://doi.org/10.14281/18241.24) |

The size figures are the one item here I could **not** verify at source — the
server returns the license form rather than headers until you submit, so they are
order-of-magnitude estimates. Everything else in this table is confirmed.

Load it with `low_memory=False` (already the case in `load_wvs.py`); mixed dtypes
across ~600 columns otherwise trigger pandas warnings.

### Columns the pipeline depends on

Wave 7 uses the `Q`-numbered scheme, which is what the existing code expects:

| Variable | Meaning | Used by |
|---|---|---|
| `B_COUNTRY_ALPHA` | ISO-3 country (`IND`) | `filter_by_country` |
| `Q260` | Sex | `verify_india_demographics`, subgroup slicing |
| `Q262` | Age in years | age bands |
| `Q275` | Highest education | education band |
| `Q288` | Income scale (1–10) | income tercile |
| `H_URBRURAL` | Urban / rural | primary subgroup axis |
| `N_REGION_ISO` / `N_REGION_WVS` | Region | region slice |
| `H_SETTLEMENT`, `LNGE_ISO` | Settlement size, interview language | language slice |
| `W_WEIGHT`, `S017` | Survey weights | weighted marginals |

### Missing-value codes

`-1` Don't know · `-2` No answer · `-3` Not applicable · `-4` Not asked ·
`-5` Missing/unknown. `recode_missing_values()` maps all five to `NaN`.

Worth separating `-4` from the rest before item selection: `-4` means the item was
never fielded in India, which is a different thing from a respondent declining.
Lumping them together will distort the `MAX_MISSINGNESS_PCT` filter in
`src/data/select_items.py`.

---

## 4. Standard vs. inverted scales — read before downloading

The page offers two parallel sets. **They unpack to the same inner filename**, so
once extracted you cannot tell them apart by name. Label your download.

- **Standard** — `WVS Cross-National Wave 7 csv v6 0.zip`
- **Inverted** — `WVS Cross-National Inverted Wave 7 csv v6 0.zip`, all scales
  flipped so higher = more of the named construct

This matters here specifically: **WorldValuesBench was built on the *inverted*
v5.0 file.** Its `data_preparation.py` header says so explicitly. So if you reuse
WVB's `codebook.json` / `answer_adjustment.json` to verbalise answer options —
which `README.md` plans to do — and feed it the *standard* file, the option text
will be reversed for many items. Nothing will crash; every downstream metric will
just be quietly wrong.

**Recommendation: download the inverted v6.0 file** (`Inverted ... csv v6 0.zip`).
That gives WVB-compatible scale directions *and* India. If you prefer the
standard file, plan to re-derive the answer mappings from
`WVS7 Codebook Variables report V6.0.pdf` rather than trusting WVB's.

Either way, verify on a few items before Phase 1 freezes `selected_items.json`.

### One more WVB caveat

WVB's mappings transfer for the `Q1`–`Q290` value items, but **not** for India's
region and language codes — those simply are not in a v5.0-derived codebook.
`N_REGION_ISO` entries for Indian states must be built from the v6.0 codebook
yourself. Budget for this in Phase 1; it is the input to the `region` subgroup
slice, which is one of the paper's headline axes.

---

## 5. Also download (free, no registration form)

From the same page, these use a plain download handler:

- **`WVS7 Codebook Variables report V6.0.pdf`** — the variable dictionary. You
  will need it constantly for `select_items.py` and for verbalisation.
- `WVS-7 Master Questionnaire 2017-2020 English.pdf` — exact English question
  wording for prompt construction.
- `List of countries WVS7 2017-2022 V6.0.pdf` — confirms India's presence and
  fieldwork dates.
- `WVS Results By Country 2017-2022 v6.0.0.pdf` — marginals to sanity-check your
  parsed India distributions against.
- `WVSA Citation format 2024.pdf`

---

## 6. Do **not** use these files

| File | Why not |
|---|---|
| `WVS TimeSeries 1981 2022 Csv v5 0.zip` | This is what the old filename half-referred to. It is the longitudinal WVS/EVS-harmonised file using `S003`/`A008` variable names — **not** `B_COUNTRY_ALPHA`/`Q260`. Incompatible with every module in `src/`. Its wave-7 slice also predates India's release. |
| `WVS Cross-National Wave 7 csv v5 0.zip` | No India. |
| Country-only India download | The repo needs the cross-national file so the country parameterisation in the README (extend to other countries) still works. |

---

## 7. Alternatives and fallbacks

**GESIS ZA7505 — Joint EVS/WVS 2017-2022** (verified to include India):

- https://search.gesis.org/research_data/ZA7505
- Version 5.0.0, DOI [10.4232/1.14320](https://doi.org/10.4232/1.14320),
  N = 156,658, fieldwork through 2023-07-02
- Geographic coverage confirmed to list **India (IN)**
- Requires a free GESIS account (heavier than the WVS form — real registration)

Caveat: this is a *merged* EVS+WVS file and uses the harmonised EVS variable
scheme, so `load_wvs.py`'s column expectations would need remapping. Treat it as
a fallback if worldvaluessurvey.org is down, not as the default.

**Other routes**

- **Institutional library** — many universities mirror WVS through ICPSR or a
  data-services desk; ask before fighting the website.
- **WVSA Secretariat**, `wvsa.secretariat@gmail.com` — the site itself directs
  download problems here. Responsive for academic requests.
- Third-party GitHub/OSF copies exist but redistribution violates the Conditions
  of Use you agreed to. Don't cite one as your source.

---

## 8. Troubleshooting

**Download link does nothing.** The site's own advice, printed on the page: *"In
case of repeated problems with the data download, try using an alternative
Internet browser."* The handlers are JS form submissions; Safari's tracking
prevention and most ad-blockers break them. Try Chrome with extensions disabled.

**Site restructured / links renamed.** The download is a POST to
`AJDownloadLicense.jsp` with a numeric `DOID`. As of 2026-08-11:

| File | DOID |
|---|---|
| Cross-National Wave 7 csv v6 0 | `11356` |
| Cross-National **Inverted** Wave 7 csv v6 0 | `11357` |
| Cross-National Wave 7 spss v6 0 | `10733` |
| Codebook Variables report V6.0 (no license form) | `11055` |

If the page layout changes, the file list still lives at
`AJDocumentation.jsp?CndWAVE=7&COUNTRY=` inside a nested iframe.

**`FileNotFoundError` from `resolve_wvs_csv`.** The zip likely extracted into a
subdirectory. `ls -R data/raw/` and move the CSV up one level.

**`ValueError: No respondents found for IND`.** You have v5.0. Re-download v6.0.
The error message says so too.

**`KeyError: No country column found`.** You downloaded the Time-Series file.
See §6.

**pandas `DtypeWarning` on load.** Expected with ~600 columns;
`low_memory=False` is already set.

**India N is not 1,692.** `load_wvs.py` warns if it drifts by more than 50.
Investigate before freezing `selected_items.json` — everything downstream
assumes this N.

---

## 9. Citation (mandatory)

```bibtex
@misc{wvs7,
  author    = {Haerpfer, C. and Inglehart, R. and Moreno, A. and Welzel, C.
               and Kizilova, K. and Diez-Medrano, J. and Lagos, M. and
               Norris, P. and Ponarin, E. and Puranen, B.},
  title     = {World Values Survey: Round Seven -- Country-Pooled Datafile
               Version 6.0},
  year      = {2022},
  publisher = {JD Systems Institute \& WVSA Secretariat},
  address   = {Madrid, Spain \& Vienna, Austria},
  doi       = {10.14281/18241.24}
}
```

Condition (b) requires *sending* the citation to the WVSA on publication —
email `wvsa.secretariat@gmail.com` when the paper lands. Easy to forget; it is
part of the agreement you signed.
