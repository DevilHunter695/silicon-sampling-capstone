# Silicon Sampling for Indian Public Opinion: A Fidelity and Failure-Mode Study

**Implementation plan — capstone (NMIMS) + publishable paper**
Owner: Bhanuj Bhalla · Timeline: ~15 weeks · Deliverables: code repo, paper, deck

---

## 1. Context

"Silicon sampling" — conditioning an LLM on a demographic profile and using its output as a stand-in for a real survey respondent — is already a commercial product (Qualtrics, VeraSight). Its foundational validation (Argyle et al. 2023, *Out of One, Many*) was conducted entirely on American survey data, and almost all follow-up work has stayed in that lane.

This project asks whether the technique holds for Indian respondents, and — more precisely — **whether it holds evenly across Indian subgroups**. The concern is concrete: LLM pretraining corpora over-represent English-speaking, urban, higher-income populations. If a market researcher uses silicon sampling to "survey India," the simulated sample may quietly reproduce urban middle-class opinion while misrepresenting rural, low-income, and non-English-dominant respondents — with no error signal to warn them.

### 1.1 Why the framing had to change (read this first)

The original framing — *"nobody has tested silicon sampling on India"* — does not survive a literature check. Three papers overlap:

| Prior work | What it did | Consequence for us |
|---|---|---|
| [WorldValuesBench](https://aclanthology.org/2024.lrec-main.1539/) (LREC-COLING 2024) | Built 20M `(demographics, question) → answer` examples from WVS-7 across 94,728 people worldwide; zero-shot eval of Alpaca-7B, Vicuna-7B, Mixtral-8x7B, GPT-3.5 with Wasserstein-1. [Code public](https://github.com/demon702/worldvaluesbench) | India is already in their data. We **cannot** claim first-to-WVS-India. |
| [BYU thesis](https://scholarsarchive.byu.edu/etd/11085/) (2025) | Fine-tuned LLaMA-3.1-70B on GSS (US) for silicon sampling; subgroup analysis by race/age | We **cannot** claim first-to-fine-tune for silicon sampling. |
| [Cross-Survey Transfer](https://arxiv.org/abs/2607.03091) (2026) | Taiwan TEDS, unseen-question prediction, vs. random-forest baselines | Different task; low overlap, but shows the "beat a classical ML baseline" bar is now standard. |

**What is still genuinely unclaimed**, and what this paper will own:

> No published work measures **within-country** silicon-sampling fidelity across Indian demographic strata, or tests whether **fine-tuning narrows or widens** the fidelity gap between well-represented and under-represented subgroups.

WorldValuesBench reports one aggregate number per country. It does not tell you that fidelity might be acceptable for urban graduates in Maharashtra and unusable for rural low-income respondents in Bihar. That distinction is the entire practical risk of the technology, and it is the contribution here.

This reframing is **stronger for publication, not weaker.** A reviewer who knows WorldValuesBench would reject the broad claim in one line. The narrow claim is precise, falsifiable, unclaimed, and socially consequential — which is exactly the profile workshop reviewers reward. It also lets us *cite WVB and BYU as foundations we build on* rather than competitors we must beat, and reuse WVB's public code.

### 1.2 Contribution claims (the four bullets that go in the paper's intro)

1. The first **intra-national** fidelity audit of silicon sampling for India, stratified by urban/rural, income, education, region, and interview language.
2. A **fair-comparison benchmark suite** including the baseline this literature usually omits: a demographic-cell lookup table built from the training split. If an LLM cannot beat a lookup table, silicon sampling adds nothing — we test this explicitly.
3. The first measurement of whether **fine-tuning redistributes fidelity** — reporting Δ in the *between-subgroup gap*, not just Δ in the mean.
4. A released, reproducible pipeline (code + prompts + eval harness), country-parameterised so others can extend it.

### 1.3 Scope decisions already made

- **India only** for this paper. USA contrast is deferred to Future Work — but the pipeline will be country-parameterised from day one (`--country IND`), so adding it later is a config flag, not a rewrite.
- Because there is no cross-national control, **all claims must be phrased as intra-India gaps** ("fidelity for rural low-income respondents is 2.4× worse than for urban graduates"), never as "worse than the West." Cite WVB's published cross-country numbers as loose external context only.
- Compute: **Kaggle free tier** (2× T4 16GB, 30 GPU-hrs/week, 12-hr sessions) — most capable free option, and session length beats Colab free.

---

## 2. Research questions (revised, precise)

- **RQ1 (fidelity):** Given only a demographic profile, how accurately does an LLM reproduce a real Indian respondent's WVS answers — at the individual level and at the population-distribution level?
- **RQ2 (value-add):** Does the LLM beat (a) the national marginal, (b) a demographic-cell lookup table, and (c) a supervised classifier on the same demographics?
- **RQ3 (equity — the headline):** Is fidelity uniform across Indian subgroups, or concentrated in well-represented strata?
- **RQ4 (fine-tuning):** Does fine-tuning on real Indian survey data improve fidelity, and does it *narrow or widen* the between-subgroup gap?

**Working hypothesis (to test, not assume):** zero-shot fidelity degrades for rural / low-income / low-education / non-English-interview respondents; fine-tuning raises the mean but may leave the gap unchanged or wider, by overfitting to majority strata.

---

## 3. Data

**Primary:** World Values Survey Wave 7, India. **N ≈ 1,692** — verify at download; this is smaller than typical assumptions and drives the CV design in §5.4.

Download from [worldvaluessurvey.org](https://www.worldvaluessurvey.org) → Data & Documentation → Statistical Data Files. Requires free registration + terms acceptance. Get the **cross-national WVS-7 CSV** (not the India-only file) so `--country USA` works later without a second download.

**Variables**

- *Demographic conditioning set:* `Q260` sex, `Q262` age, `Q273` marital status, `Q274` no. children, `Q275` education (ISCED), `Q279` employment, `Q281` occupation, `Q287` subjective social class, `Q288` income decile, `Q289` religious denomination, `H_URBRURAL`, `N_REGION_ISO` (state), `G_TOWNSIZE`, `LNGE_ISO` (interview language).
- *Subgroup slicing set (§5.5):* urban/rural, income (low/mid/high terciles), education (≤primary / secondary / tertiary), region, interview language (English vs. Hindi vs. other), age band, sex, religion.
- *Target items:* ~45 value questions selected per §5.1.

**Reuse, do not rebuild:** clone [`demon702/worldvaluesbench`](https://github.com/demon702/worldvaluesbench) for its WVS variable→natural-language question text mapping, answer-option verbalisation, and demographic verbalisation. That is 1–2 weeks of tedious codebook transcription you can skip. Check its LICENSE before copying code; cite it either way. Their repo also does not redistribute raw WVS data (licensing) — you download it yourself, same as us.

---

## 4. Models and compute

| Role | Model | Where |
|---|---|---|
| Track A — closed | Gemini 2.5 Flash | AI Studio free tier |
| Track A — open **and** Track B base | **Llama-3.1-8B-Instruct** | Kaggle T4, QLoRA 4-bit |
| Track A — open #2 | Qwen2.5-7B-Instruct | Kaggle T4 |
| Fast iteration / pipeline debug | Gemma-3-4B-it | Kaggle T4 |

**Design rule:** the open-weight Track A model **must be the same checkpoint** used as the Track B base. Otherwise "fine-tuning helped" is confounded with "different model." This is the single most important experimental-control decision in the project.

Llama-3.1-8B requires accepting a gated licence on HuggingFace (2 minutes). If that is friction, swap primary to Qwen2.5-7B-Instruct — no gating — and note the substitution in the paper.

---

## 5. Method

### 5.1 Item selection (~45 questions)

Filter WVS-7 India items by, in order:

1. **Clean response scale** — ordinal Likert (4-pt, 10-pt) or small categorical. Drop free-text and >10-category nominal items.
2. **Low missingness** — drop items with >10% missing/refused in the India sample.
3. **Non-degenerate variance** — drop items where the modal answer exceeds ~85% of responses. *This filter matters:* on a near-unanimous item, a model that always predicts the mode scores near-perfect fidelity and the metric becomes uninformative. Rank by normalised entropy and sample across the range.
4. **Domain coverage** — stratify across WVS thematic blocks (social values, trust/social capital, economic values, corruption, religion, ethics, politics/governance) so per-domain breakdowns are possible.

Freeze the item list to `data/processed/selected_items.json` and never change it after Track A begins.

### 5.2 Persona prompt construction + ablations

Four prompt conditions, all producing a first-person framing:

- **P0 — no demographics** (control). Measures the model's ungrounded prior. Essential: it is how you prove demographic conditioning does anything at all.
- **P1 — minimal** (age, sex, region).
- **P2 — full structured** (all 14 attributes as a labelled list).
- **P3 — full naturalistic backstory** (Argyle-style first-person prose paragraph).

P2 is the primary condition for headline results; P0/P1/P3 are the ablation. Store templates in `src/prompts/templates.py`, versioned.

### 5.3 Response elicitation — use logprobs, not sampling

For each `(respondent, item, model, prompt_condition)`, constrain the model to emit a single answer token and **read the log-probabilities over the valid answer-option tokens**, renormalising to a distribution.

Why this matters: it yields a full predicted distribution from **one** forward pass instead of 20+ samples. It is ~20–40× cheaper, has far lower variance, and is required for the distributional metrics in §5.4 to be stable at N=1,692.

- Open-weight models on Kaggle: direct via HF `transformers` output logits. Straightforward.
- Gemini: use the `responseLogprobs` / `logprobs` generation-config fields. **Verify this is available on the free tier before committing** — if not, fall back to temperature-1.0 sampling with n=20 per item and document the difference.
- **Log refusals and off-format outputs separately.** Refusal rate is a result, not an error — and if refusal rate correlates with subgroup, that is a publishable finding on its own.

All inference must be **resumable and cached** (write each result to disk keyed by `(respondent_id, item_id, model, condition)`), because Kaggle sessions die and Gemini rate-limits.

### 5.4 Evaluation metrics

**Individual level** — did it predict *this person*?
- Exact-match accuracy
- Mean Absolute Error on the ordinal scale (better than accuracy for Likert)
- **Negative log-likelihood of the true answer** under the predicted distribution — a proper scoring rule, and the best single number

**Distributional level** — does the simulated population match the real one? (Argyle's "algorithmic fidelity")
- **Jensen–Shannon divergence** — comparable to the silicon-sampling literature
- **Wasserstein-1 distance** — comparable to WorldValuesBench, and respects ordinal ordering, which JSD ignores

Report both. Using only JSD on ordinal Likert data is a known weakness a reviewer will flag.

**Baselines (§RQ2 — do not skip these; they are what make the paper reviewable)**
1. Uniform random
2. **National marginal** — predict the overall India distribution, ignoring demographics
3. **Demographic-cell lookup** — from the training split, the empirical answer distribution for that person's demographic cell. *If the LLM cannot beat this, silicon sampling is an expensive lookup table.*
4. **Supervised classifier** — multinomial logistic regression and gradient boosting on one-hot demographics, trained on the same split

**Cross-validation, not a single split.** With N≈1,692, an 80/20 split leaves ~338 test respondents; sliced into subgroups, some cells fall below n=30 and nothing is defensible. Use **5-fold cross-validation stratified on the subgroup variables**, so every respondent gets an out-of-fold prediction and evaluation uses the full N. Start with **3-fold on Gemma-3-4B** to validate the pipeline, then scale to 5-fold on the 8B model if the Kaggle budget allows.

**Uncertainty:** bootstrap confidence intervals over respondents (1,000 resamples) for every reported metric, including every subgroup cell. Always report **n per slice** and flag any slice with n < 30.

### 5.5 Subgroup analysis — the actual contribution

For every model × condition, report metrics per slice of: urban/rural, income tercile, education band, region, interview language, age band, sex, religion.

Then the two numbers the paper is built on:

- **Fidelity gap** = (worst slice) − (best slice), per metric
- **Δ_gap = gap(fine-tuned) − gap(zero-shot)**

`Δ_gap > 0` means fine-tuning improved the average while *widening* inequality between subgroups — helping the majority at the expense of the under-represented. If that holds, it is the headline finding and the paper's title.

### 5.6 Track B — fine-tuning

- QLoRA 4-bit, `peft` + `transformers` + `trl` (SFTTrainer) on Kaggle T4.
- Training example = **exactly the Track A P2 prompt** + the respondent's real answer as target. Identical prompt format across tracks or the comparison is invalid.
- **Respondent-level split** — a test respondent's answers never appear in training for *any* item. Stratify folds on subgroup variables.
- Suggested start: r=16, alpha=32, lr=2e-4, 2 epochs, seq len 512, dropout 0.05. Tune on fold 1 only, then freeze.
- ~1,354 train respondents × 45 items ≈ 61k examples/fold. Budget ~3–4 GPU-hrs/fold on T4 for 8B; 5 folds ≈ 20 hrs, which fits 30 hrs/week with little slack. Checkpoint to Kaggle datasets so a dead session does not cost a full run.
- **Optional ablation if time permits:** question-level holdout (train on 30 items, test on 15 unseen items, same respondents) — tests generalisation to *new questions* vs. *new people*. Nice-to-have, cut first if behind.

### 5.7 Contamination check (reviewers will ask)

WVS-7 is public and predates the models' training cutoffs. Include a short section:
- Probe whether models can reproduce WVS India aggregate statistics or recognise the instrument verbatim
- Compare fidelity on WVS items vs. paraphrased/reworded versions of the same items
- Report the result honestly either way and discuss the limitation

Cheap to run, and its absence is a standard reviewer objection.

---

## 6. Repo structure

```
CAPSTONE/
├── data/
│   ├── raw/                      # WVS CSV — gitignored, redistribution not permitted
│   └── processed/
│       ├── india_wvs7.parquet
│       ├── selected_items.json   # frozen after Track A starts
│       └── folds.json            # stratified 5-fold assignment
├── src/
│   ├── config.py                 # country code, model registry, paths
│   ├── data/
│   │   ├── load_wvs.py           # parse, recode missing (-1..-5), country filter
│   │   ├── select_items.py       # §5.1 filters
│   │   └── build_folds.py        # stratified CV split
│   ├── prompts/
│   │   ├── templates.py          # P0/P1/P2/P3
│   │   └── verbalize.py          # demographics + answer options → text (adapt from WVB)
│   ├── inference/
│   │   ├── base.py               # cached, resumable runner interface
│   │   ├── gemini.py
│   │   └── hf_local.py           # logprob extraction over answer tokens
│   ├── finetune/
│   │   ├── build_dataset.py      # (prompt, answer) pairs per fold
│   │   └── train_qlora.py        # Kaggle entrypoint
│   ├── eval/
│   │   ├── metrics.py            # JSD, W1, MAE, NLL, accuracy
│   │   ├── baselines.py          # marginal, cell-lookup, logreg, GBM
│   │   ├── subgroups.py          # slicing + fidelity gap + Δ_gap
│   │   └── bootstrap.py          # 1000-resample CIs
│   └── report/
│       └── make_figures.py
├── notebooks/
│   ├── kaggle_trackA_openweight.ipynb
│   └── kaggle_trackB_finetune.ipynb
├── results/                      # cached raw predictions + metric tables
├── paper/                        # LaTeX (ACL style)
└── deck/
```

---

## 7. Phase-by-phase schedule (15 weeks)

| Phase | Weeks | Work | Done when |
|---|---|---|---|
| **0 — Setup** | 1 | Register + download WVS-7; clone WVB repo; scaffold repo; `uv` env; write literature-review notes on the 3 overlapping papers | WVS CSV loads; India rows = ~1,692 confirmed |
| **1 — Data** | 2–3 | `load_wvs.py`, missing-value recoding, item selection (§5.1), stratified folds, demographic verbalisation | `selected_items.json` frozen (~45 items); fold balance table printed |
| **2 — Baselines first** | 3–4 | Implement **all** baselines + metrics + bootstrap **before any LLM call** | Cell-lookup and logreg numbers exist; these are the bar the LLM must clear |
| **3 — Track A** | 5–7 | Inference runners; logprob extraction; run 3 models × 4 prompt conditions with caching; refusal logging | Full prediction cache on disk; no run needs restarting from scratch |
| **4 — Track B** | 8–10 | Build fold datasets; QLoRA on Gemma-3-4B (3-fold) to validate; then Llama-3.1-8B (5-fold) | Out-of-fold predictions for all N respondents |
| **5 — Analysis** | 11–12 | Subgroup slicing, fidelity gap, Δ_gap, bootstrap CIs, contamination probe, figures | Every headline number has a CI and an n |
| **6 — Paper** | 13–14 | 8-page ACL-format write-up; release repo | Draft complete, repo public with README + reproduction steps |
| **7 — Deck + submit** | 15 | Capstone deck; arXiv submission; workshop submission | Submitted |

**Sequencing note — Phase 2 before Phase 3 is deliberate.** Building baselines before touching an LLM means you know the bar the LLM must clear *before* you have sunk weeks into inference, and it prevents the common failure of reporting an LLM number with nothing to compare it to.

---

## 8. Verification

Each phase has a concrete check, not just "it ran":

- **Phase 1:** print the India demographic marginals and eyeball against published WVS-7 India documentation (sex ratio, urban/rural share). If these are off, the recoding is wrong. Assert no `-1..-5` missing codes survive into the processed file.
- **Phase 2:** the national-marginal baseline must beat uniform-random; the cell-lookup baseline must beat the national marginal on at least most items. If not, the fold split or metric code is broken.
- **Phase 3:** run 20 respondents end-to-end first and inspect raw model output by hand before launching the full sweep. Confirm P0 (no demographics) produces *measurably different* predictions from P2 — if they are identical, demographic conditioning is silently not reaching the model and every downstream result is void.
- **Phase 4:** confirm zero respondent-ID overlap between train and test in every fold, programmatically. Confirm the fine-tuned model beats its own zero-shot self on the training distribution as a sanity check before trusting held-out numbers.
- **Phase 5:** every table cell reports n and a bootstrap CI. Any slice with n < 30 is flagged in the paper, not silently averaged.
- **End-to-end:** `make reproduce` (or a documented script sequence) regenerates every figure from cached predictions on a clean checkout.

---

## 9. Risks and contingencies

| Risk | Mitigation |
|---|---|
| **The hypothesis does not hold** — fidelity is uniform across subgroups | Still publishable. With the §5.4 baseline suite you can report "silicon sampling does not beat a demographic lookup table for India," which is a useful negative result. Do not panic at week 10; the methodology carries the paper, not the direction of the result. |
| Small subgroup cells (n<30) | 5-fold CV over full N (§5.4); report n everywhere; pre-register which slices are primary vs. exploratory |
| Gemini free tier lacks logprobs | Fall back to n=20 sampling; document; open-weight models still give exact logprobs |
| Kaggle 30hr/week exhausted | Drop to 3-fold; use Gemma-3-4B as primary; cut the §5.6 optional ablation |
| Kaggle session dies mid-training | Checkpoint every N steps to a Kaggle Dataset; all inference cached and resumable |
| Data contamination objection | §5.7 probe, run and reported regardless of outcome |
| WVS-7 India N smaller than expected | Confirmed at Phase 0. If a fold design becomes untenable, pooling WVS waves 5+6 (N≈2,001 + 4,078) is the escape hatch — but note the temporal confound explicitly |

---

## 10. Publication targets

Realistic for a strong undergraduate capstone, in order of fit:

1. **C3NLP** (Cross-Cultural NLP workshop, ACL-affiliated) — the single best fit; explicitly wants this kind of work
2. **NLP4PI** (NLP for Positive Impact) — the equity/failure-mode angle fits
3. **ACL / EMNLP / NAACL Student Research Workshop** — designed for exactly this career stage
4. **arXiv preprint** (cs.CL, cross-list cs.CY) — do this regardless, as soon as the draft is stable

Verify current deadlines directly; workshop dates move year to year. Write in ACL format from the start so submission is a formatting no-op.

**Public code release is worth real reviewer credit** for this kind of empirical paper — budget the time in Phase 6.

---

## 11. Explicitly out of scope (→ Future Work section)

- **USA contrast** — deferred by decision. Pipeline is country-parameterised so it is a config flag later. Note in Limitations that without it, no cross-national claim is made.
- Additional non-WEIRD countries (Nigeria, Indonesia)
- India-native instruments (CSDS-Lokniti NES, IHDS)
- New human data collection / focus groups — dropped as out of scope for the timeline
- Models beyond the four listed
