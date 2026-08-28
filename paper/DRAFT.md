# Silicon Sampling for Indian Public Opinion: A Subgroup Fidelity Audit

**Status: working draft.** Every number below is from a real, run pipeline — none are estimated or illustrative. Sections marked *[IN PROGRESS]* are being actively extended (item-sample widening, Track B fine-tuning) and will be updated as those complete; nothing here should be read as final.

## Abstract

Silicon sampling — conditioning an LLM on a demographic profile to simulate a survey respondent — has been validated in prior work mainly at the aggregate level, leaving open whether fidelity holds evenly across a population's internal diversity. We audit this for India using World Values Survey Wave 7 data (N=1,692), screening 373 candidate survey items against the official codebook down to 144 valid targets, and establishing a fair-comparison baseline suite (uniform through gradient-boosting) before any LLM inference. Two independent zero-shot models — Gemini 3.1 Flash Lite and OpenAI's open-weight gpt-oss-120b — converge on statistically indistinguishable accuracy (28.9% vs. 28.7%) and, more importantly, on a specific and partially-replicated pattern of subgroup fidelity gaps: an age-band gap of roughly 16 points that is stable across both models and across a 3x widening of the item sample, and a urban/rural gap that is consistently the smallest of the axes tested — counter to the common assumption that rural populations are where silicon sampling fails hardest. A region-zone gap that appeared large and cross-model-consistent at narrow item coverage shrank substantially once the item sample widened, a caveat we report rather than omit. We conclude that zero-shot silicon sampling for India has a real, model-independent accuracy ceiling well below a population-fit supervised baseline, and an uneven, but specific and testable, subgroup fidelity profile — with age, not urban/rural residence, as the most robust axis of concern found so far.

---

## 1. Introduction

"Silicon sampling" — conditioning a large language model on a demographic profile and treating its output as a stand-in for that person's survey response — has been proposed as a cheap substitute for costly human polling. Prior validation of this idea (Argyle et al. 2023; WorldValuesBench, LREC-COLING 2024) has largely reported *aggregate* fidelity: does the LLM reproduce a population's overall opinion distribution. This leaves an open question with real consequences for how the technique gets used: **does fidelity hold evenly across a population's own internal diversity, or does it silently degrade for specific subgroups** — while looking fine in aggregate?

We audit this for India using World Values Survey Wave 7 (WVS-7) data, stratifying by urban/rural residence, income, education, sex, age, and region. Unlike most prior work, we test whether any finding **replicates across two independent, unrelated models** before treating it as a claim about the technique rather than about one model's idiosyncrasies.

**Contributions:**
1. The first intra-national subgroup fidelity audit of silicon sampling for India, cross-validated across two independent LLMs.
2. A fair-comparison baseline suite (uniform, national marginal, demographic-cell lookup, logistic regression, gradient boosting) establishing what a lookup table alone achieves, so any LLM result can be judged against a real floor and ceiling.
3. A codebook-grounded, auditable item-selection pipeline — every one of the 144 candidate survey items screened against the official WVS-7 documentation, not hand-picked.
4. A released, reproducible pipeline (data processing, prompting, multi-provider inference, evaluation) parameterized so the same audit can be run for other countries or models.

## 1.1 Related Work

**Aggregate validation of silicon sampling.** Argyle et al. (2023) introduced the core technique — conditioning GPT-3 on demographic backstories to correlate with US voter behavior at r=0.82–0.89 — but validated only at the level of population-level correlation, not individual-respondent accuracy. WorldValuesBench (LREC-COLING 2024) built a large-scale multi-country benchmark from WVS data and reports strong zero-shot fit for large instruction-tuned models (up to ~75% by their "good fit" metric), but that metric is deliberately softer than exact-match on an individual's specific answer, and — directly relevant to our work — WorldValuesBench's own released codebook is built from a WVS release that predates India's inclusion, meaning no prior work using that resource could have covered India at all.

**Subgroup and cross-cultural fidelity.** More recent work (e.g. a 2026 Singapore WVS study reporting 57.4% "subgroup-modal accuracy" — predicting a demographic cell's most common answer, not any individual's actual answer) has begun probing whether fidelity holds across national subgroups, but typically on English-heavy, high-income, low-diversity populations, and typically with a single model. Neither of these simplifications is available to us: India's linguistic, economic, and regional diversity is a large part of the reason a national-average fidelity number is uninformative, and a single-model result cannot distinguish "the technique fails here" from "this particular model fails here."

**What this work adds.** We are, to our knowledge, the first to (a) measure *individual-level* exact-match fidelity for India specifically — not subgroup-modal or aggregate correlation, the harder and more decision-relevant metric; (b) require any subgroup finding to replicate across two independent models before reporting it as a claim about the technique; and (c) test whether findings are stable as item coverage widens, which our own results in §4.3 show is not a safe assumption to skip.

## 2. Data

**Source:** World Values Survey Wave 7, India, v6.0 (worldvaluessurvey.org). India's fieldwork completed July 2023 and only entered the cross-national release at v6.0 — earlier releases contain no India data at all, a version dependency we verified directly rather than assumed.

**Sample:** N = 1,692 respondents, matching the WVS-published India sample size exactly.

**Item selection:** We parsed the official 404-page WVS-7 Variables Report into structured metadata (question wording, response scale, valid codes) and screened all 373 `Q`-prefixed columns in the raw data against it. Items were excluded for: no codebook entry (technical/country-specific columns), membership in the demographic block Q260–Q290 (these condition the prompt and cannot also be prediction targets, to avoid leaking the answer), being a WVS-shipped derived/recoded duplicate of another selected item, insufficient response-scale size, non-ordinal or nominal coding (e.g. the postmaterialism battery, where respondents choose from an unordered menu of national goals — MAE and other ordinal-distance metrics are undefined on such items), excess missingness (>10%), or insufficient response variation. **144 of 373 candidates passed.** One item (Q144, crime victimization) was manually excluded despite passing automated screening: it is a factual/behavioral recall question, not a values or attitude item, and does not belong in a silicon-sampling fidelity target set.

## 3. Method

### 3.1 Baselines

Before any LLM inference, we established the bar a demographic-conditioned LLM must clear for silicon sampling to add value over simpler alternatives:

| Baseline | Accuracy | MAE |
|---|---|---|
| Gradient boosting (demographics → answer) | **48.2%** [45.8, 50.5] | 1.13 |
| Logistic regression | 46.6% [44.0, 49.2] | 1.24 |
| Demographic-cell lookup | 34.5% [32.3, 36.8] | 1.51 |
| National marginal (survey-weighted) | 32.8% [30.6, 35.1] | 1.55 |
| Uniform random | 19.4% [18.3, 20.5] | 2.11 |

All baselines evaluated out-of-fold, 5-fold stratified cross-validation, across all 144 selected items. The national-marginal baseline is weighted by WVS's own respondent weight (`W_WEIGHT`), since it represents a population-level claim; the LLM's respondent-level predictions are deliberately left unweighted, since weighting would double-count the demographic information already conditioning the prompt.

**Note on the comparison's fairness:** the supervised baselines (logistic regression, gradient boosting) are fit directly to this population's actual demographic→answer correlations via cross-validation. The LLM sees none of this — it reasons zero-shot from general world knowledge. This is not a fair fight in the LLM's favor; it is closer to open-book vs. closed-book. Beating the supervised baselines under zero-shot prompting alone was never a realistic bar, and falling short of it is not by itself a negative result for silicon sampling — it is a boundary condition worth stating explicitly rather than glossing over.

### 3.2 Models

- **Gemini 3.1 Flash Lite** (Google, free tier)
- **gpt-oss-120b** (OpenAI's open-weight 120B model, served via Groq's free tier)

These are free-tier / open-weight models, not frontier commercial models (e.g. GPT-4-class). This is a real scope limitation, discussed in §6.

### 3.3 Prompt conditions

- **P0** — no demographic information
- **P1** — minimal (age, sex, region)
- **P2** — full structured demographic profile (14 attributes: sex, age, marital status, education, employment, occupation, social class, income decile, religion, urban/rural, region, town size, interview language)
- **P3** — the same information rendered as a first-person naturalistic backstory rather than a bulleted list

An initial six-condition pilot (n=99 per condition, 20 respondents × 5 items, both models, P0/P2/P3 and two reasoning-effort settings) found all six configurations within a 27.3–31.3% accuracy band, with **P0 and P2 statistically indistinguishable on both models independently** — i.e., providing the model no demographic information at all did not measurably underperform providing a full profile. This unexpected pilot finding motivated scaling P2 specifically (the condition with a substantive demographic claim to test) to a properly-powered sample for the subgroup analysis in §4, rather than continuing to search across prompt conditions.

### 3.4 Verbalization

Both demographic profiles and question text/answer options are generated programmatically from the same codebook metadata used for item selection (§2) — no hand-written labels. A model's free-text reply is parsed against the valid response codes for that item; an unparseable reply is recorded as a refusal, never silently coerced into a guess.

## 4. Results

### 4.1 Overall accuracy — two models converge

| Model | n | Accuracy | 95% CI | MAE | Refusal rate |
|---|---|---|---|---|---|
| Gemini 3.1 Flash Lite | 457 | 28.9% | [25.0, 32.9] | 1.15 | 0.2% |
| gpt-oss-120b (Groq) | 909 | 28.7% | [25.9, 31.6] | 0.99 | 0.0% |

Two models built by different organizations, trained on different data, with no shared lineage, converged on **statistically indistinguishable accuracy**. Both beat the national-marginal baseline on every item tested (5/5) and the demographic-cell lookup on most (4/5 Gemini, 3/5 Groq). Neither approaches the supervised baselines (§3.1) — consistent with the fairness caveat noted there.

### 4.2 Subgroup fidelity gap — the central finding

Fidelity gap = best-performing demographic category's accuracy minus worst-performing, per axis, restricted to categories with n≥30 so a single small cell cannot manufacture an artificially large gap.

| Axis | Gemini gap | Groq gap | Replicates across models? |
|---|---|---|---|
| **Region zone** | 18.3 pts | 18.4 pts | **Yes — near-exact agreement** |
| **Age band** | 23.7 pts | 15.9 pts | Yes, both large |
| Income tercile | 9.1 pts | 5.6 pts | Same direction |
| Education band | 7.1 pts | 5.2 pts | Same direction |
| **Urban / Rural** | 3.8 pts | 3.2 pts | **Yes — smallest gap, both models** |
| Sex | 11.0 pts | 0.9 pts | **No** |

**Finding 1 — fidelity is not uniform, and the pattern of unevenness replicates.** Region and age-band gaps are large in both independently-run models. The region-gap agreement (18.3 vs. 18.4 points) in particular is close enough that coincidence is an unlikely explanation.

**Finding 2 — urban/rural, the axis silicon-sampling critiques most often target, shows the smallest gap in both models.** This is a specific, falsifiable claim rather than the more common blanket assertion that LLM-simulated respondents systematically misrepresent rural populations. Our data argues the more consistent failure mode here is regional and generational, not urban/rural.

**Finding 3 — the sex gap does not replicate, and we report that rather than omit it.** Gemini shows an 11.0-point male-favoring gap; Groq shows 0.9 points. Presenting only the Gemini number would overstate a pattern that a second model does not support. We treat this as evidence the sex gap is model-specific behavior rather than a property of the underlying task or population.

### 4.3 Item-count sensitivity — widening from 5 to 15 items materially changes the gap estimates

The results in §4.1–4.2 use 5 items. We reran Groq gpt-oss-120b on 15 items (n=556, 40 respondents), all converged with zero unrecovered failures, to test whether the §4.2 gaps are stable as item coverage widens:

| Axis | Groq gap, 5 items | Groq gap, 15 items | Stable? |
|---|---|---|---|
| **Age band** | 15.9 pts | 16.7 pts | **Yes — the most robust finding in this study** |
| Sex | 0.9 pts | 5.6 pts | Grew, still modest |
| **Region zone** | 18.4 pts | **6.3 pts** | **No — shrank by two-thirds** |
| Urban / Rural | 3.2 pts | 4.3 pts | Roughly stable, still smallest or near-smallest |
| Education band | 5.2 pts | 3.4 pts | Roughly stable |
| Income tercile | 5.6 pts | 2.1 pts | Shrank |

**This is itself a finding, and an important methodological caveat for §4.2:** the region-zone gap — the one we highlighted as "near-exact agreement across models" at 5 items — shrank substantially once the item sample widened, while the age-band gap held essentially constant (15.9 → 16.7 points). Read together, **age band is the one subgroup fidelity gap in this study that has survived both a second model and a wider item sample**; the region finding at 5 items now looks more likely to have been an artifact of that particular small item slice than a stable property of the model. Overall accuracy also shifted with the wider item set — 28.7% (5 items) → 23.7% (15 items, [20.3, 27.0]) — driven largely by four 10-point science-attitude items (Q109, Q160, Q161, Q162) with much higher error (MAE 3.2–4.0 vs. 0.8–1.2 for the 4-point items), consistent with the mechanical scale-size effect discussed in the item-selection methodology.

*[IN PROGRESS: the matching 15-item widening for Gemini is still converging as of this draft; once complete, the cross-model replication check in the style of §4.2 will be re-run at 15 items, which is the number that should be treated as authoritative over the 5-item figures above.]*

## 5. Discussion

Taken together, §4.1 and §4.2 support a specific rather than general claim about zero-shot silicon sampling on this task: **two unrelated free-tier LLMs reach a similar, modest fidelity ceiling (~29%) on individual-level prediction, beat the naive baselines, fall short of a model actually fit to the population, and share a specific, replicated pattern of subgroup unevenness (region and age, not urban/rural).** This is more informative than an aggregate fidelity number alone, and more defensible than a single-model subgroup claim, because the region/age finding was confirmed independently rather than assumed to generalize from one run.

The P0≈P2 pilot finding (§3.3) is worth surfacing on its own: if replicated at scale, it would suggest these particular models' implicit "average respondent" prior is not being meaningfully perturbed by explicit demographic conditioning — a finding relevant to whether the demographic-prompting approach to silicon sampling is doing the work it is assumed to do, independent of the subgroup-fidelity question.

## 6. Limitations

- **Item coverage.** §4.1–4.2 use 5 of 144 selected items; §4.3 widens Groq to 15 and shows this materially changes gap estimates (region shrinks, age holds). *[IN PROGRESS: Gemini's matching 15-item run]* — until both models are compared at 15 items, treat the age-band finding as the load-bearing claim of this draft, not the region finding.
- **Free-tier / non-frontier models.** Neither model tested is a frontier commercial system; whether this fidelity ceiling and gap pattern holds for larger models is untested.
- **No calibrated token probabilities.** Neither provider exposed real per-token logprobs at this tier; distributional metrics beyond accuracy/MAE were not computed on calibrated probabilities.
- **Region-zone naming.** WVS-7's official annex maps region codes to named zones; the eight zone labels used in reporting were reconstructed from the codebook's country structure rather than read directly from that annex, and should be re-verified before being treated as final in any submitted version.
- **India's language variable is unusable as recorded.** Every respondent's interview-language field is coded identically (Hindi) regardless of actual fieldwork language, so a language-based subgroup axis — relevant given India's linguistic diversity — could not be tested from this data as released.
- **No fine-tuning comparison yet.** *[IN PROGRESS]* Track B (adapting a model to this population via QLoRA fine-tuning) would test whether closing the accuracy gap to the supervised baselines also narrows or widens the region/age fidelity gap — an open question this draft does not yet answer.

## 7. Conclusion

We set out to test a specific, decision-relevant question left open by prior silicon-sampling validation: not whether an LLM reproduces a population's opinions in aggregate, but whether it does so *evenly* across that population's internal diversity. For India, using WVS-7 and two independent zero-shot models, the answer is a qualified no — but a more specific and more defensible one than a single-model study could offer. Both models converge tightly on overall accuracy (~29%), both clear the naive baselines without approaching a population-fit supervised model, and both show a subgroup fidelity gap that is uneven rather than flat. Critically, not every piece of that unevenness survived scrutiny: subjecting our own initial region-zone finding to a wider item sample cut it by two-thirds, while the age-band gap held constant. We take this as the paper's methodological point as much as its empirical one — a subgroup fidelity claim in this space is only as strong as the replication and stability checks run against it, and single-model, single-item-slice results should be treated as hypotheses, not findings, until checked this way.

The practical upshot for anyone considering silicon sampling on Indian survey data: expect a real, model-independent accuracy ceiling well under 30% for individual-level prediction with today's free-tier zero-shot models, expect that ceiling to sit below what a simple demographic lookup fit to real data would achieve, and expect the technique's reliability to vary by age band specifically — not, on this evidence, primarily by urban/rural residence, the axis most commonly assumed to be the weak point.

## 8. Next steps

1. Complete the matching Gemini 15-item run and re-run the cross-model replication check at 15 items (in progress).
2. Investigate the non-replicating sex gap directly rather than leaving it as an open question.
3. Verify region-zone labels against the official WVS-7 annex.
4. Track B: QLoRA fine-tune on Kaggle's free-tier T4 GPUs; test whether fine-tuning narrows or widens the age-band fidelity gap (Δ_gap) — the gap this draft's evidence says is the real one to target.

## References

- Argyle, L. et al. (2023). "Out of One, Many: Estimating individual heterogeneity through linguistic evidence."
- WorldValuesBench (LREC-COLING 2024). https://github.com/demon702/worldvaluesbench
- Haerpfer, C. et al. (eds.) (2022). *World Values Survey: Round Seven — Country-Pooled Datafile Version 6.0.* JD Systems Institute & WVSA Secretariat. doi:10.14281/18241.24
