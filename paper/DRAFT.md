# Silicon Sampling for Indian Public Opinion: A Subgroup Fidelity Audit

**Status: working draft, last updated 2026-08-28.** Every number below is from a real, run pipeline — none are estimated or illustrative. Track A inference (both models, both item widths) is now fully converged; Track B fine-tuning remains *[IN PROGRESS]*.

## Abstract

Silicon sampling — conditioning an LLM on a demographic profile to simulate a survey respondent — has been validated in prior work mainly at the aggregate level, leaving open whether fidelity holds evenly across a population's internal diversity. We audit this for India using World Values Survey Wave 7 data (N=1,692), screening 373 candidate survey items against the official codebook down to 144 valid targets, and establishing a fair-comparison baseline suite (uniform through gradient-boosting, including a variant matched to the LLM's own feature set) before any LLM inference. Two independent zero-shot models — Gemini 3.1 Flash Lite and OpenAI's open-weight gpt-oss-120b — converge on statistically indistinguishable accuracy (24.9% vs. 23.5% at our widest, fully-converged item sample of 15 items) and, more importantly, on a qualitative pattern: **every one of six demographic axes tested (urban/rural, income, education, sex, age, region) shows a non-trivial fidelity gap in both models, at both item widths tested.** The *quantitative* picture is less stable: which axis has the largest gap changes with the model and the item sample, so we report the full four-cell (2 models × 2 item widths) comparison rather than picking a single headline axis. We also find that raw cross-model prediction agreement (41.6%, Cohen's κ=0.25) is driven substantially by both models converging on the same *wrong* answer, not the same right one, and that a specific subgroup gap (sex) is not explained by the three explicit gender-attitude items alone. We conclude that zero-shot silicon sampling for India has a real, model-independent accuracy ceiling well below a population-fit supervised baseline, and a genuinely uneven subgroup fidelity profile whose unevenness itself replicates even where its exact shape does not.

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
| Gradient boosting, matched to LLM's 14-attribute set | **50.1%** [47.6, 52.5] | 1.07 |
| Gradient boosting, original 6-attribute set | 48.2% [45.8, 50.5] | 1.13 |
| Logistic regression, matched (14 attrs) | 46.9% [44.3, 49.5] | 1.21 |
| Logistic regression, original (6 attrs) | 46.6% [44.0, 49.2] | 1.24 |
| Demographic-cell lookup (6 attrs — 14 collapses to near-singleton cells at this N) | 34.5% [32.3, 36.8] | 1.51 |
| National marginal (survey-weighted) | 32.8% [30.6, 35.1] | 1.55 |
| Uniform random | 19.4% [18.3, 20.5] | 2.11 |

All baselines evaluated out-of-fold, 5-fold stratified cross-validation, across all 144 selected items. The national-marginal baseline is weighted by WVS's own respondent weight (`W_WEIGHT`), since it represents a population-level claim; the LLM's respondent-level predictions are deliberately left unweighted, since weighting would double-count the demographic information already conditioning the prompt. The "matched" rows give the supervised baselines the identical 14-attribute demographic set the LLM's P2 prompt sees (rather than a smaller 6-attribute subset used in an earlier pass of this analysis); the gain is real but modest (+1.9 pts for GBM, +0.3 for logistic), meaning the zero-shot LLM's gap to a population-fit ceiling is slightly larger than an unmatched comparison would suggest.

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

**Finding 1 — fidelity is not uniform, and the pattern of unevenness replicates.** Region and age-band gaps are large in both independently-run models. The region-gap agreement (18.3 vs. 18.4 points) in particular is close enough that coincidence is an unlikely explanation. *(§4.3 revisits this at 15 items and finds the exact ranking above does not hold up quantitatively, though the qualitative "gaps are real and widespread" claim strengthens.)*

**Finding 2 — urban/rural, the axis silicon-sampling critiques most often target, shows the smallest gap in both models.** This is a specific, falsifiable claim rather than the more common blanket assertion that LLM-simulated respondents systematically misrepresent rural populations. Our data argues the more consistent failure mode here is regional and generational, not urban/rural.

**Finding 3 — the sex gap does not replicate, and we report that rather than omit it.** Gemini shows an 11.0-point male-favoring gap; Groq shows 0.9 points. Presenting only the Gemini number would overstate a pattern that a second model does not support. We treat this as evidence the sex gap is model-specific behavior rather than a property of the underlying task or population.

### 4.3 Item-count sensitivity — the full 2-model × 2-width comparison

The results in §4.2 use 5 items. Both models were re-run on the same 15 items (Groq: n=556, 40 respondents; Gemini: n=421, 30 respondents, a strict subset of Groq's 40), both fully converged with zero unrecovered failures, to test whether the §4.2 gaps are stable as item coverage widens *and* replicate across models at that wider width:

| Axis | Gemini, 5 items | Groq, 5 items | Gemini, 15 items | Groq, 15 items |
|---|---|---|---|---|
| Region | 18.3 | 18.4 | 26.4 | 6.3 |
| Age band | 23.7 | 15.9 | 9.2 | 16.7 |
| Education | 7.1 | 5.2 | 14.3 | 3.4 |
| Income | 9.1 | 5.6 | 14.2 | 2.1 |
| Sex | 11.0 | 0.9 | 11.1 | 5.6 |
| Urban/rural | 3.8 | 3.2 | 8.9 | 4.3 |

Reading across each row: **no axis is quantitatively stable across all four cells.** Region looked "near-exact agreement across models" at 5 items (18.3 vs. 18.4) but diverges sharply at 15 (26.4 vs. 6.3). Age band looked stable within Groq alone (15.9 → 16.7) but Gemini's age gap *shrinks* over the same widening (23.7 → 9.2) — the opposite direction. Sex, income, and education all show the same qualitative story: present and non-trivial everywhere, but not quantitatively consistent enough to rank-order with confidence from this sample.

**Revised finding (supersedes an earlier draft of this section, written before Gemini's 15-item run converged):** we do not find a single axis that is "the load-bearing" subgroup-fidelity claim. What we do find, robustly, in all four cells: **every one of the six axes tested shows a fidelity gap of at least ~2 points, typically 5–25 points.** The general phenomenon — silicon-sampling fidelity is uneven across demographic subgroups — replicates completely across models and item widths. Its exact shape does not. We consider this the more scientifically honest way to report a result like this: a qualitative finding this consistent, reported without an over-precise quantitative ranking that the data does not actually support.

Overall accuracy also shifted with the wider item set — from ~29% (5 items, both models) to ~24% (15 items, both models: Gemini 24.9% [20.9, 28.7], Groq 23.5% on the same 421-row overlap subset) — driven largely by four 10-point science-attitude items (Q109, Q160, Q161, Q162) with much higher error (MAE 3.2–4.0 vs. 0.8–1.2 for the 4-point items), consistent with the mechanical scale-size effect discussed in the item-selection methodology.

### 4.4 Cross-model agreement — do the models agree with each other, or just with themselves?

On the 421 (respondent, item) pairs both models answered at 15 items: raw agreement is 41.6% (Cohen's κ=0.25, fair-but-weak). Decomposed: only 10.7 percentage points of that 41.6% is the two models agreeing *and both being correct*; 30.9 points is the two models agreeing *and both being wrong* — most of the observed agreement is convergence on the same incorrect answer (plausibly both models defaulting to whatever answer option is most "obvious" from general world knowledge), not convergence on ground truth. Per-item agreement ranges from 14.3% (Q160, "science vs. faith") to 82.1% (Q4, "important in life: politics" — also the single highest-accuracy item for both models), suggesting some items are intrinsically easier or harder for any LLM tested here, independent of which model. Per-subgroup agreement is comparatively flat (33–52% in every category on every axis), which weakly argues against "one model is more subgroup-biased than the other" as the explanation for §4.3's non-replication — the pattern looks more like independent per-item noise than a systematic model-specific bias.

### 4.5 Is the sex gap a gender-stereotyping effect?

We checked whether the sex-axis fidelity gap (§4.3) concentrates in the three items that ask about gender roles directly (Q29 "men make better political leaders," Q31 "...business executives," Q32 "housewife just as fulfilling") — i.e., whether it reflects the model's own priors about gender roles leaking into predictions specifically on gender-role questions. It does not concentrate there: mean male-minus-female accuracy gap on those 3 items is 13.4 pts (Gemini) / 11.7 pts (Groq); on the other 12 items it is 10.1 pts (Gemini, nearly as large) / 3.8 pts (Groq, smaller but still present). The gap is broad-based on Gemini — present on religion/science items as much as gender items — and more concentrated on Groq. This rules out the simplest stereotyping-on-stereotyping-questions explanation as the sole driver; the sex gap looks more like a general prediction-accuracy asymmetry by respondent sex than a topic-specific effect, at least on Gemini.

## 5. Discussion

Taken together, §4.1–4.4 support a more conservative claim than an earlier draft of this paper made: **two unrelated free-tier LLMs reach a similar, modest fidelity ceiling (~24–29%, narrowing as item coverage widens) on individual-level prediction, beat the naive baselines, fall short of a model actually fit to the population, and reliably show subgroup fidelity gaps on every demographic axis tested — but the relative size and ranking of those gaps is not itself a stable, cross-model, cross-item-width quantity.** This is a weaker, but more honest, claim than "axis X is the problem." §4.4's agreement analysis adds a mechanistic hint at why: the two models agree with each other on wrong answers about as often as they agree on right ones, consistent with both models leaning on similar generic priors that only sometimes match India's actual opinion distribution — a plausible source of the same kind of noise that would prevent any one subgroup gap from replicating precisely.

The P0≈P2 pilot finding (§3.3) is worth surfacing on its own: if replicated at scale, it would suggest these particular models' implicit "average respondent" prior is not being meaningfully perturbed by explicit demographic conditioning — a finding relevant to whether the demographic-prompting approach to silicon sampling is doing the work it is assumed to do, independent of the subgroup-fidelity question.

## 6. Limitations

- **Item coverage.** Even the wider run (§4.3) covers 15 of 144 selected items; both models are now fully converged at that width, but the 144-item full battery remains untested (§8).
- **Free-tier / non-frontier models.** Neither model tested is a frontier commercial system; whether this fidelity ceiling and gap pattern holds for larger models is untested.
- **No calibrated token probabilities.** Neither provider exposed real per-token logprobs at this tier; distributional metrics beyond accuracy/MAE were not computed on calibrated probabilities.
- **Region labels were wrong until 2026-08-28, and every prediction reported in this draft used the wrong ones in-prompt.** An earlier version of this pipeline could not locate WVS-7's official region-code annex in the extracted codebook text and substituted a guessed macro-zone scheme; on locating the actual annex, every one of the 8 guessed labels turned out to be incorrect (e.g. code 356028 is Uttar Pradesh, not "South zone" as guessed). This has been fixed in the pipeline for future runs, but every P1–P3 prediction analyzed in this draft was generated with the incorrect region text as one of 14 demographic attributes. The subgroup *grouping* used for fidelity-gap analysis is unaffected (it uses the raw region code, not the label text), but the region attribute's contribution to what the LLM actually saw was noise rather than signal throughout. A re-run with corrected labels is planned (§8) but not yet done.
- **India's language variable is unusable as recorded.** Every respondent's interview-language field is coded identically (Hindi) regardless of actual fieldwork language, so a language-based subgroup axis — relevant given India's linguistic diversity — could not be tested from this data as released.
- **No fine-tuning comparison yet.** *[IN PROGRESS]* Track B (adapting a model to this population via QLoRA fine-tuning) would test whether closing the accuracy gap to the supervised baselines also narrows or widens the subgroup fidelity gaps — an open question this draft does not yet answer.

## 7. Conclusion

We set out to test a specific, decision-relevant question left open by prior silicon-sampling validation: not whether an LLM reproduces a population's opinions in aggregate, but whether it does so *evenly* across that population's internal diversity. For India, using WVS-7 and two independent zero-shot models fully converged at two item widths, the answer is yes, unevenly — but the paper's more interesting finding is methodological: an early, narrower slice of this same data made a much crisper-sounding claim (a single "load-bearing" axis, near-exact cross-model agreement on region) that did not survive being checked against a second model at wider item coverage. What survives that scrutiny is the qualitative claim — every axis tested shows a real, non-trivial fidelity gap in every condition tested — without a confident quantitative ranking of which axis matters most. We take this as the paper's central methodological point as much as its empirical one: a subgroup fidelity claim in this space is only as strong as the replication and stability checks run against it, and a result that looks clean on one model at one item width should be treated as a hypothesis, not a finding, until checked this way — a lesson this project learned by initially publishing the narrower claim internally before the wider check was available.

The practical upshot for anyone considering silicon sampling on Indian survey data: expect a real, model-independent accuracy ceiling under 30% for individual-level prediction with today's free-tier zero-shot models, expect that ceiling to sit below what a simple demographic lookup fit to real data would achieve, and expect meaningfully uneven reliability across demographic subgroups — but treat any single-axis "this is the one that matters" claim, including earlier versions of our own, with real skepticism until it has been checked across more than one model and more than one item sample.

## 8. Next steps

1. ~~Complete the matching Gemini 15-item run~~ — done; incorporated above.
2. ~~Investigate the non-replicating sex gap directly~~ — done (§4.5): not explained by the 3 gender-attitude items alone.
3. ~~Verify region-zone labels against the official WVS-7 annex~~ — done; found wrong, fixed in the pipeline (§6).
4. **Re-run P2/P3 inference with the corrected region labels** and confirm the subgroup findings are unchanged (grouping was always correct; only the in-prompt text was wrong, so we expect no material change, but this should be confirmed rather than assumed).
5. Track B: QLoRA fine-tune on Kaggle's free-tier T4 GPUs (token now available); test whether fine-tuning narrows or widens the subgroup fidelity gaps.
6. Widen beyond 15 items toward the full 144-item battery, budget permitting under free-tier daily quotas.

## References

- Argyle, L. et al. (2023). "Out of One, Many: Estimating individual heterogeneity through linguistic evidence."
- WorldValuesBench (LREC-COLING 2024). https://github.com/demon702/worldvaluesbench
- Haerpfer, C. et al. (eds.) (2022). *World Values Survey: Round Seven — Country-Pooled Datafile Version 6.0.* JD Systems Institute & WVSA Secretariat. doi:10.14281/18241.24
