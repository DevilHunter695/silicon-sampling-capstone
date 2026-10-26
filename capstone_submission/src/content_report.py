TITLE = "Silicon Sampling for Indian Public Opinion"
SUBTITLE = "How do LLM respondents use demographics? Prompting, fine-tuning and held-out-item transfer"
GROUP = "55"
STUDENTS = [("Bhanuj Bhalla", "70022300017"), ("Ishaan Bansal", "70022300011"), ("Yash Bhende", "70022300137"), ("Krishank Ayyella", "70022300136")]
MENTOR = "Ms. Khushbu Chauhan"
B = []
def H1(t): B.append(("h1", t))
def H2(t): B.append(("h2", t))
def P(t): B.append(("p", t))
def BUL(items): B.append(("bul", items))
def TAB(cap, rows, widths): B.append(("table", cap, rows, widths))
def FIG(path, cap, w=14.0): B.append(("fig", path, cap, w))
def BOX(cap, text): B.append(("box", cap, text))

ABSTRACT = ("Silicon sampling asks a large language model (LLM) to answer a survey as a respondent with a given demographic profile. If it worked, it would be far cheaper than fielding a questionnaire, "
 "but an average accuracy can hide unequal performance across demographic groups, which matters in a country as diverse as India. This project builds and evaluates a complete silicon-sampling pipeline for India on two public surveys, "
 "Pew Research Center's Religion in India (N = 29,999) and the World Values Survey wave 7 (WVS-7) for India (N = 1,692), and scores every method on identical respondent-item rows against matched classifiers. "
 "The first phase used free-tier models (Gemini 3.1 Flash Lite and gpt-oss-120b) on WVS-7 and found accuracy of 22 to 26% on 15 items, at or below the majority class, with subgroup gaps that did not replicate across models. "
 "The second phase studied Qwen2.5-7B-Instruct on the Pew survey. Zero-shot, the model reacts strongly to the demographic profile (the profile explains 29% of its prediction variance against 4% for real answers) but uses it weakly, with a correlation of 0.22 between predicted and true group differences, and removing the profile raises accuracy by 3.6 points. "
 "After QLoRA fine-tuning on 184,853 respondent-level examples the model reaches 38.0% on 6,000 held-out respondents (majority class 30.4%, logistic regression 38.5%, gradient boosting 40.0%), closes 79% of the gap to the best classifier, and is well calibrated (expected calibration error 0.009). "
 "An approximate re-implementation of the adapter shows no detectable gain on eight unseen questions (28.9% against 28.5%). We also report measured computational cost, scaling estimates, and the engineering lessons learned. "
 "A paper from this work was accepted at the Hinweis Fifth International Conference on Artificial Intelligence and Data Science (AIDE-2026, paper ID 591).")
KEYWORDS = "silicon sampling, survey simulation, large language models, QLoRA, calibration, demographic conditioning, computational social science, Pew Research Center, World Values Survey, India"

ACK = ("We thank our project mentor, Ms. Khushbu Chauhan, for her guidance, regular feedback and encouragement throughout this project. "
       "We are grateful to the Capstone and Technical Project Committee of the Computer Engineering Department, Mukesh Patel School of Technology Management & Engineering, NMIMS, for the opportunity and the structure of the capstone programme. "
       "We thank the university's AI Lab for GPU access used for fine-tuning, and the Pew Research Center and the World Values Survey Association for making their survey data publicly available for research.")

H1("1. INTRODUCTION")
H2("1.1 Background and Motivation")
P("Public opinion surveys are expensive and slow. A national face-to-face survey needs thousands of interviews, trained interviewers and months of fieldwork. Silicon sampling offers a shortcut: give a large language model a demographic profile (for example, a rural woman aged 35 to 44 from eastern India), put a survey question to it, and treat its answer as that person's answer. Repeated over a synthetic population, this would give opinion estimates in minutes and at a tiny fraction of the cost of fieldwork [1].")
P("The idea is attractive, but the evidence is mixed. Language models can reproduce some demographic-attitude correlations in United States data [1], yet they misrepresent many groups [2]. At the level of individual respondents they match or trail simple statistical baselines trained on the same demographics, over-attribute attitudes to demographics and inflate the gaps between groups [3], [4]. Persona variables explain little of the real variance in human answers [5]. Almost all of this work concerns the United States or cross-national averages.")
P("India makes the question sharper. It is among the most diverse countries in the world along region, religion, caste, language, income and age. A model that matches the national average but fails for particular states, religions or income bands would give misleading answers exactly where inclusive policy-making needs the most care. Aggregate accuracy alone cannot reveal that failure.")
H2("1.2 Problem Statement")
P("We need to know, for India, what the demographic profile actually contributes to an LLM's survey answers, whether supervised adaptation on respondent-level data changes that, whether any gain carries over to questions the model never saw, and what all of this costs compared with a plain statistical classifier. The problem is not to maximise a single accuracy number. It is to measure, honestly and on identical data, how well LLM respondents represent different groups of Indian respondents, and to report where they fail.")
H2("1.3 Objectives and Research Questions")
BUL(["**RQ1 (use).** How does a prompted LLM use the demographic profile: how much, and in which direction?",
     "**RQ2 (adaptation).** What does respondent-level fine-tuning change, relative to the zero-shot model and to matched classifiers?",
     "**RQ3 (transfer).** Does the adaptation transfer to questions the model never saw?",
     "**RQ4 (scale and data).** Do a larger zero-shot model or more labelled respondents change the picture?"])
P("Alongside these, the project has an engineering objective: to build a reproducible, cached and auditable pipeline, so that every number in this report can be regenerated from code and data.")
H2("1.4 Scope")
P("The study uses two public surveys. WVS-7 India supports a first phase with free-tier API models. The Pew Religion in India survey supports the main phase: zero-shot controls, QLoRA fine-tuning of Qwen2.5-7B-Instruct, scoring on a GPU, diagnostics, a held-out-item transfer test and a cost analysis. All fine-tuning results use one fold, one seed and one epoch. Frontier commercial models are out of scope. Survey weights are not applied, so statements describe the sample, not the Indian population.")
H2("1.5 Contributions")
BUL(["A reproducible evaluation framework that scores zero-shot, fine-tuned, larger and classical predictors on identical rows (Chapter 4).",
     "Profile-use diagnostics that separate accuracy from mechanism: a profile ablation, the demographic variance share, and the directional validity of group differences (Section 4.6).",
     "Evidence from India that a zero-shot 7B model uses the profile heavily but with weak direction, and that fine-tuning recovers most, though not all, of the gap to matched classifiers (Chapter 6).",
     "A held-out-item transfer test, reported with its limits, and measured computational cost with scaling estimates (Chapters 5 and 6).",
     "A conference paper accepted at AIDE-2026 (Hinweis Research, paper ID 591)."])
H2("1.6 Organisation of the Report")
P("Chapter 2 reviews the literature. Chapter 3 describes the data. Chapter 4 presents the system and evaluation framework. Chapter 5 covers implementation and computational requirements. Chapter 6 reports experiments and results. Chapter 7 discusses the findings and their limits, and Chapter 8 concludes with future work. The appendices give the prompt, the item lists, the repository map and a log of the experiments.")

H1("2. LITERATURE REVIEW")
H2("2.1 Silicon Sampling and Algorithmic Fidelity")
P("Argyle et al. [1] showed that GPT-3 conditioned on sociodemographic backstories reproduces demographic-attitude correlations in United States data, which they called algorithmic fidelity. Santurkar et al. [2] compared model opinions with those of many US demographic groups and found substantial misalignment that explicit steering does not remove. Durmus et al. [6] found that models resemble the opinions of some countries more than others, and Bisbee et al. [7] documented reduced variance and sensitivity to prompt wording. Shankar et al. [8] audited religious viewpoints across Asian populations and found minorities poorly represented.")
H2("2.2 Baselines and Over-Determination")
P("Miranda and Balbi [4] found that random forests perform comparably to LLMs at the individual level on the American National Election Studies, with LLMs matching aggregate distributions better. Chen, Zhu and Zheng [3] evaluated four models on the General Social Survey and the 63-country World Values Survey against marginal, demographic-lookup, logistic and random-forest baselines. No model beat the strongest baseline, models over-determined attitudes by demographics, and segment gaps were inflated two- to fourfold. Hu and Collier [5] showed that persona variables explain under 10% of the variance in annotation data.")
H2("2.3 Adapting Models with Survey Data")
P("Suh et al. [9] fine-tuned models on a large collection of survey data and found generalisation to unseen subpopulations and surveys. Cao et al. [10] specialised models to survey response distributions across countries and found gains on unseen questions, countries and a new survey, but least on unseen questions. Krsteski et al. [11] found that statistical rectification can be worth more than fine-tuning when human data are scarce, and Kim and Lee [12] trained respondent embeddings over a frozen model.")
H2("2.4 Recent Work in 2026")
P("Several 2026 reports reach similar conclusions on other populations. Wang [13] found negligible individual-level recovery on the European Social Survey, where an average of neighbouring countries beat every model configuration. Lam et al. [14] found that frontier models give moderately faithful means but understate variance in Singapore. Robbani [15] found, for Mistral-7B on US panel data, that demographic information is readable in internal representations but barely changes predictions. Choi et al. [16] found that fine-tuning on small pilot samples gives balanced structural and marginal fidelity on a Korean survey.")
H2("2.5 Research Gap")
P("The literature establishes that LLM respondents often match or trail matched baselines and over-determine attitudes by demographics. What is missing is an India-specific study that (a) separates what the profile contributes from the model's prior, (b) compares a fine-tuned model with its own zero-shot version on the same rows, (c) tests whether adaptation transfers to questions the model never saw, and (d) reports the computational cost. This project addresses those four points.")

H1("3. DATA")
H2("3.1 World Values Survey Wave 7, India")
P("The WVS-7 cross-national release, version 6.0, was used. India's fieldwork finished in July 2023, after version 5.0 shipped, so earlier releases contain no India rows. The file has 97,220 respondents in 66 countries, with exactly 1,692 from India, and 613 columns [17]. The WVS Association's terms forbid redistributing the data, so the raw file, the codebook and the respondent-level cleaned file are git-ignored; only derived, non-microdata artefacts are in the repository.")
P("Cleaning recoded the survey's missing-value codes (-1 to -5) to missing across all numeric columns, which affected 6,887,680 cells, and filtered to India. Overall missingness after recoding is 18.9%. The official 404-page codebook was parsed into structured metadata for 353 questions, and every question column (373 candidates) was screened against it in a fixed order.")
TAB("Table 3.1. WVS-7 item screening (first failure wins).",
    [["Screening step", "Rule", "Items rejected"],
     ["No codebook entry", "Column has no matching codebook entry", "20"],
     ["Demographic block", "Q260-Q290 condition the prompt; using one as a target would leak the answer", "40"],
     ["Derived variant", "WVS-shipped recode or duplicate of another item", "13"],
     ["Manually excluded", "Binary factual items, unordered menus, eligibility codes", "9"],
     ["Scale too small", "Fewer than 4 response options", "83"],
     ["Non-ordinal codes", "Codes not a contiguous integer run", "1"],
     ["High missingness", "More than 10% missing", "63"],
     ["Passed all screens", "", "144"]], [4.0, 8.0, 2.5])
P("373 candidates minus 229 rejected leaves 144 items. Item metadata for all 144 is stored in a committed file so the selection is auditable. India's interview-language variable is constant (Hindi) for all 1,692 respondents, so no language-based subgroup analysis is possible.")
H2("3.2 Pew Religion in India")
P("The Pew Research Center survey Religion in India [18] interviewed 29,999 adults face to face between November 2019 and March 2020. Question wording and labels were parsed from the survey's DDI metadata, and 73 items passed screening (binary items, demographic columns, items with more than 10% missing answers and near-unanimous items were excluded). The first eight items of the screened list are used for fine-tuning and evaluation: number of children, reading scripture, politicians' role in religion, attending worship, abortion legality, sex-selective checkups, watching religious television and the nature of religious identity. Items 9 to 16 (meditation, self-rated health, trust in health advice, what it takes to be truly Indian, television serials and praying) are never used for training and form the transfer set. Each has 4 to 8 answer options.")
H2("3.3 Splits")
P("Respondents are split into five folds, stratified jointly on urban/rural and income band, with a fixed random seed of 42. All scoring is out-of-fold, and no respondent appears in both training and test data. For Pew this report uses fold 0: 23,999 training and 6,000 test respondents, which gives 46,209 test answers on the eight trained items. The Pew prompt carries nine attributes (sex, age band, marital status, education, religion, caste, urban/rural, region, income band). The WVS prompt carries 14.")

H1("4. SYSTEM DESIGN AND EVALUATION FRAMEWORK")
H2("4.1 Architecture")
P("Figure 4.1 shows the system. Both surveys are verbalised into the same prompt, passed to four kinds of predictors, read out as next-token probabilities, and scored on identical rows with the same diagnostics. The key design rule is that every predictor sees the same prompt and the same rows, so any difference between methods is a difference in the method.")
FIG("fig_framework.png", "Figure 4.1. Evaluation framework: data, prompt, predictors, and readout with diagnostics.", 14.5)
H2("4.2 Prompt Construction and Verbalisation")
P("Each respondent-item pair becomes one prompt (called P2): a sentence listing the respondent's profile, the question text, the numbered answer options and an instruction to reply with a single digit. Every demographic string and every answer option is built programmatically from the same codebook metadata used for item selection, so there is no hand-typed text. Conditions studied are P0 (no profile), P1 (age, sex, region), P2 (full structured profile, the primary condition) and P3 (the same information as a first-person backstory). The closing instruction is identical for every model; an earlier weaker wording produced a 13% refusal rate, and the final wording brought it to 0%.")
H2("4.3 Readout")
P("The model's next-token distribution over the option digits gives the probabilities, and its argmax gives the hard prediction. This readout is used for every zero-shot condition and for the fine-tuned model, so comparisons are like for like. Responses that cannot be parsed are recorded as refusals, never silently coerced into guesses.")
H2("4.4 Baselines and Controls")
P("Matched baselines are fit per item on the training respondents with exactly the attributes in the prompt: a majority class, logistic regression and histogram gradient boosting (GBM), plus a demographic-cell lookup for WVS. They are scored on the same rows as the LLMs. Zero-shot controls are *chat* (the model's chat template), *raw* (the bare prompt, the fine-tuning format), *no profile* (the same prompt for every respondent) and *prior-corrected* (probabilities rescaled by the training marginal over the model's own mean prediction, which needs no test labels).")
H2("4.5 Fine-Tuning")
P("The adapter was trained with QLoRA [19], [20]: a 4-bit NF4 base model, rank 16, alpha 32, dropout 0.05 on the query, key, value and output projections, giving 10,092,544 trainable parameters out of 4.36 billion (0.23%). Training ran for one epoch over 184,853 examples from 23,999 respondents, with an effective batch of 32, peak learning rate 2e-4, seed 42 and the loss over the whole sequence. Qwen2.5-7B-Instruct [21] is the base model.")
H2("4.6 Diagnostics")
BUL(["**Accuracy, NLL and ECE.** Exact match, negative log-likelihood and expected calibration error, with respondent-clustered bootstrap confidence intervals.",
     "**Demographic variance share.** The increase in R-squared when profile dummies are added to item fixed effects, for predictions and for true answers.",
     "**Directional validity.** For each (item, axis, category) cell, the category mean minus the item mean, for predictions and for truth; we report their correlation r and the slope of predicted on true.",
     "**Distributional fidelity.** The Jensen-Shannon divergence between the mean predicted distribution and the empirical answer distribution in each subgroup cell.",
     "**Subgroup gaps.** Best minus worst category accuracy, tested by label permutation with Holm correction across axes."])
H2("4.7 Transfer and Data-Efficiency Protocols")
P("For transfer, the adapter and its base are applied to the eight unseen items for 120 respondents, and the difference between the in-domain gain and the unseen-item gain is bootstrapped. For the data requirement, classical models are trained on nested respondent subsets (25 to 23,999 respondents, five random subsets each) and scored on the same 200 test respondents.")

H1("5. IMPLEMENTATION AND COMPUTATIONAL REQUIREMENTS")
H2("5.1 Technology Stack")
TAB("Table 5.1. Technology stack.",
    [["Layer", "Tools"],
     ["Data and baselines", "Python, pandas, pyarrow, scikit-learn"],
     ["Fine-tuning", "Hugging Face Transformers, PEFT (LoRA), bitsandbytes (NF4), TRL"],
     ["Local inference", "MLX and mlx-lm, 4-bit Qwen2.5-7B-Instruct, Apple M1 with 16 GB"],
     ["GPU scoring", "Kaggle notebooks, two Tesla T4 GPUs, fp16"],
     ["API models", "Groq (gpt-oss-120b), Google AI Studio (Gemini 3.1 Flash Lite)"],
     ["Reproducibility", "Git, fixed seeds, committed JSON results, scripted tables and figures"]], [4.0, 10.5])
H2("5.2 Repository Structure")
P("The code is organised so that data handling, prompting, inference and evaluation are independent modules. `src/data` loads and screens the surveys and builds folds. `src/prompts` verbalises profiles and items. `src/inference` contains the cached runner and the provider integrations. `src/eval` holds the metrics, baselines, bootstrap and subgroup analysis. The Pew fine-tuning scripts, the Kaggle scoring and analysis scripts, and the paper sources live alongside. The repository is at github.com/DevilHunter695/silicon-sampling-capstone.")
H2("5.3 Phase 1: Free-Tier Models on WVS-7")
P("The first phase ran Gemini 3.1 Flash Lite and gpt-oss-120b (via Groq) at temperature 0. Every prediction is cached to disk before anything else touches it, so a killed run resumes without repeating API calls. Pacing respects each provider's rate limit (Groq 1,000 requests a day, Gemini about 500). Both models spend output tokens on internal reasoning before the visible answer, so token budgets were set empirically.")
H2("5.4 Phase 2: Local Inference on Apple Silicon")
P("Qwen2.5-7B-Instruct was run locally in 4-bit MLX on a 16 GB laptop. Per-respondent prefix caching reuses the profile prefix across a respondent's items, giving 2.3 seconds per Pew prompt and 1.7 seconds per WVS prompt, about 5 hours per 1,000 respondents on eight items. A memory cache limit stopped the system swapping to disk.")
H2("5.5 Phase 3: Fine-Tuning and GPU Scoring")
P("The QLoRA run was done on a university lab GPU on 29 September 2026: 5,777 optimizer steps and 396.7 minutes of training (6.6 hours), 482 minutes in total including inference. Its own greedy decoding scored 36.1%. Because that decoding ended the prompt without the separator used in training, we rescored the same adapter on Kaggle T4 GPUs in the training format, taking the next-token distribution over the option digits. This faithful readout scores 38.0% and gives real probabilities for calibration and distributional analysis. Scoring all 46,209 rows for one condition took about 75 minutes on two T4 GPUs working in parallel.")
H2("5.6 Computational Requirements and Scalability")
TAB("Table 5.2. Hardware and measured cost.",
    [["Task", "Hardware", "Measured cost"],
     ["QLoRA fine-tuning, 184,853 examples", "One lab GPU", "5,777 steps, 6.6 h (396.7 min)"],
     ["Scoring one condition, 46,209 rows", "Two Kaggle T4, fp16", "About 75 min; 0.19-0.21 s per row per GPU"],
     ["Local zero-shot scoring", "Apple M1, 16 GB, 4-bit MLX", "2.3 s per prompt with prefix caching; about 5 h per 1,000 respondents"],
     ["Matched classical baselines, 8 items", "Laptop CPU", "Under 1 min for the whole fold"],
     ["gpt-oss-120b, 774 rows", "Free API tier", "Limited by rate limits; several hours"]], [5.6, 3.9, 5.0])
P("Cost grows linearly with the number of respondent-item rows, because every row needs one forward pass over a prompt of about 230 tokens, roughly 2 x 7.6 billion x 230 = 3.5 TFLOP for a 7B model. Training cost grows linearly with the number of examples. It does not depend on the number of subgroups. Scoring all 29,999 Pew respondents on all 73 screened items (about 2.2 million rows) would take roughly 116 GPU-hours on a T4 at the measured rate, or about 58 hours on two. Fine-tuning on all 73 items (about 1.75 million examples) would take roughly 63 hours at the lab GPU's rate and about 290 hours at the T4 rate. These are linear extrapolations, not measurements. A classifier, by contrast, trains in seconds per item and predicts a row in microseconds, several orders of magnitude cheaper. Cost can be cut by caching the profile prefix, by batching and by using smaller or distilled models, although we did not test the last of these.")
H2("5.7 Engineering Problems Found and Fixed")
P("Several problems were found during the project and fixed, and some changed results. The WVS region labels in the prompt were reconstructed from the extracted codebook text and turned out to be wrong on every code; checking against the official annex and re-running both models moved subgroup gaps by several points (for example Gemini's urban/rural gap fell from 8.9 to 4.3). The item-selection code originally rejected 354 of 373 candidates because of a data-type bug. A cache bug had treated transient API failures as final answers. A model name containing a slash broke cache file paths. Importing the inference package eagerly pulled in heavy libraries and crashed lightweight environments. The Pew run's apparent collapse to a constant answer on one item was traced to the original greedy decoding, not the adapter. Fifteen bugs in all were logged and fixed (Appendix D).")

H1("6. EXPERIMENTS AND RESULTS")
H2("6.1 Phase 1: WVS-7 India with Free-Tier Models")
P("Baselines were established before any LLM inference. Out of fold across all 144 items and five folds, gradient boosting reaches 48.2% (50.1% with the full 14 prompt attributes), logistic regression 46.6% (46.9% matched), the demographic-cell lookup 34.5%, the survey-weighted national marginal 32.8%, and uniform guessing 19.4%.")
TAB("Table 6.1. WVS-7 India, 15 items: zero-shot LLM accuracy and baselines on the same rows.",
    [["Row sample", "Gemini 3.1 Flash Lite", "gpt-oss-120b", "Qwen2.5-7B"],
     ["Respondents / rows", "30 / 421", "40 / 556", "120 / 1,716"],
     ["LLM accuracy (%)", "25.7 [20.8, 30.5]", "23.0 [19.6, 26.7]", "22.1 [20.0, 24.4]"],
     ["Majority class", "20.9", "22.3", "24.9"],
     ["GBM (14 attributes)", "35.2", "34.4", "37.5"]], [4.0, 3.6, 3.4, 3.4])
P("All configurations land in a narrow band. Over the 144 items the majority class reaches 45.5%, so the 15 evaluated items are harder than average and each LLM is compared with the majority class on its own rows. The profile did not help: on WVS the profile changes Qwen's accuracy by -1.3 points (-4.2 to +1.5). On seven of the 15 items the model gives the same answer to nearly every respondent, and its group differences show no directional validity (r = -0.03 over 345 cells, against 0.64 for GBM). Subgroup gaps could not be told apart from random group assignment in 11 of 12 tests. Across the two API models the ranking of gap sizes changed with the model and the number of items, so no axis had a stable quantitative ranking. Cross-model agreement on the 421 shared rows was 36.8% with Cohen's kappa 0.18.")
H2("6.2 RQ1: Zero-Shot Qwen2.5-7B on Pew")
TAB("Table 6.2. Pew, 200 held-out respondents (1,548 rows).",
    [["Predictor", "Accuracy (%) [95% CI]", "NLL", "Dir. r", "Profile var."],
     ["True answers", "-", "-", "-", "4.0%"],
     ["Majority class (uses labels)", "30.2 [28.1, 32.4]", "1.59", "-", "-"],
     ["Zero-shot, no profile", "21.1 [19.1, 23.1]", "3.92", "-", "0.0%"],
     ["Zero-shot, profile (chat)", "17.5 [15.7, 19.4]", "3.88", "0.22", "29.4%"],
     ["Zero-shot, prior-corrected", "26.3 [24.0, 28.6]", "-", "-", "-"],
     ["Fine-tuned", "37.7 [35.1, 40.3]", "-", "0.58", "7.1%"],
     ["Gradient boosting (uses labels)", "40.4 [37.3, 43.4]", "1.43", "0.64", "6.1%"]], [5.0, 3.6, 1.5, 1.6, 2.2])
P("With the chat template the model scores 17.5%, 12.7 points below the majority class and at chance (19.4%). Removing the profile raises accuracy to 21.1%, so the profile changes accuracy by -3.6 points (-5.9 to -1.2). Yet the model reacts strongly to the profile: its prediction changes on 50% of rows relative to no profile, and the profile explains 29% of its prediction variance, against 4% for real answers and 6% for GBM. Its group differences are weakly aligned with the truth (r = 0.22, 95% CI 0.02 to 0.35) and strongly attenuated (slope 0.18). On the full fold (6,000 respondents, 46,209 rows) the same model scores 16.1% with the chat template and 18.0% with the raw prompt, so the subset result is not a sampling artefact. A prior correction recovers about two thirds of the distance to the majority class (26.3%) and no more.")
FIG("fig_direction.png", "Figure 6.1. Predicted versus true group differences (182 cells, 200 respondents). Zero-shot differences are nearly flat; the fine-tuned and GBM predictions follow the diagonal.", 15.0)
H2("6.3 RQ2: Fine-Tuning")
TAB("Table 6.3. Pew fold 0: 46,209 held-out answers from 6,000 respondents.",
    [["Method", "Accuracy (%) [95% CI]", "NLL", "ECE"],
     ["Majority class", "30.4 [29.9, 30.8]", "1.59", "0.001"],
     ["Logistic regression", "38.5 [38.0, 39.0]", "1.47", "0.004"],
     ["Gradient boosting", "40.0 [39.6, 40.5]", "1.44", "0.005"],
     ["Fine-tuned Qwen2.5-7B", "38.0 [37.5, 38.4]", "1.48", "0.009"],
     ["Zero-shot Qwen2.5-7B (raw)", "18.0 [17.6, 18.4]", "3.07", "0.452"],
     ["Zero-shot Qwen2.5-7B (chat)", "16.1 [15.7, 16.4]", "4.26", "0.519"]], [5.6, 4.4, 2.0, 2.0])
P("The fine-tuned model reaches 38.0%: 7.6 points above the majority class, 0.5 below logistic regression (-0.8 to -0.2) and 2.1 below GBM, which closes 79% (75 to 83) of the gap between the majority class and GBM. Its NLL of 1.48 sits between logistic regression and the majority class, above GBM, and it is well calibrated. It trails GBM on every one of the eight items, by 1.2 to 2.7 points. Averaging its probabilities with GBM's gives no accuracy gain over GBM. On the subset it is 20.2 points above zero-shot (paired 95% CI 16.9 to 23.5). Its group differences correlate r = 0.58 with the truth on the subset (GBM 0.64) and 0.82 on the full fold (GBM 0.91). Hard predictions inflate between-group differences for all three methods (2.0 times for the fine-tuned model, 2.0 for GBM and 2.6 for logistic regression).")
TAB("Table 6.4. Subgroup accuracy gaps on Pew fold 0 (percentage points).",
    [["Axis", "Majority", "GBM", "Fine-tuned gap [95% CI]", "p", "Fine-tuned minus GBM"],
     ["Religion", "5.3", "9.1", "10.0 [7.3, 13.5]", "<0.001", "+1.0 [-2.1, +4.0]"],
     ["Region", "13.4", "7.4", "8.8 [7.1, 11.3]", "<0.001", "+1.4 [+0.1, +3.5]"],
     ["Education", "6.6", "1.2", "2.0 [0.9, 3.6]", "0.058", "+0.8 [-0.8, +2.2]"],
     ["Age", "5.6", "3.8", "2.7 [1.7, 4.2]", "0.005", "-1.2 [-2.4, +0.5]"],
     ["Caste", "2.8", "1.5", "3.4 [1.5, 6.3]", "0.071", "+1.9 [-1.5, +4.0]"],
     ["Urban/rural", "2.9", "2.0", "1.3 [0.3, 2.2]", "0.012", "-0.7 [-1.5, +0.1]"],
     ["Income", "2.6", "1.5", "1.7 [0.6, 2.7]", "<0.001", "+0.2 [-0.6, +1.1]"],
     ["Sex", "2.1", "0.7", "0.4 [0.0, 1.3]", "0.37", "-0.3 [-1.0, +0.5]"]], [2.4, 1.8, 1.5, 3.6, 1.6, 3.6])
P("The fine-tuned model's gaps differ from zero on five of eight axes after Holm correction (religion 10.0 points, region 8.8, age 2.7, income 1.7, urban/rural 1.3) and differ from GBM's by at most 1.9 points. On probabilities, its predicted subgroup distributions are close to the truth: the mean Jensen-Shannon divergence is 0.009 across religion cells and 0.006 across region cells, against 0.004 and 0.002 for logistic regression, 0.029 and 0.020 for the item marginal, and 0.271 and 0.293 for the zero-shot model.")
H2("6.4 RQ3: Transfer to Unseen Questions")
P("The adapter was applied to eight Pew items it never saw. With the approximate MLX re-implementation, the adapter scores 32.9% on the trained items against 18.8% for its base (a gain of 14.1 points), while the original adapter scores 40.5% on the same rows, so the port is weaker. On the unseen items the adapter scores 28.9% and its base 28.5%, a difference of +0.4 points (-5.1 to +5.9). Over respondents the in-domain gain exceeds the unseen gain by 13.7 points (7.0 to 20.4), but that interval treats the eight items on each side as fixed. Across items the difference is 14.0 points with a standard error of 10.2 (permutation p = 0.18), and the unseen gain runs from -21 to +36 points per item. We therefore do not claim that the in-domain gain is reliably larger; the safe reading is that no gain on unseen items is detectable. Classifiers fitted with labels on these items reach 37.6% (majority), 44.7% (logistic) and 47.6% (GBM), so here the adapted LLM is no substitute for data on the new question.")
FIG("fig_transfer.png", "Figure 6.2. (a) Accuracy on trained and unseen items (120 respondents). (b) Gain of the fine-tuned model over its zero-shot base, with paired 95% confidence intervals.", 15.0)
H2("6.5 RQ4: Scale and Data")
P("gpt-oss-120b scores 27.6% on 100 respondents (774 rows), 2.3 points below the majority class (30.0%, not significant) and 11.8 points below the fine-tuned 7B model. It is a different and larger model, so we cannot attribute the difference from the 7B model to scale. Averaged over five random subsets, logistic regression matches the majority class up to 100 labelled respondents, beats it by 3.9 points at 400 and by 6.4 at 1,600; GBM clears the majority class only at about 1,600 respondents, and its NLL is still worse than the majority class's there. On the same 200 test respondents the fine-tuned model, trained on 23,999, scores 37.7%, slightly above logistic regression trained on 1,600 (36.4%), within sampling error.")
FIG("fig_lc.png", "Figure 6.3. Classical models on nested respondent subsets: accuracy and NLL against the number of training respondents.", 12.0)
H2("6.6 Summary of Findings")
TAB("Table 6.5. Summary of findings.",
    [["Question", "Finding", "Status"],
     ["RQ1 use", "Profile explains 29% of zero-shot variance (4% in real answers) but direction r = 0.22; removing it raises accuracy by 3.6 points", "Established (200 respondents; confirmed on full fold)"],
     ["RQ2 adaptation", "38.0% vs GBM 40.0%, majority 30.4%; closes 79% of the gap; ECE 0.009", "Established (6,000 respondents, one fold and seed)"],
     ["RQ3 transfer", "No detectable gain on 8 unseen items (28.9% vs 28.5%)", "Limited (approximate port, 8 items)"],
     ["RQ4 scale and data", "gpt-oss-120b at the majority class; logistic needs about 400 respondents, GBM about 1,600", "Established on subsets"]], [3.0, 8.0, 3.5])

H1("7. DISCUSSION")
H2("7.1 Interpretation")
P("A zero-shot model does not ignore the profile. It varies with it far more than real answers do, in directions that match the data only weakly, and the net effect on exact-match accuracy is slightly negative. That fits small, noisy persona effects [5] and the over-determination reported in [3]. We did not vary prompts, so we cannot say whether richer or better-worded demographics would help.")
P("Fine-tuning on respondent-level data gets item marginals and group differences close to a classifier's, but not all the way. Accuracy ends 2.1 points below GBM, directional validity on the full fold is 0.82 against 0.91, and the two models agree on 72% of predictions. The absence of a detectable gain on unseen items fits an item-specific mapping from profile to answer, but given the port, the number of items and the narrow training set it is not evidence that transfer is impossible.")
H2("7.2 Cost and Benefit")
P("Section 5.6 shows what each answer costs: about 0.2 seconds of T4 time per row for the 7B model, hours of GPU time to fine-tune, and seconds for a classifier. Where matched classifiers are equally accurate on the trained questions and the adapter shows no detectable gain on new ones, the cost favours the classifier. The case for LLM respondents has to rest on settings a classifier cannot serve, which our held-out-item test probed only weakly.")
H2("7.3 Evaluation Practice")
P("Accuracy comparisons need a majority-class baseline, because it was 45.5% over the 144 WVS items. Inflation and concentration of hard predictions should be read against baselines that show them too. Subgroup-gap claims need a permutation null and detection thresholds, which at tens of respondents exceed any plausible effect. Verification mattered as much as modelling: the region-label error and the decoding artefact both changed conclusions until they were found.")
H2("7.4 Limitations and Threats to Validity")
P("Everything comes from one fold, one seed and one epoch. Scoring used fp16 on a T4 while training used bf16, and the original run's greedy decoding scores lower, so results depend on the readout; we use the one that matches training. Fine-tuning used eight religion- and fertility-related items, so transfer to broader item sets is untested. The transfer results use an MLX re-implementation about 7 to 8 points weaker than the original, so that comparison is conservative for the original model, and it rests on eight items and 120 respondents. The local zero-shot controls use random subsets of 100 to 200 respondents, although the full-fold zero-shot result agrees. We tested one 7B model locally and two free-tier API models, and did not test frontier models on these surveys. Pew survey weights are not applied. Accuracy is exact match on ordinal items with argmax predictions, and the variance share is an in-sample increase in R-squared.")

H1("8. CONCLUSION AND FUTURE WORK")
H2("8.1 Conclusion")
P("A prompted 7B model varies strongly with a demographic profile but uses it weakly and slightly harmfully. Fine-tuning on respondent-level data recovers most, though not all, of the accuracy and direction gap to classifiers on the trained questions, and shows no detectable accuracy gain on eight new ones. In our settings matched classifiers remain the benchmark that LLM respondents have to beat, and they are far cheaper to run. The pipeline, results and code are reproducible, and the work has been accepted for presentation at AIDE-2026.")
H2("8.2 Future Work")
BUL(["Fine-tune on many items and test transfer on dozens of unseen items with an item-level bootstrap, using the original adapter instead of a port.",
     "Repeat the study across folds and seeds.",
     "Test richer text-conditioned personas and aggregate distribution estimates from calibrated probabilities.",
     "Measure the cost side more carefully, including smaller and distilled models and energy use.",
     "Widen the WVS evaluation from 15 items towards the full 144."])
H2("8.3 Publication Outcome")
P("A paper from this work, titled How Do LLM Respondents Use Demographics? Prompting, Fine-Tuning and Held-Out-Item Transfer for Survey Simulation in India, was accepted with modification as a full research paper at the Hinweis Fifth International Conference on Artificial Intelligence and Data Science (AIDE-2026, paper ID 591, hybrid, Bengaluru, 28 to 29 November 2026). The reviewers asked for a clearer paper structure and a discussion of computational requirements; both were added, and Chapter 5 of this report contains the same cost analysis.")

REFS = [
 "L. P. Argyle et al., “Out of one, many: Using language models to simulate human samples,” Political Analysis, vol. 31, no. 3, pp. 337–351, 2023.",
 "S. Santurkar, E. Durmus, F. Ladhak, C. Lee, P. Liang, and T. Hashimoto, “Whose opinions do language models reflect?” in Proc. ICML, 2023.",
 "Z. Chen, D. Zhu, and L. N. Zheng, “When synthetic users fail: A cross-domain benchmark of LLM-simulated human survey responses,” arXiv:2607.26348, 2026.",
 "F. Miranda and P. P. Balbi, “Simulating public opinion: Comparing distributional and individual-level predictions from LLMs and random forests,” Entropy, vol. 27, no. 9, Art. 923, 2025.",
 "T. Hu and N. Collier, “Quantifying the persona effect in LLM simulations,” in Proc. 62nd Annu. Meeting ACL, 2024, pp. 10289–10307.",
 "E. Durmus et al., “Towards measuring the representation of subjective global opinions in language models,” arXiv:2306.16388, 2023.",
 "J. Bisbee, J. D. Clinton, C. Dorff, B. Kenkel, and J. M. Larson, “Synthetic replacements for human survey data? The perils of large language models,” Political Analysis, vol. 32, no. 4, pp. 401–416, 2024.",
 "H. Shankar et al., “Mind the gap: Pitfalls of LLM alignment with Asian public opinion,” in Proc. ICWSM, 2026, arXiv:2603.06264.",
 "J. Suh, E. Jahanparast, S. Moon, M. Kang, and S. Chang, “Language model fine-tuning on scaled survey data for predicting distributions of public opinions,” in Proc. 63rd Annu. Meeting ACL, 2025, pp. 21147–21170.",
 "Y. Cao et al., “Specializing large language models to simulate survey response distributions for global populations,” in Proc. NAACL, 2025.",
 "S. Krsteski, G. Russo, S. Chang, R. West, and K. Gligorić, “Valid survey simulations with limited human data: The roles of prompting, fine-tuning, and rectification,” arXiv:2510.11408, 2025.",
 "J. Kim and B. Lee, “AI-augmented surveys: Leveraging large language models and surveys for opinion prediction,” arXiv:2305.09620, 2023.",
 "C. Wang, “Silicon sampling answers with country-level assumptions, not individual attitudes: Cross-national evidence from the European Social Survey,” arXiv:2609.16395, 2026.",
 "N. H. Y. Lam, J. X. Voo, and X. Ma, “The conditional superiority of fast silicon sampling,” arXiv:2608.14079, 2026.",
 "F. D. Robbani, “Readable, faithful, used: Three dissociable properties of demographic identity in a language model,” arXiv:2608.18768, 2026.",
 "E. C. Choi, Y. Kim, P. Pugalenthi, H.-E. Chen, and B.-R. Huang, “Beyond the mean: Three-axis fidelity for aligning LLM-based survey simulators from small pilot data,” in Pluralistic Alignment Workshop at ICML, 2026, arXiv:2606.28963.",
 "C. Haerpfer et al. (eds.), World Values Survey: Round Seven, Country-Pooled Datafile Version 6.0.0. Madrid, Spain and Vienna, Austria: JD Systems Institute & WVSA Secretariat, 2024, doi:10.14281/18241.24.",
 "Pew Research Center, “Religion in India: Tolerance and segregation,” Jun. 2021.",
 "T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, “QLoRA: Efficient finetuning of quantized LLMs,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2023.",
 "E. J. Hu et al., “LoRA: Low-rank adaptation of large language models,” in Proc. ICLR, 2022.",
 "Qwen Team, “Qwen2.5 technical report,” arXiv:2412.15115, 2024.",
 "F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” J. Mach. Learn. Res., vol. 12, pp. 2825–2830, 2011.",
 "OpenAI, “gpt-oss-120b & gpt-oss-20b model card,” arXiv:2508.10925, Aug. 2025.",
 "A. Hannun, J. Digani, A. Katharopoulos, and R. Collobert, “MLX: Efficient and flexible machine learning on Apple silicon,” software, 2023. [Online]. Available: https://github.com/ml-explore/mlx",
]

APPX = []
APPX.append(("A. EXAMPLE PROMPT (P2)", [("box", "Prompt sent to the model for respondent 27684 on the Pew item about politicians and religion.",
 "You are a survey respondent with the following demographic profile:\n- Sex: Female\n- Age: 35-44\n- Marital status: Married or living with a partner\n- Education level: Literate but no formal schooling\n- Income decile: Rs. 10,000 and below\n- Religion: Hindu\n- Urban/Rural: Rural\n- Region: East\n- Caste: Other Backward Class\n\nAs this person, please answer the following survey question. Your answer should reflect the values and perspectives typical of someone with your background:\n\nQuestion: How much influence should politicians have in religious matters?\n\nAnswer options:\n1. A large influence\n2. Some influence\n3. Not too much influence\n4. No influence at all\n\nYour answer:\n\nThis is anonymized survey-simulation research. Respond with exactly one character: one of 1/2/3/4. No words, no punctuation, no explanation, just that single digit."),
 ("p", "For this randomly drawn test row the true answer is option 1. The zero-shot model puts 77% of its probability on option 3, the fine-tuned model spreads probability over all four options (top choice 2), and gradient boosting puts 49% on option 1.")]))
APPX.append(("B. PEW ITEMS USED", [("table", "Table B.1. Items used for fine-tuning (1 to 8) and held out for transfer (9 to 16).",
 [["#", "Item (Pew code)", "Role"], ["1", "Number of children (QFERTrec)", "Trained"], ["2", "Reads scripture (Q50)", "Trained"], ["3", "Politicians' role in religion (Q12)", "Trained"], ["4", "Attends worship (Q3132rec)", "Trained"],
  ["5", "Abortion legality (ABORT)", "Trained"], ["6", "Sex-selection checkup (Q15)", "Trained"], ["7", "Watches religious TV (Q39a)", "Trained"], ["8", "Nature of religious identity (Q28rec)", "Trained"],
  ["9", "Meditation (Q34a)", "Held out"], ["10", "Self-rated health (Q13)", "Held out"], ["11", "Trust in health advice (Q14b)", "Held out"], ["12", "What it takes to be truly Indian (Q69a)", "Held out"],
  ["13", "What it takes to be truly Indian (Q69b)", "Held out"], ["14", "Watches serials (Q39b)", "Held out"], ["15", "Trust in health advice (Q14c)", "Held out"], ["16", "Praying (Q54a)", "Held out"]], [1.2, 9.5, 3.5])]))
APPX.append(("C. REPOSITORY MAP", [("table", "Table C.1. Main repository components.",
 [["Path", "Contents"], ["src/data/", "Survey loading, codebook parsing, item screening, fold construction"], ["src/prompts/", "Verbalisation of profiles and items; prompt conditions P0 to P3"],
  ["src/inference/", "Cached runner, Gemini and Groq integrations, shared instruction and parsing"], ["src/eval/", "Metrics, baselines, bootstrap, subgroup analysis"],
  ["scripts/", "Pew fine-tuning, inference and evaluation scripts"], ["kaggle_experiments/", "GPU training, scoring and analysis scripts, pre-specified analysis plan"],
  ["paper/", "Conference paper sources, tables, figures, camera-ready build, poster"], ["results/ (not in git)", "Predictions and caches containing respondent-level data"]], [4.2, 10.3])]))
APPX.append(("D. LOG OF PROBLEMS FOUND AND FIXED", [("bul", [
 "Missing-value recode count logged only the last column's count.", "Country filter failed silently on a wrong-version file.", "Ordinal-scale check rejected 354 of 373 candidates because every numeric column was float.",
 "Hardcoded scale metadata was wrong for 8 of 18 items; replaced by codebook-grounded screening.", "Eager imports crashed lightweight environments.", "Report showed a raw code instead of the resolved label.",
 "A CSS grid left a solid block when tiles did not fill the last row.", "A stale return statement raised a NameError after a refactor.", "Packaging failed because the package name did not match the source layout.",
 "Prompt formatters crashed on the full demographics dictionary.", "A model name containing a slash broke cache file paths.", "An unconditional log-probability request could have zeroed every real prediction.",
 "Support detection for log-probabilities doubled API usage until it was cached.", "The resumable cache treated transient failures as permanent.", "A retry wrapper looped without incrementing its counter.",
 "WVS region labels were wrong on every code; fixed against the official annex and both models re-run.", "The apparent collapse on one Pew item came from the original run's decoding, not the adapter."])]))
APPX.append(("E. EXPERIMENT LOG", [("table", "Table E.1. Main stages of the project.",
 [["Stage", "What was done", "Outcome"],
  ["Data and items", "WVS-7 cleaning, codebook parsing, item screening, folds", "144 items, 1,692 India respondents"],
  ["Baselines", "Majority, marginal, cell lookup, logistic, GBM, out of fold", "GBM 48.2%, matched 50.1%"],
  ["Phase 1 (API)", "Gemini and gpt-oss-120b, P0 to P3, 5 and 15 items", "22 to 26% on 15 items; gaps do not replicate"],
  ["Pew pipeline", "DDI parsing, 73 items, folds, 8 trained and 8 held-out items", "Fold 0: 23,999 / 6,000 respondents"],
  ["Fine-tuning", "QLoRA of Qwen2.5-7B on a lab GPU", "184,853 examples, 6.6 h; 36.1% greedy decode"],
  ["Faithful scoring", "Rescored the adapter on Kaggle T4 GPUs in training format", "38.0%, ECE 0.009"],
  ["Diagnostics", "Direction, variance share, JSD, subgroup gaps", "See Chapter 6"],
  ["Transfer and scale", "MLX port on 8 unseen items; gpt-oss-120b; learning curves", "No detectable transfer; majority-level 120B"],
  ["Paper", "Hinweis AIDE-2026, paper ID 591", "Accepted with modification; camera-ready sent"]], [3.0, 7.0, 4.5])]))
APPX.append(("F. GLOSSARY", [("bul", [
 "**Silicon sampling:** using an LLM conditioned on a demographic profile as a stand-in survey respondent.", "**Zero-shot:** using a model without any task-specific training.",
 "**QLoRA:** fine-tuning with low-rank adapters on a 4-bit quantised base model.", "**NLL:** negative log-likelihood of the true answer under the predicted distribution; lower is better.",
 "**ECE:** expected calibration error; how far stated confidence is from accuracy.", "**Directional validity:** agreement between predicted and true differences between demographic groups.",
 "**JSD:** Jensen-Shannon divergence between two distributions, 0 to 1.", "**Holm correction:** a multiple-testing adjustment of p-values.",
 "**Majority class:** always predicting the most common answer; the minimum bar any method must beat."])]))
