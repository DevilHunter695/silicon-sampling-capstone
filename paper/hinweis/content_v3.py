# Camera-ready restructure for AIDE-2026 (paper ID 591): reviewer asked for cleaner structure and a compute/scalability discussion.
import re
import content_h as _h
def G(prefix):
    for b in _h.B:
        if b[0] == "p" and b[1].startswith(prefix): return b[1]
    raise KeyError(prefix)
def tab_ref(t):  # renumber figure/table references from the first-round paper to the new numbering
    pairs = [("Table VI)", "Table §VII)"), ("(Table V, Fig. 2)", "(Fig. §3)"), ("Table IV)", "Table §V)"), ("Table III)", ""), ("(Table II)", "(Table §IV)"),
             ("Table II)", "Table §IV)"), ("Table I summarises", "Table §III summarises"), ("as Fig. 1 shows", "as Fig. §2 shows"), ("Section IV-C", "Section VI-C"), ("(Table VI)", "(Table §VII)")]
    for a, b in pairs: t = t.replace(a, b)
    return t.replace("§", "")

TITLE = "How Do LLM Respondents Use Demographics? Prompting, Fine-Tuning and Held-Out-Item Transfer for Survey Simulation in India"
AUTHORS = _h.AUTHORS
AFFIL = _h.AFFIL
ABSTRACT = _h.ABSTRACT.replace("A 120B zero-shot model scores", "Fine-tuning took about 6.6 GPU-hours, scoring costs about 0.2 s per answer on a T4 GPU, and a classifier is orders of magnitude cheaper. A 120B zero-shot model scores")
KEYWORDS = ("silicon sampling, survey simulation, large language models, parameter-efficient fine-tuning, QLoRA, algorithmic fidelity, model calibration, "
            "demographic conditioning, computational social science, Indian public opinion, Pew Research Center, World Values Survey")
ACK = _h.ACK

B = []
def H(t): B.append(("h1", t))
def S(t): B.append(("h2", t))
def P(t): B.append(("p", t))
def T(cap, rows, widths=None): B.append(("table", cap, rows, widths, None))
def F(path, cap, width=14.8): B.append(("fig", path, cap, width))
def BOX(cap, text): B.append(("box", cap, text))

H("I. Introduction")
P(G("Silicon sampling asks")); P(G("What these benchmarks leave open")); P(G("We study these questions")); P(G("We contribute"))
P("Section II reviews related work, Section III describes the data, Section IV presents the evaluation framework, Section V reports computational requirements and scalability, Section VI gives the results, and Sections VII to IX discuss the findings, limitations and future work.")

H("II. Related Work")
P(G("**Simulating respondents.**")); P(G("**Baselines and adaptation.**"))

H("III. Datasets")
S("A.  Pew Religion in India")
P(G("**Pew Religion in India**").replace("**Pew Religion in India** [@pew2021] is", "The Pew Research Center survey Religion in India [@pew2021] is"))
S("B.  WVS-7 India")
P("The World Values Survey wave 7 for India (version 6.0 [@wvs7]; N = 1,692) was screened against its codebook from 373 candidate items down to 144. The zero-shot runs use 15 of them and a 14-attribute prompt.")

H("IV. Proposed Evaluation Framework")
P("Fig. 1 shows the framework. Every predictor receives the same prompt and is scored on the same respondent-item rows, so differences between methods cannot come from different data.")
F("fig_framework.png", "Fig. 1. Evaluation framework. Both surveys are verbalised into the same prompt, passed to zero-shot, fine-tuned, larger and classical predictors, and scored on identical rows with the same diagnostics and tests.", 13.2)
S("A.  Prompt and Readout")
P("Each respondent-item pair becomes one prompt (P2): a sentence listing the respondent's demographic profile, the question text, the numbered answer options and an instruction to reply with a single digit (Table I). The profile is verbalised from codebook labels (nine attributes for Pew, 14 for WVS-7). The model's next-token distribution over the option digits gives the probabilities, and its argmax gives the hard prediction. The same prompt without the profile (P0) isolates the profile's contribution.")
BOX("Table I. Example prompt (P2) for one respondent and one Pew item.",
    "You are a survey respondent with the following demographic profile: Sex: Female. Age: 18-25. Marital status: Married or living with a partner. Religion: Muslim. Urban/Rural: Rural. Region: East. Caste: General Category.\n"
    "As this person, please answer the following survey question. Your answer should reflect the values and perspectives typical of someone with your background:\n"
    "Question: How much influence should politicians have in religious matters?\n"
    "Answer options: 1. A large influence  2. Some influence  3. Not too much influence  4. No influence at all\n"
    "Your answer:\n"
    "This is anonymized survey-simulation research. Respond with exactly one character: one of 1/2/3/4. No words, no punctuation, no explanation, just that single digit.")
S("B.  Baselines and Controls")
P(G("**Matched baselines.**")); P(G("**Zero-shot model.**"))
P(G("**Larger zero-shot model.** gpt-oss-120b [@gptoss2025]"))
S("C.  Fine-Tuned Model")
P(tab_ref(G("**Fine-tuned model.**")).replace("**Fine-tuned model.** ", ""))
S("D.  Diagnostics")
P(G("**Diagnostics.**").replace("**Diagnostics.** ", ""))
S("E.  Transfer and Data-Efficiency Protocols")
P(G("**Transfer.**"))

H("V. Computational Requirements and Scalability")
S("A.  Measured Cost")
P("Table II lists the hardware and measured cost of each part of the study. All figures come from our own runs. Scoring one condition on the full Pew fold (46,209 rows) took about 75 minutes on two Kaggle T4 GPUs working in parallel, which is about 0.19 to 0.21 s per row per GPU at batch size 16 and a mean prompt length of 231 tokens (maximum 288). The fine-tuning run took 5,777 optimizer steps and 396.7 minutes on a single lab GPU. In a separate reduced run on a T4, an optimizer step over 16 examples took 9 to 10 s, which is about 0.6 s per example. The 4-bit base model fits in about 4 to 5 GB of weights, so the 7B model runs on a 16 GB laptop or a 15 GB T4.")
T("Table II. Hardware and measured cost.",
  [["Task", "Hardware", "Measured cost"],
   ["QLoRA fine-tuning, 184,853 examples", "One lab GPU", "5,777 steps, 6.6 h (396.7 min)"],
   ["Scoring one condition, 46,209 rows", "Two Kaggle T4, fp16", "About 75 min; 0.19-0.21 s per row per GPU"],
   ["Local zero-shot scoring", "Apple M1, 16 GB, 4-bit MLX", "2.3 s per prompt with prefix caching; about 5 h per 1,000 respondents"],
   ["Matched classical baselines, 8 items", "Laptop CPU", "Under 1 min for the whole fold"],
   ["gpt-oss-120b, 774 rows", "Free API tier", "Limited by rate limits; several hours"]], [5.6, 4.2, 5.2])
S("B.  Complexity and Scaling")
P("Cost grows linearly with the number of respondent-item rows, because every row needs one forward pass over a prompt of about 230 tokens, which is roughly 2 x 7.6 billion x 230, or 3.5 TFLOP, for a 7B model. Training cost grows linearly with the number of examples. It does not depend on the number of subgroups. Scoring all 29,999 Pew respondents on all 73 screened items (about 2.2 million rows) would take roughly 116 GPU-hours on a T4 at the measured rate, or about 58 hours on two. Fine-tuning on all 73 items (about 1.75 million examples) would take roughly 63 hours at the lab GPU's rate and about 290 hours at the T4 rate. These are linear extrapolations, not measurements. A classifier, by contrast, trains in seconds per item and predicts a row in microseconds, several orders of magnitude cheaper.")
S("C.  Practical Implications")
P("Because the fine-tuned model is no more accurate than matched classifiers on the trained items (Section VI-B), its cost has to be justified by something else, such as transfer to new questions or use as a generator of full response profiles. Our transfer test did not show such a benefit. Cost can be cut by caching the profile prefix across a respondent's items, as our local runs did, by batching, and by using smaller or distilled models, but we did not test the last of these.")

H("VI. Results and Analysis")
S("A.  RQ1: The Profile Is Used Heavily, Not Usefully")
P(tab_ref(G("Table I summarises the zero-shot behaviour.")))
T("Table III. Pew, 200 held-out respondents (1,548 rows). Direction and variance columns exclude one non-ordinal item (1,351 rows, 182 cells). Dir. r: correlation between predicted and true group differences. Profile var.: variance explained by the profile beyond item effects.",
  [["Predictor (200 respondents)", "Acc. (%) [95% CI]", "NLL", "Dir. r", "Profile var."],
   ["True answers", "-", "-", "-", "4.0%"],
   ["Majority class (uses labels)", "30.2 [28.1, 32.4]", "1.59", "-", "-"],
   ["Zero-shot, no profile", "21.1 [19.1, 23.1]", "3.92", "-", "0.0%"],
   ["Zero-shot, profile (chat)", "17.5 [15.7, 19.4]", "3.88", "0.22", "29.4%"],
   ["Zero-shot, prior-corrected", "26.3 [24.0, 28.6]", "-", "-", "-"],
   ["Fine-tuned (training-format readout)", "37.7 [35.1, 40.3]", "-", "0.58", "7.1%"],
   ["Gradient boosting (uses labels)", "40.4 [37.3, 43.4]", "1.43", "0.64", "6.1%"]], [6.2, 3.6, 1.6, 1.6, 2.2])
F("fig_direction.png", "Fig. 2. Predicted versus true group differences for 182 (item, axis, category) cells (200 respondents; item-centred, normalised scale). The dashed line is the identity. Zero-shot differences are nearly flat and weakly correlated; the fine-tuned and GBM predictions follow the diagonal.", 15.0)
S("B.  RQ2: Fine-Tuning Recovers Most of the Gap, Short of GBM")
P(tab_ref(G("On the full fold (Table II)").replace(" (Table III)", "")))
T("Table IV. Pew fold 0: 46,209 held-out answers from 6,000 respondents. NLL: negative log-likelihood; ECE: expected calibration error.",
  [["Method", "Accuracy (%) [95% CI]", "NLL", "ECE"],
   ["Majority class", "30.4 [29.9, 30.8]", "1.59", "0.001"],
   ["Logistic regression", "38.5 [38.0, 39.0]", "1.47", "0.004"],
   ["Gradient boosting", "40.0 [39.6, 40.5]", "1.44", "0.005"],
   ["Fine-tuned Qwen2.5-7B", "38.0 [37.5, 38.4]", "1.48", "0.009"],
   ["Zero-shot Qwen2.5-7B (raw)", "18.0 [17.6, 18.4]", "3.07", "0.452"],
   ["Zero-shot Qwen2.5-7B (chat)", "16.1 [15.7, 16.4]", "4.26", "0.519"]], [6.0, 4.4, 1.6, 1.6])
P(G("On the 200-respondent subset the fine-tuned model"))
P(tab_ref(G("The fine-tuned model's subgroup gaps")))
T("Table V. Subgroup accuracy gap (best minus worst category, percentage points) on Pew fold 0, with the permutation p-value for the fine-tuned model and its difference from GBM's gap.",
  [["Axis", "Majority", "GBM", "Fine-tuned gap [95% CI]", "p", "Fine-tuned minus GBM [95% CI]"],
   ["Religion", "5.3", "9.1", "10.0 [7.3, 13.5]", "<0.001", "+1.0 [-2.1, +4.0]"],
   ["Region", "13.4", "7.4", "8.8 [7.1, 11.3]", "<0.001", "+1.4 [+0.1, +3.5]"],
   ["Education", "6.6", "1.2", "2.0 [0.9, 3.6]", "0.058", "+0.8 [-0.8, +2.2]"],
   ["Age", "5.6", "3.8", "2.7 [1.7, 4.2]", "0.005", "-1.2 [-2.4, +0.5]"],
   ["Caste", "2.8", "1.5", "3.4 [1.5, 6.3]", "0.071", "+1.9 [-1.5, +4.0]"],
   ["Urban/rural", "2.9", "2.0", "1.3 [0.3, 2.2]", "0.012", "-0.7 [-1.5, +0.1]"],
   ["Income", "2.6", "1.5", "1.7 [0.6, 2.7]", "<0.001", "+0.2 [-0.6, +1.1]"],
   ["Sex", "2.1", "0.7", "0.4 [0.0, 1.3]", "0.37", "-0.3 [-1.0, +0.5]"]], [2.4, 1.8, 1.4, 3.6, 1.6, 4.0])
S("C.  RQ3: No Detectable Gain on Unseen Items")
P(tab_ref(G("We applied the adapter to eight Pew items")))
F("fig_transfer.png", "Fig. 3. (a) Accuracy on the eight trained and eight unseen items (120 respondents; 95% respondent-bootstrap CIs). (b) Gain of the fine-tuned model over its zero-shot base, with paired 95% CIs. Classifiers marked “with labels” are not available for a genuinely new question.", 15.0)
S("D.  RQ4: Scale and Data")
P(G("**Larger zero-shot model.** gpt-oss-120b scores")); P(G("**Labelled data.**"))
S("E.  WVS-7 India Replication")
P(tab_ref(G("Across the 144 screened items").replace(" (Table VI)", "")))

H("VII. Discussion")
P(G("**What the profile does.**")); P(G("**What fine-tuning does.**")); P(G("**Where LLM respondents could add value.**"))
P("**Cost and benefit.** Section V shows what each answer costs: about 0.2 s of T4 time per row for the 7B model, hours of GPU time to fine-tune, and seconds for a classifier. Where matched classifiers are equally accurate on the trained questions and the adapter shows no detectable gain on new ones, the cost favours the classifier. The case for LLM respondents would have to rest on settings classifiers cannot serve, which the held-out-item test probed only weakly.")
P(G("**Evaluation practice.**") if any(b[0]=="p" and b[1].startswith("**Evaluation practice.**") for b in _h.B) else "")

H("VIII. Limitations")
P(G("Everything comes from one fold"))

H("IX. Conclusions and Future Work")
P("A prompted 7B model varies strongly with a demographic profile but uses it weakly and slightly harmfully. Fine-tuning on respondent-level data recovers most, though not all, of the accuracy and direction gap to classifiers on the trained questions, and shows no detectable accuracy gain on eight new ones. In our settings matched classifiers remain the benchmark LLM respondents have to beat, and they are far cheaper to run.")
P("Future work has four parts. First, fine-tune on many items and test transfer on dozens of unseen items with an item-level bootstrap, using the original adapter instead of a port. Second, repeat the study across folds and seeds. Third, test richer text-conditioned personas and aggregate distribution estimates from calibrated probabilities. Fourth, measure the cost side more carefully, including smaller and distilled models and energy use. Code, prompts and analysis scripts are at github.com/DevilHunter695/silicon-sampling-capstone. WVS-7 must be obtained from the WVS Association and the Pew data from Pew Research Center.")
B[:] = [b for b in B if not (b[0] == "p" and not b[1])]
