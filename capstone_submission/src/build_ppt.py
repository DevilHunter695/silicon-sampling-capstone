import sys
from PIL import ImageFont
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
NAVY, ORG, LGT, GRY, WHT = RGBColor(0x00, 0x1d, 0x3e), RGBColor(0xff, 0xb0, 0x20), RGBColor(0xee, 0xf2, 0xfb), RGBColor(0x4a, 0x5a, 0x80), RGBColor(255, 255, 255)
prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
TOTAL = [0]

FONT_R = "/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts/Calibri.ttf"
FONT_B = "/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts/Calibrib.ttf"
_fc = {}
def _font(sz, b):
    k = (round(sz * 4), b)
    if k not in _fc: _fc[k] = ImageFont.truetype(FONT_B if b else FONT_R, int(round(sz * 4)))
    return _fc[k]
def needed_height(paras, width_in, size, bullet, bold, space):
    wpt = (width_in - 0.16) * 72 * 4
    total = 0.0
    for p in (paras if isinstance(paras, list) else [paras]):
        runs = p if isinstance(p, list) else [(p, bold)]
        if bullet: runs = [("\u2022  ", False)] + runs
        words = []
        for t, b in runs:
            for w in t.split(" "): words.append((w, b))
        lines, cur = 1, 0.0
        for w, b in words:
            wl = _font(size, b).getlength(w + " ")
            if cur + wl > wpt and cur > 0: lines += 1; cur = wl
            else: cur += wl
        total += lines * size * 1.2 + space
    return (total + 6) / 72
WARN = []
def rect(sl, x, y, w, h, fill, line=None):
    s = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)); s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None: s.line.fill.background()
    else: s.line.color.rgb = line
    s.shadow.inherit = False; return s
def text(sl, x, y, w, h, paras, size=18, color=NAVY, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, bullet=False, space=6):
    if isinstance(paras, str): paras = [paras]
    sz = size
    while needed_height(paras, w, sz, bullet, bold, space) > h and sz > 10: sz -= 0.5
    if needed_height(paras, w, sz, bullet, bold, space) > h: WARN.append((TOTAL[0], str(paras)[:50], round(needed_height(paras, w, sz, bullet, bold, space), 2), h))
    size = sz
    tb = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.08); tf.margin_top = tf.margin_bottom = Inches(0.04)
    if isinstance(paras, str): paras = [paras]
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); para.alignment = align; para.space_after = Pt(space)
        runs = p if isinstance(p, list) else [(p, bold)]
        if bullet: runs = [("•  ", False)] + runs
        for t, b in runs:
            r = para.add_run(); r.text = t; r.font.size = Pt(size); r.font.bold = b; r.font.color.rgb = color; r.font.name = "Calibri"
    return tb
def slide(title, notes, n=None):
    sl = prs.slides.add_slide(BLANK); TOTAL[0] += 1
    rect(sl, 0, 0, 13.333, 1.05, NAVY); rect(sl, 0, 1.05, 13.333, 0.06, ORG)
    text(sl, 0.45, 0.15, 12.3, 0.8, title, size=30, color=WHT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    rect(sl, 0, 7.12, 13.333, 0.38, NAVY)
    text(sl, 0.4, 7.14, 9, 0.32, "Group 55  |  Silicon Sampling for Indian Public Opinion  |  NMIMS MPSTME  |  Capstone 2026-27", size=11, color=WHT, anchor=MSO_ANCHOR.MIDDLE)
    text(sl, 11.9, 7.14, 1.0, 0.32, str(TOTAL[0]), size=11, color=WHT, align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
    sl.notes_slide.notes_text_frame.text = notes
    return sl
def pic(sl, path, x, y, w=None, h=None):
    return sl.shapes.add_picture(path, Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)
def card(sl, x, y, w, h, head, paras, size=16, bullet=True):
    rect(sl, x, y, w, h, LGT, RGBColor(0xb8, 0xc4, 0xe4)); rect(sl, x, y, w, 0.42, NAVY)
    text(sl, x + 0.1, y + 0.02, w - 0.2, 0.4, head, size=15, color=WHT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    text(sl, x + 0.12, y + 0.5, w - 0.24, h - 0.55, paras, size=size, bullet=bullet, space=5)
def table(sl, x, y, w, rows, colw, size=13, rowh=0.38, bold_rows=()):
    sh = sl.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(rowh * len(rows))); t = sh.table
    for ci, cw in enumerate(colw): t.columns[ci].width = Inches(cw)
    for ri, row in enumerate(rows):
        t.rows[ri].height = Inches(rowh)
        for ci, val in enumerate(row):
            c = t.cell(ri, ci); c.text = ""; p = c.text_frame.paragraphs[0]; r = p.add_run(); r.text = val; r.font.size = Pt(size); r.font.name = "Calibri"
            r.font.bold = (ri == 0 or ri in bold_rows); r.font.color.rgb = WHT if ri == 0 else NAVY
            p.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER; c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.margin_left = c.margin_right = Inches(0.06); c.fill.solid(); c.fill.fore_color.rgb = NAVY if ri == 0 else (LGT if ri % 2 else WHT)
    return t

# 1 Title
sl = prs.slides.add_slide(BLANK); TOTAL[0] += 1
rect(sl, 0, 0, 13.333, 7.5, NAVY); rect(sl, 0, 5.05, 13.333, 0.07, ORG)
pic(sl, "logo_nmims.png", 0.5, 0.45, h=1.5)
text(sl, 2.5, 0.65, 9, 1.0, "MUKESH PATEL SCHOOL OF TECHNOLOGY MANAGEMENT & ENGINEERING", size=20, color=WHT, bold=True)
text(sl, 0.6, 2.1, 12.2, 1.3, [[("SILICON SAMPLING ", True)], ], size=54, color=WHT, bold=True)
text(sl, 0.6, 2.95, 12.2, 1.0, [[("FOR INDIAN PUBLIC OPINION", True)]], size=54, color=ORG, bold=True)
text(sl, 0.6, 4.0, 12.2, 0.9, "How do LLM respondents use demographics? Prompting, fine-tuning and held-out-item transfer", size=22, color=RGBColor(0xdb, 0xe4, 0xff))
text(sl, 0.6, 5.3, 6.4, 1.8, [[("GROUP 55", True)], "Bhanuj Bhalla  (SAP 70022300017)", "Ishaan Bansal  (SAP 70022300011)", "Yash Bhende  (SAP 70022300137)", "Krishank Ayyella  (SAP 70022300136)"], size=18, color=WHT, space=3)
text(sl, 7.4, 5.3, 5.5, 1.8, [[("PROJECT MENTOR", True)], "Ms. Khushbu Chauhan", "", [("Capstone Project, A.Y. 2026-27", False)], "Final presentation: 31 October 2026"], size=18, color=WHT, space=3)
sl.notes_slide.notes_text_frame.text = "(30 s) Introduce the group, the mentor and the project. One sentence on what silicon sampling is: using a language model as a stand-in survey respondent."

# 2 Motivation
sl = slide("Motivation: can a language model stand in for a survey respondent?",
 "(1 min) Silicon sampling gives an LLM a demographic profile and treats its answer as that person's survey answer. If it worked it would cut cost and time. The danger is that a good average can hide poor performance for specific groups, which matters in a country as diverse as India. Our question is what the profile actually contributes, not just the headline accuracy.")
card(sl, 0.5, 1.4, 4.0, 5.4, "THE IDEA", ["Give an LLM a demographic profile.", "Ask the survey question.", "Treat its answer as that person's answer.", "Promise: far cheaper and faster than fielding a questionnaire."], size=17)
card(sl, 4.67, 1.4, 4.0, 5.4, "THE RISK", ["A good average can hide unequal performance across groups.", "India is highly diverse: region, religion, caste, income, age.", "Past work mostly reports aggregate fidelity, mostly for the US."], size=17)
card(sl, 8.83, 1.4, 4.0, 5.4, "OUR QUESTION", ["What does the profile itself contribute?", "Does fine-tuning on respondent-level data help?", "Does the help carry over to questions never seen?", "How does it compare with a simple classifier, and at what cost?"], size=17)

# 3 Literature
sl = slide("Literature review and the gap",
 "(1 min) Argyle 2023 introduced algorithmic fidelity on US data. Santurkar 2023 found misalignment with many groups. Chen 2026 and Miranda 2025 show models match or trail baselines trained on the same demographics and inflate group gaps. Hu and Collier 2024 show persona variables explain under ten percent of variance. Suh 2025 and Cao 2025 fine-tune on survey data. Recent 2026 papers on the European Social Survey, Singapore and a mechanistic study of Mistral-7B find weak individual-level recovery. The gap: India, what the profile contributes, and held-out-item transfer.")
table(sl, 0.5, 1.45, 12.3, [["Study", "Finding"],
 ["Argyle et al. 2023; Santurkar et al. 2023", "Models reproduce some US demographic-attitude correlations, but misrepresent many groups"],
 ["Chen et al. 2026; Miranda & Balbi 2025", "Match or trail baselines at the individual level; over-attribute attitudes; inflate gaps 2-4x"],
 ["Hu & Collier 2024", "Persona variables explain under 10% of annotation variance"],
 ["Suh et al. 2025; Cao et al. 2025; Choi et al. 2026", "Fine-tuning generalises to unseen subpopulations and surveys, less to unseen questions"],
 ["Wang 2026; Lam et al. 2026; Robbani 2026", "Negligible individual-level recovery (ESS, Singapore); demographics readable but barely used"]],
 [4.4, 7.9], size=14, rowh=0.62)
rect(sl, 0.5, 5.4, 12.3, 1.4, LGT, ORG)
text(sl, 0.7, 5.5, 11.9, 1.2, [[("Research gap:  ", True), ("an India study that separates what the profile contributes from the model's prior, adds a same-model fine-tuned comparison, and tests transfer to questions the model never saw, all against matched classifiers on identical rows.", False)]], size=18)

# 4 Objectives
sl = slide("Objectives and research questions",
 "(45 s) Four research questions. RQ1 asks how a prompted model uses the profile. RQ2 asks what fine-tuning changes compared with zero-shot and matched classifiers. RQ3 asks whether the adaptation transfers. RQ4 asks about scale and data. Scope: two public surveys, one open 7B model fine-tuned, free-tier API models as controls.")
for i, (h, t) in enumerate((("RQ1  USE", "How does a prompted LLM use the demographic profile: how much, and in which direction?"),
                            ("RQ2  ADAPTATION", "What does respondent-level fine-tuning change, relative to zero-shot and to matched classifiers?"),
                            ("RQ3  TRANSFER", "Does the adaptation carry over to questions the model never saw?"),
                            ("RQ4  SCALE & DATA", "Do a larger zero-shot model or more labelled respondents change the picture?"))):
    card(sl, 0.5 + (i % 2) * 6.2, 1.4 + (i // 2) * 2.55, 6.0, 2.35, h, [t], size=19, bullet=False)
text(sl, 0.5, 6.55, 12.3, 0.5, "Scope: Pew Religion in India (N = 29,999) and WVS-7 India (N = 1,692); Qwen2.5-7B-Instruct zero-shot and QLoRA fine-tuned; gpt-oss-120b and Gemini as controls.", size=14, color=GRY)

# 5 Data
sl = slide("Data: two public surveys, codebook-driven item screening",
 "(1 min) Pew Religion in India: 29,999 face-to-face interviews, 2019 to 2020; 73 items passed screening and we use the first eight for fine-tuning and held out items 9 to 16 for transfer. WVS-7 India: 1,692 respondents, 373 candidate items screened to 144 by a codebook-driven pipeline. Five-fold respondent-level splits, fold 0 used here: 23,999 train and 6,000 test respondents. WVS microdata are not redistributed.")
card(sl, 0.5, 1.4, 6.0, 4.0, "PEW RELIGION IN INDIA", ["N = 29,999 adults, face-to-face, Nov 2019 to Mar 2020", "73 items passed screening; 8 used for fine-tuning", "Items 9-16 held out for the transfer test", "Fold 0: 23,999 train / 6,000 test respondents", "46,209 test answers on the trained items", "9 profile attributes in the prompt"], size=16)
card(sl, 6.8, 1.4, 6.0, 4.0, "WVS-7 INDIA (v6.0)", ["N = 1,692 respondents, fieldwork 2023", "373 candidate items screened down to 144", "Screening ledger: codebook match, demographic block, scale size, missingness, entropy", "15 items used for zero-shot replication", "14 profile attributes in the prompt", "Language variable constant, so no language axis"], size=16)
table(sl, 0.5, 5.6, 12.3, [["Design choice", "Why"], ["Respondent-level 5-fold split, out-of-fold scoring", "No respondent appears in both training and test"], ["Same rows for every method", "Differences cannot come from different data"]], [5.2, 7.1], size=13, rowh=0.42)

# 6 System design
sl = slide("System design: one prompt, identical rows, many predictors",
 "(1.5 min) This is the complete system. Both surveys are verbalised into the same prompt from codebook labels, so no hand-typed text. The prompt goes to four kinds of predictors: the zero-shot model, the QLoRA fine-tuned model, larger zero-shot models, and matched classical baselines. The readout is the next-token distribution over the option digits, so we get probabilities, not just a label. Diagnostics and tests sit on top: accuracy, negative log-likelihood, calibration, variance share, directional validity, Jensen-Shannon divergence, subgroup gaps with Holm correction, held-out-item transfer, and compute.")
pic(sl, "fig_framework.png", 0.6, 1.4, w=12.1)
text(sl, 0.6, 6.3, 12.1, 0.7, "Every predictor sees the same prompt and is scored on the same respondent-item rows, so any difference between methods is a difference in the method.", size=17, color=GRY, align=PP_ALIGN.CENTER)

# 7 Prompt + implementation
sl = slide("Implementation: prompt, models and infrastructure",
 "(1.5 min) On the left, the exact prompt format: a profile built from codebook labels, the question, numbered options, and an instruction to answer with a single digit. That wording cut refusals from thirteen percent to zero. On the right, the stack: Python, pandas and scikit-learn for baselines, Hugging Face and PEFT with bitsandbytes for QLoRA, MLX on an Apple M1 for local runs, Kaggle T4 GPUs for full-fold scoring, and free-tier APIs for controls. Every call is cached, so runs are resumable.")
rect(sl, 0.5, 1.4, 6.6, 5.4, LGT, RGBColor(0xb8, 0xc4, 0xe4))
text(sl, 0.6, 1.45, 6.4, 5.3, [[("PROMPT (P2) FOR ONE RESPONDENT-ITEM PAIR", True)],
 "You are a survey respondent with the following demographic profile: Sex: Female. Age: 35-44. Married. Education: literate but no formal schooling. Income: Rs. 10,000 and below. Religion: Hindu. Rural. Region: East. Caste: OBC.",
 "As this person, answer the following survey question.",
 "Question: How much influence should politicians have in religious matters?",
 "Options: 1. A large influence  2. Some influence  3. Not too much influence  4. No influence at all",
 "Respond with exactly one character: 1/2/3/4. No words, no explanation, just that single digit."], size=14, space=7)
card(sl, 7.3, 1.4, 5.5, 5.4, "STACK AND INFRASTRUCTURE", ["Python: pandas, scikit-learn (baselines)", "Hugging Face + PEFT + bitsandbytes: QLoRA fine-tuning", "MLX on Apple M1 (16 GB): local 4-bit inference", "Kaggle 2 x T4 GPUs: full-fold scoring", "Free APIs: gpt-oss-120b (Groq), Gemini", "Caching and resume on every call; refusals recorded, never guessed", "All code, prompts and scripts on GitHub"], size=15)

# 8 Demo
sl = slide("Demo: one respondent, one question, three predictors",
 "(1.5 min, live or recorded) This is one randomly chosen test row, not a cherry-picked one: respondent 27684, question on politicians' influence in religious matters. The true answer is option 1, a large influence. The zero-shot model is confidently wrong: it puts 77 percent on option 3. The fine-tuned model spreads its probability and its top choice is option 2, also wrong, but its distribution is far more reasonable. Gradient boosting gets it right at 49 percent. The point: look at the probabilities, not only the label. For the live part, run the prompt through the model and show the same bar chart.")
pic(sl, "c_demo.png", 0.5, 1.4, w=7.4)
card(sl, 8.1, 1.4, 4.7, 5.4, "HOW TO READ THIS ROW", ["Respondent 27684: rural Hindu woman, 35-44, OBC, East.", "True answer: 1, A large influence.", "Zero-shot: 77% on option 3. Confident and wrong.", "Fine-tuned: spread over options 1-4, top choice 2 (wrong).", "Gradient boosting: 49% on option 1 (right).", "Randomly drawn row (fixed seed), shown as is."], size=15)

# 9 zero-shot results
sl = slide("Result 1: zero-shot uses the profile heavily, but not usefully",
 "(1.5 min) On a 200-respondent subset, zero-shot Qwen scores 17.5 percent, below the 30.2 percent majority class and at chance. Removing the profile raises accuracy to 21.1 percent, so the profile changes accuracy by minus 3.6 points. Yet the profile explains 29 percent of the model's prediction variance against 4 percent for real answers: the model reacts strongly but in the wrong directions. On the full fold of 6,000 respondents it scores 16.1 and 18.0 percent, so this is not a sampling artefact. A label-free prior correction recovers about two thirds of the distance to the majority class.")
pic(sl, "c_subset.png", 0.5, 1.4, w=7.6)
card(sl, 8.3, 1.4, 4.5, 5.4, "KEY NUMBERS", ["Zero-shot: 17.5% vs 30.2% majority (chance 19.4%)", "No profile: 21.1%. The profile lowers accuracy by 3.6 points.", "Profile explains 29% of prediction variance vs 4% in real answers", "Direction r = 0.22, slope 0.18 (weak, attenuated)", "Full fold, 6,000 respondents: 16.1% (chat), 18.0% (raw)"], size=15)

# 10 fine-tuning
sl = slide("Result 2: fine-tuning closes most of the gap, short of the best classifier",
 "(1.5 min) QLoRA fine-tuning of Qwen2.5-7B on 184,853 examples took 6.6 GPU-hours. Scored on the full fold in the training format, it reaches 38.0 percent: 7.6 points above the majority class, half a point below logistic regression and 2.1 below gradient boosting, closing 79 percent of the gap. It is well calibrated, with expected calibration error 0.009. One earlier worry, an apparent collapse on one item, turned out to be a decoding artefact of the original run, not of the adapter.")
pic(sl, "c_fold0.png", 0.5, 1.4, w=7.6)
table(sl, 8.3, 1.5, 4.5, [["Method", "Acc.", "NLL", "ECE"], ["Majority", "30.4", "1.59", "0.001"], ["Logistic", "38.5", "1.47", "0.004"], ["GBM", "40.0", "1.44", "0.005"], ["Fine-tuned", "38.0", "1.48", "0.009"], ["Zero-shot", "16.1", "4.26", "0.519"]], [1.7, 0.9, 0.9, 1.0], size=13, rowh=0.4)
text(sl, 8.3, 4.1, 4.5, 2.7, [[("Fine-tuned model", True)], "+7.6 points over majority", "-0.5 vs logistic, -2.1 vs GBM", "Closes 79% of the gap to GBM", "Trails GBM on every item (1.2 to 2.7 points)", "ECE 0.009: well calibrated"], size=15, bullet=False, space=4)

# 11 diagnostics
sl = slide("Result 3: direction of group differences and subgroup fairness",
 "(1.5 min) Directional validity asks whether the model gets the direction and size of between-group differences right. Zero-shot has r of 0.22 and a slope of 0.18: nearly flat. Fine-tuned reaches 0.58 on the subset and 0.82 on the full fold, with GBM at 0.64 and 0.91. Subgroup gaps: religion 10 points, region 8.8, age 2.7, income 1.7, urban rural 1.3, all significant after Holm correction. On probability distributions, the fine-tuned model is about two to three times further from the truth than classifiers but more than ten times closer than zero-shot.")
pic(sl, "fig_direction.png", 1.15, 1.3, w=11.0)
card(sl, 0.5, 5.05, 6.0, 1.95, "SUBGROUP GAPS (FINE-TUNED, POINTS)", ["Religion 10.0, region 8.8, age 2.7, income 1.7, urban/rural 1.3", "Five of eight axes significant after Holm correction"], size=14)
card(sl, 6.8, 5.05, 6.0, 1.95, "DISTRIBUTION FIDELITY (JSD, RELIGION CELLS)", ["Fine-tuned 0.009, logistic 0.004, item marginal 0.029", "Zero-shot 0.271: more than 10x worse than fine-tuned"], size=14)

# 12 transfer and scale
sl = slide("Result 4: transfer to unseen questions, scale and data",
 "(1.5 min) The transfer test applies the adapter to eight Pew items it never saw. With our approximate MLX port the adapter scores 28.9 percent against 28.5 percent for its base: no detectable gain. The in-domain gain is 13.7 points larger over respondents, but across items the difference has a standard error of 10.2 and a permutation p of 0.18, so we do not claim it is reliable. Scale: gpt-oss-120b scores 27.6 percent, at the majority class, not beyond. Data: logistic regression needs about 400 labelled respondents, GBM about 1,600, to clear the majority class.")
pic(sl, "fig_transfer.png", 0.5, 1.35, w=7.4)
pic(sl, "fig_lc.png", 8.5, 1.3, w=3.9)
card(sl, 0.5, 4.2, 7.4, 2.75, "TRANSFER (8 UNSEEN ITEMS, 120 RESPONDENTS)", ["Adapter 28.9% vs base 28.5%: +0.4 points (-5.1 to +5.9)", "In-domain gain +14.1 points; item-level difference not significant (p = 0.18)", "Approximate port, 8 items: not proof that transfer is impossible"], size=14)
card(sl, 8.1, 3.95, 4.7, 3.0, "SCALE AND DATA", ["gpt-oss-120b: 27.6% on 100 respondents (majority 30.0%)", "Logistic needs about 400 respondents, GBM about 1,600, to beat majority", "Classifiers are far cheaper per row"], size=14)

# 13 compute
sl = slide("Computational requirements and scalability",
 "(1 min) We measured the cost. Scoring one condition on the full fold, 46,209 rows, took about 75 minutes on two T4 GPUs, roughly 0.2 seconds per row. The lab fine-tuning run took 396.7 minutes. Cost is linear in respondent-item rows: scoring all 29,999 respondents on all 73 items is about 2.2 million rows, roughly 116 T4-hours at our measured rate, an extrapolation. Fine-tuning on all items would take about 63 hours at the lab GPU's rate. A classifier trains in seconds and predicts in microseconds.")
table(sl, 0.5, 1.45, 12.3, [["Task", "Hardware", "Measured cost"],
 ["QLoRA fine-tuning, 184,853 examples", "One lab GPU", "5,777 steps, 6.6 h (396.7 min)"],
 ["Scoring one condition, 46,209 rows", "Two Kaggle T4, fp16", "About 75 min; 0.19-0.21 s per row per GPU"],
 ["Local zero-shot scoring", "Apple M1, 16 GB, 4-bit MLX", "2.3 s per prompt; about 5 h per 1,000 respondents"],
 ["Matched classical baselines, 8 items", "Laptop CPU", "Under 1 minute for the whole fold"],
 ["gpt-oss-120b, 774 rows", "Free API tier", "Rate-limited; several hours"]], [4.7, 3.4, 4.2], size=14, rowh=0.55)
card(sl, 0.5, 5.0, 12.3, 1.95, "SCALING (LINEAR EXTRAPOLATIONS, NOT MEASUREMENTS)", ["All 29,999 respondents x 73 items = 2.2 million rows: about 116 T4-hours to score (about 58 h on two)", "Fine-tuning on all 73 items (1.75 million examples): about 63 h at the lab GPU rate, about 290 h on a T4"], size=15)

# 14 WVS and lessons
sl = slide("Track A (WVS-7) replication and engineering lessons",
 "(1 min) The first phase used free-tier APIs on WVS-7 India. Accuracy stays in a 22 to 26 percent band, below or at the majority class on the same rows, and group differences carry no directional signal. We also found and fixed real bugs: region labels in the prompt were wrong on every code until verified against the official annex, which moved subgroup gaps by several points. Caching had treated transient failures as final. A decoding artefact also explained an apparent collapse in the original fine-tuning run. Verification mattered more than speed.")
table(sl, 0.5, 1.45, 6.4, [["WVS-7 India, 15 items", "LLM acc.", "Majority"], ["Gemini 3.1 Flash Lite (30 resp.)", "25.7%", "20.9%"], ["gpt-oss-120b (40 resp.)", "23.0%", "22.3%"], ["Qwen2.5-7B (120 resp.)", "22.1%", "24.9%"], ["GBM (14 attributes)", "34-38%", "-"]], [3.6, 1.5, 1.3], size=14, rowh=0.5)
card(sl, 7.1, 1.4, 5.7, 5.4, "LESSONS FROM BUILDING IT", ["Region labels were wrong on every code until checked against the WVS annex; fixing it moved gaps by several points", "A cache that treated failures as final silently re-served 'no answer'", "An apparent collapse on one item was a decoding artefact, found by re-scoring the adapter", "Prompt wording cut refusals from 13% to 0%", "Always report the majority baseline and identical-row comparisons"], size=15)
text(sl, 0.5, 4.3, 6.4, 2.5, [[("WVS findings", True)], "LLM accuracy 22-26%, at or below the majority class on the same rows", "Group differences show no directional signal (r = -0.03)", "No subgroup gap distinguishable from chance"], size=15, bullet=False, space=5)

# 15 limitations & future
sl = slide("Limitations and future work",
 "(1 min) We are honest about limits: one fold, one seed, one epoch; the transfer test uses an approximate port and only eight items; zero-shot controls on subsets; no frontier models; survey weights not applied. Future work: fine-tune on many items and test transfer on dozens of unseen items with an item-level bootstrap using the original adapter, repeat across folds and seeds, test richer personas and calibrated aggregate estimates, and measure the cost side including smaller and distilled models.")
card(sl, 0.5, 1.4, 6.0, 5.4, "LIMITATIONS", ["One fold, one seed, one epoch", "Transfer test: approximate MLX port, 8 items, 120 respondents", "Zero-shot controls on 100-200 respondent subsets", "No frontier model tested on these surveys", "Pew survey weights not applied", "Exact-match accuracy; scoring in fp16 on T4 vs bf16 training"], size=16)
card(sl, 6.8, 1.4, 6.0, 5.4, "FUTURE WORK", ["Fine-tune on many items; test transfer on dozens of unseen items", "Repeat across folds and seeds; use the original adapter for transfer", "Richer text-conditioned personas", "Calibrated aggregate estimates, not argmax", "Smaller and distilled models; energy and cost measurement"], size=16)

# 16 conclusion
sl = slide("Conclusion, publication and Q&A",
 "(1 min) Take-home: a prompted 7B model varies strongly with a profile but uses it weakly and slightly harmfully; fine-tuning on respondent-level data recovers most of the gap to classifiers on trained questions and shows no detectable gain on new ones; matched classifiers remain the benchmark and are far cheaper. The paper from this work was accepted at the Hinweis Fifth International Conference on AI and Data Science, AIDE-2026, paper ID 591, accepted with modification and revised for camera-ready. Invite questions.")
card(sl, 0.5, 1.4, 7.4, 5.4, "CONCLUSIONS", ["A prompted 7B model varies strongly with the profile but uses it weakly and slightly harmfully", "Fine-tuning recovers 79% of the gap to the best classifier: 38.0% vs 40.0% (GBM)", "No detectable gain on unseen questions in our test", "Matched classifiers remain the benchmark, and are far cheaper to run", "Check direction, calibration and subgroup gaps, not only accuracy"], size=17)
card(sl, 8.1, 1.4, 4.7, 3.0, "PUBLICATION", ["Hinweis 5th Int. Conf. on AI and Data Science (AIDE-2026), Paper ID 591", "Accepted with modification; camera-ready submitted"], size=15)
card(sl, 8.1, 4.6, 4.7, 2.2, "THANK YOU", ["Questions?", "Code: github.com/DevilHunter695/silicon-sampling-capstone"], size=15)
prs.save(sys.argv[1]); print("saved", sys.argv[1], TOTAL[0], "slides"); print("OVERFLOW WARNINGS:", WARN)
