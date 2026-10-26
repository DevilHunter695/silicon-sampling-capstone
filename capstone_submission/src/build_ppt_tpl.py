import sys, copy
from PIL import ImageFont
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
TEMPLATE, OUT = sys.argv[1], sys.argv[2]
RED, DK, LGT, GRY, WHT = RGBColor(0xBC, 0x31, 0x2C), RGBColor(0x1A, 0x1A, 0x1A), RGBColor(0xF3, 0xF3, 0xF3), RGBColor(0x55, 0x55, 0x55), RGBColor(255, 255, 255)
FONT = "Times New Roman"
FR = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"; FB = "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"
_fc = {}
def _font(sz, b):
    k = (round(sz * 4), b)
    if k not in _fc: _fc[k] = ImageFont.truetype(FB if b else FR, int(round(sz * 4)))
    return _fc[k]
def need_h(paras, w_in, size, bullet, bold, space):
    wpt = (w_in - 0.16) * 72 * 4; tot = 0.0
    for p in (paras if isinstance(paras, list) else [paras]):
        runs = p if isinstance(p, list) else [(p, bold)]
        if bullet: runs = [("•  ", False)] + runs
        words = [(w, b) for t, b in runs for w in t.split(" ")]
        lines, cur = 1, 0.0
        for w, b in words:
            wl = _font(size, b).getlength(w + " ")
            if cur + wl > wpt and cur > 0: lines += 1; cur = wl
            else: cur += wl
        tot += lines * size * 1.17 + space
    return (tot + 6) / 72
WARN = []
prs = Presentation(TEMPLATE)
SW, SH = prs.slide_width / 914400, prs.slide_height / 914400
N = [1]
def rect(sl, x, y, w, h, fill, line=None):
    s = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)); s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None: s.line.fill.background()
    else: s.line.color.rgb = line; s.line.width = Pt(0.75)
    s.shadow.inherit = False; return s
def text(sl, x, y, w, h, paras, size=14, color=DK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, bullet=False, space=4, minsize=9.5):
    if isinstance(paras, str): paras = [paras]
    sz = size
    while need_h(paras, w, sz, bullet, bold, space) > h and sz > minsize: sz -= 0.5
    if need_h(paras, w, sz, bullet, bold, space) > h: WARN.append((N[0], str(paras)[:45], round(need_h(paras, w, sz, bullet, bold, space), 2), h))
    tb = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.08); tf.margin_top = tf.margin_bottom = Inches(0.03)
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); para.alignment = align; para.space_after = Pt(space)
        runs = p if isinstance(p, list) else [(p, bold)]
        if bullet: runs = [("•  ", False)] + runs
        for t, b in runs:
            r = para.add_run(); r.text = t; r.font.size = Pt(sz); r.font.bold = b; r.font.color.rgb = color; r.font.name = FONT
    return tb
def pic(sl, path, x, y, w=None, h=None): return sl.shapes.add_picture(path, Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)
def card(sl, x, y, w, h, head, paras, size=13, bullet=True):
    rect(sl, x, y, w, h, LGT, RGBColor(0xC8, 0xC8, 0xC8)); rect(sl, x, y, w, 0.34, RED)
    text(sl, x + 0.06, y + 0.01, w - 0.12, 0.32, head, size=12, color=WHT, bold=True, anchor=MSO_ANCHOR.MIDDLE, minsize=9)
    text(sl, x + 0.08, y + 0.4, w - 0.16, h - 0.45, paras, size=size, bullet=bullet, space=3)
def table(sl, x, y, w, rows, colw, size=11, rowh=0.34):
    sh = sl.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(rowh * len(rows))); t = sh.table
    for ci, cw in enumerate(colw): t.columns[ci].width = Inches(cw)
    for ri, row in enumerate(rows):
        t.rows[ri].height = Inches(rowh)
        for ci, val in enumerate(row):
            c = t.cell(ri, ci); c.text = ""; p = c.text_frame.paragraphs[0]; r = p.add_run(); r.text = val; r.font.size = Pt(size); r.font.name = FONT
            r.font.bold = (ri == 0); r.font.color.rgb = WHT if ri == 0 else DK
            p.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER; c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.margin_left = c.margin_right = Inches(0.05); c.margin_top = c.margin_bottom = Inches(0.02)
            c.fill.solid(); c.fill.fore_color.rgb = RED if ri == 0 else (LGT if ri % 2 else WHT)
    return t
def footer(sl):
    text(sl, 2.2, 6.97, 5.6, 0.32, "Computer Engineering Department, MPSTME, Mumbai Campus  |  Group 55", size=10.5, color=DK, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(sl, 8.3, 6.97, 1.2, 0.32, str(N[0]), size=11, color=DK, align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
def new_slide(title, notes):
    N[0] += 1
    sl = prs.slides.add_slide(prs.slide_layouts[2])
    for ph in list(sl.placeholders):
        if ph.placeholder_format.type != 1: ph._element.getparent().remove(ph._element)   # keep only the title
    t = sl.shapes.title; t.left, t.top, t.width, t.height = Inches(0.5), Inches(1.5), Inches(9.0), Inches(0.6)
    tf = t.text_frame; tf.clear(); tf.vertical_anchor = MSO_ANCHOR.MIDDLE; p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
    sz = 24
    while need_h([[(title, True)]], 9.0, sz, False, True, 0) > 0.6 and sz > 16: sz -= 1
    r = p.add_run(); r.text = title; r.font.size = Pt(sz); r.font.bold = True; r.font.name = FONT; r.font.color.rgb = DK
    rect(sl, 0.5, 2.12, 9.0, 0.04, RED)
    footer(sl); sl.notes_slide.notes_text_frame.text = notes
    return sl

# ---------------- slide 1: reuse the template's title slide ----------------
s1 = prs.slides[0]
for sh in s1.shapes:
    if sh.is_placeholder and sh.placeholder_format.type == 3:   # center title
        sh.left, sh.top, sh.width, sh.height = Inches(0.75), Inches(1.55), Inches(8.5), Inches(2.45)
        tf = sh.text_frame; tf.clear(); tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        lines = [("Capstone Project", 32, True), ("Final Presentation", 32, True), ("A.Y. 2026-2027", 22, False), ("Project Title: Silicon Sampling for Indian Public Opinion", 22, True)]
        for i, (t, sz, b) in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = PP_ALIGN.CENTER
            r = p.add_run(); r.text = t; r.font.size = Pt(sz); r.font.bold = b; r.font.name = FONT; r.font.color.rgb = DK
    if sh.is_placeholder and sh.placeholder_format.type == 4:   # subtitle
        sh.left, sh.top, sh.width, sh.height = Inches(0.9), Inches(4.1), Inches(8.2), Inches(2.7)
        tf = sh.text_frame; tf.clear()
        rows = [("Presented by: Group 55", True, 18), ("Bhanuj Bhalla (SAP 70022300017)     Ishaan Bansal (SAP 70022300011)", False, 15), ("Yash Bhende (SAP 70022300137)     Krishank Ayyella (SAP 70022300136)", False, 15), ("", False, 8), ("Under the guidance of: Ms. Khushbu Chauhan", True, 18)]
        for i, (t, b, sz) in enumerate(rows):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = PP_ALIGN.CENTER
            r = p.add_run(); r.text = t; r.font.size = Pt(sz); r.font.bold = b; r.font.name = FONT; r.font.color.rgb = DK
s1.notes_slide.notes_text_frame.text = "(30 s) Introduce the group, the mentor and the project. One sentence on what silicon sampling is: using a language model as a stand-in survey respondent."

# ---------------- content slides (4:3, inside the template's safe area x 0.5-9.5, y 2.25-6.9) ----------------
Y0 = 2.3
sl = new_slide("Motivation: can a language model stand in for a survey respondent?",
 "(1 min) Silicon sampling gives an LLM a demographic profile and treats its answer as that person's survey answer. If it worked it would cut cost and time. The danger is that a good average can hide poor performance for specific groups, which matters in a country as diverse as India. Our question is what the profile actually contributes, not just the headline accuracy.")
card(sl, 0.5, Y0, 2.95, 4.5, "THE IDEA", ["Give an LLM a demographic profile.", "Ask the survey question.", "Treat its answer as that person's answer.", "Promise: far cheaper and faster than fielding a questionnaire."], size=14)
card(sl, 3.58, Y0, 2.95, 4.5, "THE RISK", ["A good average can hide unequal performance across groups.", "India is highly diverse: region, religion, caste, income, age.", "Past work mostly reports aggregate fidelity, mostly for the US."], size=14)
card(sl, 6.65, Y0, 2.85, 4.5, "OUR QUESTION", ["What does the profile itself contribute?", "Does fine-tuning on respondent-level data help?", "Does the help carry over to unseen questions?", "How does it compare with a simple classifier, and at what cost?"], size=14)

sl = new_slide("Literature review and the gap",
 "(1 min) Argyle 2023 introduced algorithmic fidelity on US data. Santurkar 2023 found misalignment with many groups. Chen 2026 and Miranda 2025 show models match or trail baselines trained on the same demographics and inflate group gaps. Hu and Collier 2024 show persona variables explain under ten percent of variance. Suh 2025 and Cao 2025 fine-tune on survey data. Recent 2026 papers find weak individual-level recovery. The gap: India, what the profile contributes, and held-out-item transfer.")
table(sl, 0.5, Y0, 9.0, [["Study", "Finding"],
 ["Argyle 2023; Santurkar 2023", "Reproduce some US correlations, but misrepresent many groups"],
 ["Chen 2026; Miranda & Balbi 2025", "Match or trail baselines; over-attribute attitudes; inflate gaps 2-4x"],
 ["Hu & Collier 2024", "Persona variables explain under 10% of variance"],
 ["Suh 2025; Cao 2025; Choi 2026", "Fine-tuning generalises to unseen subpopulations, less to unseen questions"],
 ["Wang 2026; Lam 2026; Robbani 2026", "Weak individual-level recovery; demographics readable but barely used"]], [3.0, 6.0], size=11.5, rowh=0.52)
rect(sl, 0.5, 5.6, 9.0, 1.2, LGT, RED)
text(sl, 0.6, 5.65, 8.8, 1.1, [[("Research gap:  ", True), ("an India study that separates what the profile contributes from the model's prior, adds a same-model fine-tuned comparison, and tests transfer to unseen questions, all against matched classifiers on identical rows.", False)]], size=14)

sl = new_slide("Objectives and research questions",
 "(45 s) Four research questions. RQ1 asks how a prompted model uses the profile. RQ2 asks what fine-tuning changes compared with zero-shot and matched classifiers. RQ3 asks whether the adaptation transfers. RQ4 asks about scale and data. Scope: two public surveys, one open 7B model fine-tuned, free-tier API models as controls.")
for i, (h, t) in enumerate((("RQ1  USE", "How does a prompted LLM use the demographic profile: how much, and in which direction?"), ("RQ2  ADAPTATION", "What does respondent-level fine-tuning change, relative to zero-shot and to matched classifiers?"),
                            ("RQ3  TRANSFER", "Does the adaptation carry over to questions the model never saw?"), ("RQ4  SCALE & DATA", "Do a larger zero-shot model or more labelled respondents change the picture?"))):
    card(sl, 0.5 + (i % 2) * 4.6, Y0 + (i // 2) * 1.95, 4.4, 1.8, h, [t], size=14, bullet=False)
text(sl, 0.5, 6.2, 9.0, 0.7, "Scope: Pew Religion in India (N = 29,999) and WVS-7 India (N = 1,692); Qwen2.5-7B-Instruct zero-shot and QLoRA fine-tuned; gpt-oss-120b and Gemini as controls.", size=12, color=GRY)

sl = new_slide("Data: two public surveys, codebook-driven item screening",
 "(1 min) Pew Religion in India: 29,999 face-to-face interviews, 2019 to 2020; 73 items passed screening and we use the first eight for fine-tuning and held out items 9 to 16 for transfer. WVS-7 India: 1,692 respondents, 373 candidate items screened to 144 by a codebook-driven pipeline. Five-fold respondent-level splits, fold 0 used here: 23,999 train and 6,000 test respondents. WVS microdata are not redistributed.")
card(sl, 0.5, Y0, 4.4, 3.2, "PEW RELIGION IN INDIA", ["N = 29,999 adults, face-to-face, 2019-20", "73 items screened; 8 used for fine-tuning", "Items 9-16 held out for transfer", "Fold 0: 23,999 train / 6,000 test", "46,209 test answers; 9 profile attributes"], size=13)
card(sl, 5.1, Y0, 4.4, 3.2, "WVS-7 INDIA (v6.0)", ["N = 1,692 respondents, fieldwork 2023", "373 candidate items screened to 144", "15 items used for zero-shot replication", "14 profile attributes; language constant", "Microdata not redistributed"], size=13)
table(sl, 0.5, 5.65, 9.0, [["Design choice", "Why"], ["Respondent-level 5-fold split, out-of-fold scoring", "No respondent is in both training and test"], ["Same rows for every method", "Differences cannot come from different data"]], [4.4, 4.6], size=11.5, rowh=0.38)

sl = new_slide("System design: one prompt, identical rows, many predictors",
 "(1.5 min) This is the complete system. Both surveys are verbalised into the same prompt from codebook labels, so no hand-typed text. The prompt goes to four kinds of predictors: the zero-shot model, the QLoRA fine-tuned model, larger zero-shot models, and matched classical baselines. The readout is the next-token distribution over the option digits, so we get probabilities, not just a label. Diagnostics and tests sit on top: accuracy, negative log-likelihood, calibration, variance share, directional validity, Jensen-Shannon divergence, subgroup gaps with Holm correction, held-out-item transfer, and compute.")
pic(sl, "fig_framework.png", 0.5, Y0 + 0.1, w=9.0)
text(sl, 0.5, 6.15, 9.0, 0.7, "Every predictor sees the same prompt and is scored on the same respondent-item rows, so any difference between methods is a difference in the method.", size=13, color=GRY, align=PP_ALIGN.CENTER)

sl = new_slide("Implementation: prompt, models and infrastructure",
 "(1.5 min) On the left, the exact prompt format: a profile built from codebook labels, the question, numbered options, and an instruction to answer with a single digit. That wording cut refusals from thirteen percent to zero. On the right, the stack: Python, pandas and scikit-learn for baselines, Hugging Face and PEFT with bitsandbytes for QLoRA, MLX on an Apple M1 for local runs, Kaggle T4 GPUs for full-fold scoring, and free-tier APIs for controls. Every call is cached, so runs are resumable.")
rect(sl, 0.5, Y0, 4.6, 4.5, LGT, RGBColor(0xC8, 0xC8, 0xC8))
text(sl, 0.55, Y0 + 0.03, 4.5, 4.45, [[("PROMPT (P2), ONE RESPONDENT-ITEM PAIR", True)],
 "You are a survey respondent with the following demographic profile: Sex: Female. Age: 35-44. Married. Education: literate, no formal schooling. Income: Rs. 10,000 and below. Religion: Hindu. Rural. Region: East. Caste: OBC.",
 "As this person, answer the following survey question.", "Question: How much influence should politicians have in religious matters?",
 "Options: 1. A large influence  2. Some influence  3. Not too much influence  4. No influence at all", "Respond with exactly one character: 1/2/3/4. No words, no explanation, just that single digit."], size=12, space=5)
card(sl, 5.3, Y0, 4.2, 4.5, "STACK AND INFRASTRUCTURE", ["Python: pandas, scikit-learn (baselines)", "Hugging Face + PEFT + bitsandbytes: QLoRA", "MLX on Apple M1 (16 GB): local 4-bit inference", "Kaggle 2 x T4: full-fold scoring", "Free APIs: gpt-oss-120b (Groq), Gemini", "Caching and resume on every call", "All code and prompts on GitHub"], size=13)

sl = new_slide("Demo: one respondent, one question, three predictors",
 "(1.5 min, live or recorded) This is one randomly chosen test row, not a cherry-picked one: respondent 27684, question on politicians' influence in religious matters. The true answer is option 1, a large influence. The zero-shot model is confidently wrong: it puts 77 percent on option 3. The fine-tuned model spreads its probability and its top choice is option 2, also wrong, but its distribution is far more reasonable. Gradient boosting gets it right at 49 percent. The point: look at the probabilities, not only the label.")
pic(sl, "c_demo.png", 0.5, Y0 + 0.2, w=5.5)
card(sl, 6.15, Y0, 3.35, 4.5, "HOW TO READ THIS ROW", ["Respondent 27684: rural Hindu woman, 35-44, OBC, East.", "True answer: 1, A large influence.", "Zero-shot: 77% on option 3. Confident and wrong.", "Fine-tuned: spread over 1-4, top choice 2 (wrong).", "Gradient boosting: 49% on option 1 (right).", "Randomly drawn row, shown as is."], size=12)
text(sl, 0.5, 5.7, 5.5, 1.1, "Live demo: build the prompt for any respondent-item pair, run the model, and plot the probability bars.", size=12, color=GRY)

sl = new_slide("Result 1: zero-shot uses the profile heavily, but not usefully",
 "(1.5 min) On a 200-respondent subset, zero-shot Qwen scores 17.5 percent, below the 30.2 percent majority class and at chance. Removing the profile raises accuracy to 21.1 percent, so the profile changes accuracy by minus 3.6 points. Yet the profile explains 29 percent of the model's prediction variance against 4 percent for real answers: the model reacts strongly but in the wrong directions. On the full fold of 6,000 respondents it scores 16.1 and 18.0 percent, so this is not a sampling artefact. A label-free prior correction recovers about two thirds of the distance to the majority class.")
pic(sl, "c_subset.png", 0.5, Y0 + 0.1, w=5.5)
card(sl, 6.15, Y0, 3.35, 4.5, "KEY NUMBERS", ["Zero-shot 17.5% vs 30.2% majority (chance 19.4%)", "No profile: 21.1%; the profile lowers accuracy by 3.6 points", "Profile explains 29% of variance vs 4% in real answers", "Direction r = 0.22, slope 0.18", "Full fold: 16.1% (chat), 18.0% (raw)"], size=12.5)

sl = new_slide("Result 2: fine-tuning closes most of the gap to the best classifier",
 "(1.5 min) QLoRA fine-tuning of Qwen2.5-7B on 184,853 examples took 6.6 GPU-hours. Scored on the full fold in the training format, it reaches 38.0 percent: 7.6 points above the majority class, half a point below logistic regression and 2.1 below gradient boosting, closing 79 percent of the gap. It is well calibrated, with expected calibration error 0.009. One earlier worry, an apparent collapse on one item, turned out to be a decoding artefact of the original run, not of the adapter.")
pic(sl, "c_fold0.png", 0.5, Y0 + 0.1, w=5.4)
table(sl, 6.05, Y0, 3.45, [["Method", "Acc.", "NLL", "ECE"], ["Majority", "30.4", "1.59", "0.001"], ["Logistic", "38.5", "1.47", "0.004"], ["GBM", "40.0", "1.44", "0.005"], ["Fine-tuned", "38.0", "1.48", "0.009"], ["Zero-shot", "16.1", "4.26", "0.519"]], [1.2, 0.7, 0.7, 0.85], size=11, rowh=0.36)
text(sl, 6.05, 4.6, 3.45, 2.2, ["+7.6 points over majority", "-0.5 vs logistic, -2.1 vs GBM", "Closes 79% of the gap to GBM", "Trails GBM on every item (1.2 to 2.7 points)", "ECE 0.009: well calibrated"], size=12, bullet=True, space=3)

sl = new_slide("Result 3: direction of group differences and subgroup gaps",
 "(1.5 min) Directional validity asks whether the model gets the direction and size of between-group differences right. Zero-shot has r of 0.22 and a slope of 0.18: nearly flat. Fine-tuned reaches 0.58 on the subset and 0.82 on the full fold, with GBM at 0.64 and 0.91. Subgroup gaps: religion 10 points, region 8.8, age 2.7, income 1.7, urban rural 1.3, all significant after Holm correction. On probability distributions, the fine-tuned model is about two to three times further from the truth than classifiers but more than ten times closer than zero-shot.")
pic(sl, "fig_direction.png", 0.5, Y0 + 0.05, w=9.0)
card(sl, 0.5, 5.4, 4.4, 1.45, "SUBGROUP GAPS (FINE-TUNED)", ["Religion 10.0, region 8.8, age 2.7, income 1.7, urban/rural 1.3 points", "Five of eight axes significant (Holm)"], size=11.5)
card(sl, 5.1, 5.4, 4.4, 1.45, "DISTRIBUTION FIDELITY (JSD)", ["Fine-tuned 0.009, logistic 0.004, marginal 0.029", "Zero-shot 0.271: over 10x worse than fine-tuned"], size=11.5)

sl = new_slide("Result 4: transfer to unseen questions, scale and data",
 "(1.5 min) The transfer test applies the adapter to eight Pew items it never saw. With our approximate MLX port the adapter scores 28.9 percent against 28.5 percent for its base: no detectable gain. The in-domain gain is 13.7 points larger over respondents, but across items the difference has a standard error of 10.2 and a permutation p of 0.18, so we do not claim it is reliable. Scale: gpt-oss-120b scores 27.6 percent, at the majority class, not beyond. Data: logistic regression needs about 400 labelled respondents, GBM about 1,600, to clear the majority class.")
pic(sl, "fig_transfer.png", 0.5, Y0 + 0.05, w=5.7)
pic(sl, "fig_lc.png", 6.4, Y0 + 0.0, w=3.1)
card(sl, 0.5, 4.5, 5.7, 2.35, "TRANSFER (8 UNSEEN ITEMS, 120 RESPONDENTS)", ["Adapter 28.9% vs base 28.5%: +0.4 points (-5.1 to +5.9)", "In-domain gain +14.1; item-level difference not significant (p = 0.18)", "Approximate port, 8 items: not proof transfer is impossible"], size=11.5)
card(sl, 6.4, 4.2, 3.1, 2.65, "SCALE AND DATA", ["gpt-oss-120b: 27.6% (majority 30.0%)", "Logistic needs ~400 respondents, GBM ~1,600, to beat majority"], size=11.5)

sl = new_slide("Computational requirements and scalability",
 "(1 min) We measured the cost. Scoring one condition on the full fold, 46,209 rows, took about 75 minutes on two T4 GPUs, roughly 0.2 seconds per row. The lab fine-tuning run took 396.7 minutes. Cost is linear in respondent-item rows: scoring all 29,999 respondents on all 73 items is about 2.2 million rows, roughly 116 T4-hours at our measured rate, an extrapolation. Fine-tuning on all items would take about 63 hours at the lab GPU's rate. A classifier trains in seconds and predicts in microseconds.")
table(sl, 0.5, Y0, 9.0, [["Task", "Hardware", "Measured cost"],
 ["QLoRA fine-tuning, 184,853 examples", "One lab GPU", "5,777 steps, 6.6 h (396.7 min)"],
 ["Scoring one condition, 46,209 rows", "Two Kaggle T4, fp16", "About 75 min; 0.19-0.21 s per row per GPU"],
 ["Local zero-shot scoring", "Apple M1, 16 GB, 4-bit", "2.3 s per prompt; about 5 h per 1,000 respondents"],
 ["Classical baselines, 8 items", "Laptop CPU", "Under 1 minute for the whole fold"],
 ["gpt-oss-120b, 774 rows", "Free API tier", "Rate-limited; several hours"]], [3.3, 2.1, 3.6], size=11, rowh=0.5)
card(sl, 0.5, 5.45, 9.0, 1.4, "SCALING (LINEAR EXTRAPOLATIONS, NOT MEASUREMENTS)", ["All 29,999 respondents x 73 items = 2.2 million rows: about 116 T4-hours to score (about 58 h on two)", "Fine-tuning on all 73 items (1.75 million examples): about 63 h at the lab GPU rate, about 290 h on a T4"], size=11.5)

sl = new_slide("WVS-7 replication and engineering lessons",
 "(1 min) The first phase used free-tier APIs on WVS-7 India. Accuracy stays in a 22 to 26 percent band, below or at the majority class on the same rows, and group differences carry no directional signal. We also found and fixed real bugs: region labels in the prompt were wrong on every code until verified against the official annex, which moved subgroup gaps by several points. Caching had treated transient failures as final. A decoding artefact also explained an apparent collapse in the original fine-tuning run. Verification mattered more than speed.")
table(sl, 0.5, Y0, 4.4, [["WVS-7, 15 items", "LLM", "Majority"], ["Gemini 3.1 Flash Lite (30)", "25.7%", "20.9%"], ["gpt-oss-120b (40)", "23.0%", "22.3%"], ["Qwen2.5-7B (120)", "22.1%", "24.9%"], ["GBM (14 attributes)", "34-38%", "-"]], [2.4, 1.0, 1.0], size=11, rowh=0.42)
text(sl, 0.5, 4.55, 4.4, 2.3, [[("WVS findings", True)], "LLM accuracy 22-26%, at or below the majority class on the same rows", "Group differences show no directional signal (r = -0.03)", "No subgroup gap distinguishable from chance"], size=12, space=4)
card(sl, 5.1, Y0, 4.4, 4.5, "LESSONS FROM BUILDING IT", ["Region labels were wrong on every code until checked against the WVS annex; fixing it moved gaps by several points", "A cache that treated failures as final silently re-served 'no answer'", "An apparent collapse on one item was a decoding artefact", "Prompt wording cut refusals from 13% to 0%", "Always report the majority baseline"], size=12)

sl = new_slide("Limitations and future work",
 "(1 min) We are honest about limits: one fold, one seed, one epoch; the transfer test uses an approximate port and only eight items; zero-shot controls on subsets; no frontier models; survey weights not applied. Future work: fine-tune on many items and test transfer on dozens of unseen items with an item-level bootstrap using the original adapter, repeat across folds and seeds, test richer personas and calibrated aggregate estimates, and measure the cost side including smaller and distilled models.")
card(sl, 0.5, Y0, 4.4, 4.5, "LIMITATIONS", ["One fold, one seed, one epoch", "Transfer: approximate MLX port, 8 items, 120 respondents", "Zero-shot controls on 100-200 respondent subsets", "No frontier model on these surveys", "Pew survey weights not applied", "Exact-match accuracy; fp16 scoring vs bf16 training"], size=13)
card(sl, 5.1, Y0, 4.4, 4.5, "FUTURE WORK", ["Fine-tune on many items; test transfer on dozens of unseen items", "Repeat across folds and seeds; use the original adapter", "Richer text-conditioned personas", "Calibrated aggregate estimates, not argmax", "Smaller and distilled models; energy and cost"], size=13)

sl = new_slide("Conclusion, publication and Q&A",
 "(1 min) Take-home: a prompted 7B model varies strongly with a profile but uses it weakly and slightly harmfully; fine-tuning on respondent-level data recovers most of the gap to classifiers on trained questions and shows no detectable gain on new ones; matched classifiers remain the benchmark and are far cheaper. The paper from this work was accepted at the Hinweis Fifth International Conference on AI and Data Science, AIDE-2026, paper ID 591, accepted with modification and revised for camera-ready. Invite questions.")
card(sl, 0.5, Y0, 5.4, 4.5, "CONCLUSIONS", ["A prompted 7B model varies strongly with the profile but uses it weakly and slightly harmfully", "Fine-tuning recovers 79% of the gap to the best classifier: 38.0% vs 40.0%", "No detectable gain on unseen questions in our test", "Matched classifiers remain the benchmark, and are far cheaper", "Check direction, calibration and subgroup gaps, not only accuracy"], size=13)
card(sl, 6.05, Y0, 3.45, 2.5, "PUBLICATION", ["Hinweis AIDE-2026, Paper ID 591", "Accepted with modification; camera-ready sent"], size=12)
card(sl, 6.05, 4.9, 3.45, 1.95, "THANK YOU", ["Questions?", "github.com/DevilHunter695/silicon-sampling-capstone"], size=11.5)
prs.save(OUT); print("saved", OUT, len(prs.slides), "slides"); print("OVERFLOW:", [w for w in WARN if w[1] and not w[1].startswith("['Computer")])
