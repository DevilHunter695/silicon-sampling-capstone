"""Builds poster_college.html: the college poster template (A4 portrait, 2 pages) filled with our project.
Layout follows the template: mint frame, italic serif title, mint rules, text left + figures right, 3 text columns, footer."""
import json, html
NAVY, GOLD, RED, GREY, GREEN, INK = "#14264f", "#d99a1b", "#c23b32", "#9aa3b8", "#1f8a55", "#1b2235"
F = "font-family:Roboto,Arial,sans-serif"

def svg(w, h, body): return f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" style="display:block;{F}">{body}</svg>'
def t(x, y, s, size=8, anchor="start", fill=INK, weight=400, extra=""): return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-weight="{weight}" {extra}>{html.escape(str(s))}</text>'
def rect(x, y, w, h, fill, rx=2, stroke="none", sw=1): return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'

# ---------- Image 01: study design (475 x 168) ----------
def fig_design():
    cols = [("1  DATA", ["Pew Religion in India", "29,999 adults, 8 items", "fold 0: 23,999 train", "6,000 test respondents", "", "WVS-7 India", "1,692 adults, 15 items", "(zero-shot replication)"]),
            ("2  PROMPT", ["Profile (9 attributes)", "from codebook labels", "+ question text", "+ numbered options", "= prompt P2", "", "No-profile control", "= prompt P0"]),
            ("3  PREDICTORS", ["Zero-shot Qwen2.5-7B", "QLoRA fine-tuned", "Qwen2.5-7B", "Larger zero-shot LLMs", "Majority class", "Logistic regression", "Gradient boosting", ""]),
            ("4  READOUT", ["Probability of each", "answer digit (next token)", "", "Accuracy, NLL, ECE", "Profile variance share", "Directional validity", "Subgroup gaps (Holm)", ""])]
    s = ""; cw = 100; gap = 17
    for i, (h, lines) in enumerate(cols):
        x = 12 + i * (cw + gap)
        s += rect(x, 6, cw, 154, "#f4f8f6", 5, "#b9d9c9", 1) + t(x + 8, 20, h, 8, weight=700, fill=NAVY)
        for j, l in enumerate(lines):
            bold = l in ("Pew Religion in India", "WVS-7 India", "No-profile control", "QLoRA fine-tuned")
            s += t(x + 8, 36 + j * 15, l, 7.6, weight=700 if bold else 400)
        if i < 3: s += f'<path d="M{x+cw+3} 83 L{x+cw+gap-3} 83 M{x+cw+gap-8} 79 L{x+cw+gap-3} 83 L{x+cw+gap-8} 87" stroke="{NAVY}" stroke-width="1.5" fill="none"/>'
    return svg(475, 168, s)

# ---------- Image 02: Pew data (226 x 213) ----------
def fig_pew():
    s = t(113, 16, "Pew Religion in India", 9.5, "middle", NAVY, 700)
    s += rect(14, 28, 198, 24, NAVY, 3) + t(113, 44, "29,999 respondents", 9.5, "middle", "#fff", 700)
    s += rect(14, 62, 150, 22, GREY, 3) + t(89, 77, "23,999 train", 8.5, "middle", "#fff", 700) + rect(168, 62, 44, 22, GOLD, 3) + t(190, 77, "6,000", 8.5, "middle", "#fff", 700) + t(190, 96, "test", 8, "middle")
    s += rect(14, 108, 198, 24, NAVY, 3) + t(113, 124, "73 items screened", 9.5, "middle", "#fff", 700)
    s += rect(14, 142, 98, 22, GREEN, 3) + t(63, 157, "8 trained", 8.5, "middle", "#fff", 700) + rect(116, 142, 96, 22, GREY, 3) + t(164, 157, "8 held out", 8.5, "middle", "#fff", 700)
    s += t(113, 186, "46,209 test answers", 10, "middle", NAVY, 700) + t(113, 200, "(6,000 respondents x 8 items, fold 0)", 7.6, "middle")
    return svg(226, 213, s)

# ---------- Image 03: WVS (226 x 213) ----------
def fig_wvs():
    s = t(113, 16, "WVS-7 India, zero-shot", 9.5, "middle", NAVY, 700) + t(113, 29, "1,692 adults, 373 items screened to 144", 7.6, "middle")
    X = lambda v: 20 + v / 45 * 190
    for v in (0, 10, 20, 30, 40):
        s += f'<line x1="{X(v)}" x2="{X(v)}" y1="44" y2="150" stroke="#dfe5ef"/>' + t(X(v), 162, f"{v}%", 7.5, "middle")
    rows = [("Free LLMs zero-shot", 22, 26, RED), ("Majority class", 24, 26, GREY), ("Logistic / GBM", 34, 38, NAVY)]
    for i, (l, a, b, c) in enumerate(rows):
        y = 56 + i * 32
        s += t(20, y - 4, l, 8, weight=700) + rect(X(a), y, X(b) - X(a), 14, c, 2) + t(X(b) + 5, y + 11, f"{a} to {b}%", 8, weight=700)
    s += t(113, 186, "Gemini 3.1 Flash Lite, gpt-oss-120b,", 7.6, "middle") + t(113, 198, "Qwen2.5-7B (4-bit, Apple M1)", 7.6, "middle")
    return svg(226, 213, s)

# ---------- Image 04: prompt as seen (HTML) ----------
def fig_prompt():
    L = ["<b>You are a survey respondent with the following demographic profile:</b>", "- Sex: Male", "- Age: 18-25", "- Marital status: Unmarried", "- Education level: Senior secondary", "- Income decile: Rs. 10,001 and above", "- Religion: Hindu", "- Urban/Rural: Rural", "- Region: West", "- Caste: General Category",
         "<b>Question:</b> How many children have you had?", "<b>Answer options:</b> 1. No Children", "2. One Child  3. Two Children", "4. Three or More Children", "<b>Respond with exactly one character:</b>", "1/2/3/4. No words, no explanation.", "<b>Your answer:</b>"]
    return '<div class="prompt">' + "<br>".join(L) + "</div>"

# ---------- Image 05: predictors (226 x 213) ----------
def fig_pred():
    items = [("Majority class", "most common answer", GREY), ("Logistic regression", "9 attributes, laptop CPU", GREY), ("Gradient boosting", "9 attributes, laptop CPU", GOLD),
             ("Zero-shot Qwen2.5-7B", "prompted only", RED), ("Fine-tuned Qwen2.5-7B", "QLoRA r=16, 10.1M params", NAVY)]
    s = t(113, 16, "Five predictors, identical rows", 9.5, "middle", NAVY, 700)
    for i, (a, b, c) in enumerate(items):
        y = 30 + i * 34
        s += rect(10, y, 206, 28, "#f4f8f6", 4, "#d3e6dc", 1) + rect(10, y, 7, 28, c, 3) + t(26, y + 12, a, 8.8, weight=700) + t(26, y + 23, b, 7.6, fill="#4a566f")
    s += t(113, 206, "all see the same prompt and the same test rows", 7.5, "middle")
    return svg(226, 213, s)

# ---------- Image 06: accuracy (475 x 168) ----------
def fig_acc():
    R = [("Zero-shot Qwen2.5-7B", 16.1, RED), ("Majority class", 30.4, GREY), ("Fine-tuned Qwen2.5-7B", 38.0, NAVY), ("Logistic regression", 38.5, GREY), ("Gradient boosting", 40.0, GOLD)]
    x0, x1 = 128, 440; X = lambda v: x0 + v / 45 * (x1 - x0); s = ""
    for v in (0, 10, 20, 30, 40):
        s += f'<line x1="{X(v)}" x2="{X(v)}" y1="8" y2="140" stroke="#e3e8f1"/>' + t(X(v), 156, f"{v}%", 7.8, "middle")
    for i, (l, v, c) in enumerate(R):
        y = 12 + i * 26
        s += t(x0 - 8, y + 14, l, 8.6, "end", weight=700 if "Fine" in l else 400) + rect(x0, y, X(v) - x0, 19, c, 2) + t(X(v) + 6, y + 14, f"{v:.1f}%", 8.8, weight=700)
    s += f'<line x1="{X(30.4)}" x2="{X(30.4)}" y1="6" y2="142" stroke="{INK}" stroke-dasharray="3 3"/>'
    return svg(475, 168, s)

# ---------- Image 07: calibration (226 x 213) ----------
def fig_cal():
    B = [[24.9, 25.0, 13068], [34.1, 34.9, 18609], [45.1, 44.6, 7228], [53.5, 56.2, 5307], [64.7, 63.4, 659], [74.2, 68.3, 454], [85.2, 81.6, 38], [96.8, 97.8, 846]]
    L, R_, T, Bt = 38, 212, 28, 172; X = lambda v: L + v / 100 * (R_ - L); Y = lambda v: Bt - v / 100 * (Bt - T)
    s = t(113, 14, "Calibration, fine-tuned model", 9.5, "middle", NAVY, 700)
    for v in (0, 25, 50, 75, 100):
        s += f'<line x1="{L}" x2="{R_}" y1="{Y(v)}" y2="{Y(v)}" stroke="#e3e8f1"/>' + t(L - 5, Y(v) + 3, f"{v}", 7.5, "end") + t(X(v), Bt + 12, f"{v}", 7.5, "middle")
    s += f'<line x1="{X(0)}" y1="{Y(0)}" x2="{X(100)}" y2="{Y(100)}" stroke="{GREY}" stroke-dasharray="4 3"/>'
    s += f'<polyline fill="none" stroke="{NAVY}" stroke-width="1.6" points="{" ".join(f"{X(a)},{Y(b)}" for a, b, _ in B)}"/>'
    for a, b, n in B: s += f'<circle cx="{X(a)}" cy="{Y(b)}" r="{2.5 + (n ** .5) / 22:.1f}" fill="{NAVY}" fill-opacity=".25" stroke="{NAVY}"/>'
    s += t(112, 198, "stated confidence (%)", 7.8, "middle") + t(10, 100, "correct (%)", 7.8, "middle", extra='transform="rotate(-90 10 100)"') + t(113, 209, "ECE 0.009 on 46,209 answers", 7.8, "middle", NAVY, 700)
    return svg(226, 213, s)

# ---------- Image 08: per-question (226 x 213) ----------
def fig_items():
    I = [("Number of children", 56.4, 31.6), ("Reading religious books", 30.8, 21.1), ("Worship attendance", 29.3, 21.5), ("What being [religion] means", 32.3, 25.9), ("Watching religious TV", 31.3, 26.5), ("Abortion", 42.8, 38.6), ("Sex-selective checkups", 45.2, 43.8), ("Politicians in religion", 36.7, 35.4)]
    X = lambda v: 20 + v / 60 * 186; s = t(113, 14, "Accuracy by question", 9.5, "middle", NAVY, 700)
    s += f'<circle cx="62" cy="26" r="3.5" fill="{NAVY}"/>' + t(68, 29, "fine-tuned", 7.6) + f'<circle cx="124" cy="26" r="3.5" fill="{GREY}"/>' + t(130, 29, "majority", 7.6)
    for i, (l, a, m) in enumerate(I):
        y = 52 + i * 19
        s += t(8, y - 6, l.replace("[religion]", "religion"), 7.4, fill="#3a4560") + f'<line x1="{X(m)}" x2="{X(a)}" y1="{y}" y2="{y}" stroke="#b9c5dc" stroke-width="3"/><circle cx="{X(m)}" cy="{y}" r="3.5" fill="{GREY}"/><circle cx="{X(a)}" cy="{y}" r="4.2" fill="{NAVY}"/>' + t(X(max(a, m)) + 8, y + 3, f"{a:.1f}%", 7.6, weight=700)
    s += t(113, 207, "lift is largest where the profile decides the answer", 7.5, "middle")
    return svg(226, 213, s)

# ---------- Image 09: profile use (226 x 213) ----------
def fig_profile():
    s = t(113, 14, "How each method uses the profile", 9.5, "middle", NAVY, 700) + t(10, 34, "Profile share of prediction variance", 8, weight=700)
    X = lambda v: 10 + v / 35 * 150
    s += rect(10, 40, X(29) - 10, 15, RED, 2) + t(X(29) + 5, 52, "29%  zero-shot model", 8) + rect(10, 60, X(4) - 10, 15, GREEN, 2) + t(X(4) + 5, 72, "4%  real answers", 8)
    s += t(10, 104, "Group-difference correlation with truth", 8, weight=700)
    Xr = lambda v: 10 + v / .8 * 150
    for i, (l, v, c) in enumerate([("zero-shot", .22, RED), ("fine-tuned", .58, NAVY), ("grad. boosting", .64, GOLD)]):
        y = 112 + i * 22
        s += rect(10, y, Xr(v) - 10, 15, c, 2) + t(Xr(v) + 5, y + 12, f"{l}  r = {v:.2f}", 8)
    s += t(113, 191, "200-respondent subset; removing the profile", 7.4, "middle") + t(113, 202, "raises zero-shot accuracy by 3.6 points", 7.4, "middle")
    return svg(226, 213, s)

# ---------- Image 10: transfer and scale (226 x 213) ----------
def fig_transfer():
    s = t(113, 14, "Transfer and scale", 9.5, "middle", NAVY, 700) + t(10, 34, "8 unseen Pew items", 8, weight=700)
    X = lambda v: 10 + v / 35 * 150
    s += rect(10, 40, X(28.9) - 10, 15, NAVY, 2) + t(X(28.9) + 5, 52, "28.9%  adapter", 8) + rect(10, 60, X(28.5) - 10, 15, GREY, 2) + t(X(28.5) + 5, 72, "28.5%  base model", 8) + t(10, 88, "p = 0.18: no detectable transfer", 7.6, fill="#4a566f")
    s += t(10, 112, "Larger zero-shot LLM", 8, weight=700) + rect(10, 118, X(27.6) - 10, 15, RED, 2) + t(X(27.6) + 5, 130, "27.6%  gpt-oss-120b", 8) + t(10, 144, "100 respondents; majority is 30.2%", 7.6, fill="#4a566f")
    s += t(10, 168, "Labelled respondents needed to beat", 8, weight=700) + t(10, 180, "the majority guess", 8, weight=700)
    s += rect(10, 186, 62, 16, GREY, 2) + t(41, 198, "logistic ~400", 7.8, "middle", "#fff", 700) + rect(78, 186, 100, 16, GOLD, 2) + t(128, 198, "gradient boosting ~1,600", 7.8, "middle", "#fff", 700)
    return svg(226, 213, s)

def img(n, cap, body, cls="sm"): return f'<figure class="{cls}"><div class="box">{body}</div><figcaption>Image {n:02d} : {cap}</figcaption></figure>'

CSS = """
@page{size:794px 1123px;margin:0}*{box-sizing:border-box}html,body{margin:0;padding:0}
body{font-family:Roboto,Arial,sans-serif;color:#1b2235;-webkit-print-color-adjust:exact;print-color-adjust:exact}
.page{width:794px;height:1123px;position:relative;overflow:hidden;border:7px solid #cdeadb;background:#fff;page-break-after:always}
.page:last-child{page-break-after:auto}
.title{position:absolute;top:0;left:0;right:0;height:52px;text-align:center;font:italic 700 30px/50px "Playfair Display",Georgia,serif;letter-spacing:.2px;color:#0e0e12}
.rule{position:absolute;left:26px;right:26px;height:3px;background:#cdeadb}
.t{position:absolute;font-size:7.9pt;line-height:1.34;overflow:hidden;text-align:left;color:#1b2235}
.t h4{margin:0 0 3px;font-size:7.6pt;letter-spacing:.07em;text-transform:uppercase;color:#14264f;font-weight:700}
.t h4:not(:first-child){margin-top:7px}
.t p{margin:0 0 4px}.t ul{margin:0 0 4px;padding-left:11px}.t li{margin:0 0 2px}
.t b{font-weight:700}
.t .link{font-style:italic;position:absolute;bottom:0;left:0;right:0;font-size:7pt;line-height:1.3}
figure{position:absolute;margin:0}
figure .box{border:1px solid #111;background:#fff;overflow:hidden}
figure.lg .box{width:475px;height:168px}figure.sm .box{width:226px;height:213px}
figcaption{font-size:6.6pt;line-height:1.25;margin-top:4px;color:#2a3146}
figure.sm{width:226px}figure.lg{width:475px}
.prompt{font:5.7pt/1.4 "Roboto Mono",Menlo,monospace;padding:7px 8px;color:#1b2235;height:213px;overflow:hidden;background:#fbfdfc}
.foot{position:absolute;left:41px;right:37px;top:1050px;height:58px;font-size:7.1pt;line-height:1.55;letter-spacing:.02em;text-transform:uppercase;color:#10131c}
.foot .l{position:absolute;left:0;top:6px}
.foot .r{position:absolute;right:0;top:0;display:flex;align-items:center;gap:10px}
.foot .r img{height:52px}
.foot .r .sch{font:700 8.3pt/1.18 "Roboto Condensed",Roboto,Arial,sans-serif;text-transform:uppercase;width:208px;letter-spacing:.3px}
"""

def foot(): return ('<div class="foot"><div class="l">Group 55 | Bhanuj Bhalla 70022300017 &middot; Ishaan Bansal 70022300011<br>Yash Bhende 70022300137 &middot; Krishank Ayyella 70022300136<br>Mentor: Ms. Khushbu Chauhan</div>'
 '<div class="r"><img src="logo_dark.png" alt="NMIMS"><div class="sch">Mukesh Patel School of Technology Management &amp; Engineering</div></div></div>')
def T(top, left, w, h, body): return f'<div class="t" style="top:{top}px;left:{left}px;width:{w}px;height:{h}px">{body}</div>'
def at(el, top, left): return el.replace("<figure", f'<figure style="top:{top}px;left:{left}px"', 1)
def page(body, title=True): return f'<div class="page"><div class="title">Silicon Sampling for Indian Public Opinion</div><div class="rule" style="top:52px"></div>{body}<div class="rule" style="top:522px"></div><div class="rule" style="top:784px"></div>{foot()}</div>'

LX, RX, W = 41, 282, 224
P1 = (
 T(70, LX, W, 196, '<h4>Abstract / Introduction / Motivation</h4><p><b>Silicon sampling</b> gives a language model a respondent\'s demographic profile and treats its answer as that person\'s survey answer. If it were faithful, surveys would become far cheaper.</p><p>We test this in India on <b>Pew Religion in India</b> (29,999 adults) and <b>WVS-7 India</b> (1,692 adults). Every method is scored on the same respondent-question rows as matched classifiers, and we check <b>direction</b>, <b>calibration</b> and <b>subgroup gaps</b>, not only accuracy.</p>')
 + at(img(1, "Study design: one prompt, identical rows for every predictor", fig_design(), "lg"), 76, RX)
 + T(274, LX, W, 246, '<h4>Literature Review / Background</h4><ul><li>US attitudes are reproduced (Argyle 2023) but many groups misrepresented (Santurkar 2023).</li><li>Individually, LLMs match or trail baselines and inflate group gaps (Chen 2026; Miranda 2025).</li><li>Personas explain under 10% of variance (Hu 2024); fine-tuning can generalise (Suh 2025).</li></ul><p><b>Gap:</b> India, the profile\'s own contribution, transfer to unseen questions.</p><h4>Objectives</h4><p><b>RQ1</b> How does a prompted LLM use the profile? <b>RQ2</b> What does fine-tuning change versus classifiers? <b>RQ3</b> Does it transfer to unseen questions? <b>RQ4</b> Do size or more data matter?</p>')
 + at(img(2, "Pew Religion in India: respondents and items", fig_pew()), 274, RX) + at(img(3, "WVS-7 India: first zero-shot replication", fig_wvs()), 274, RX + 249)
 + T(538, LX, W, 240, '<h4>Methods</h4><p><b>Data.</b> Pew fold 0: 23,999 training and 6,000 test respondents, 8 items, 46,209 test answers; items 9 to 16 held out for transfer.</p><p><b>Prompt.</b> Profile, question and numbered options in one fixed template; a no-profile control measures what the profile adds.</p><p><b>Predictors.</b> Zero-shot Qwen2.5-7B, the same model fine-tuned with QLoRA, and majority, logistic and gradient-boosting classifiers on the same rows.</p><p><b>Readout.</b> The probability of each answer digit at the next token.</p>')
 + at(img(4, "What the model sees: a real prompt", fig_prompt()), 538, RX) + at(img(5, "Five predictors on the same rows", fig_pred()), 538, RX + 249)
 + T(800, 41, 227, 236, '<h4>Methods (continued)</h4><p><b>QLoRA fine-tuning.</b> 4-bit NF4 base, LoRA rank 16 and alpha 32 on the attention projections, 10.1M trainable parameters, 184,853 examples, 5,777 steps, 6.6 hours on a university lab GPU.</p><p><b>Scoring.</b> Kaggle 2 x T4 in fp16, about 0.2 s per answer and 75 minutes per condition. Classifiers train in under a minute on a laptop.</p><p><b>Reproducibility.</b> Code, folds and prompts are in the project repository, and every figure on this poster is computed from stored model outputs.</p>')
 + T(800, 285, 227, 236, '<h4>Diagnostics</h4><ul><li>Accuracy and negative log-likelihood</li><li>Expected calibration error</li><li>Profile share of prediction variance</li><li>Directional validity: correlation and slope of group differences</li><li>Jensen-Shannon divergence</li><li>Subgroup gaps, Holm-corrected permutation tests</li><li>Respondent-clustered bootstrap intervals</li><li>Per-question and per-subgroup breakdowns</li></ul>')
 + T(800, 529, 227, 236, '<h4>Lessons from the build</h4><ul><li>WVS region labels were wrong on every code until checked against the official annex; fixing it moved subgroup gaps by several points.</li><li>The first prompt wording caused 13% refusals; the final wording gave 0%.</li><li>An apparent collapse on one item was a decoding artefact, fixed by the next-digit readout.</li><li>All reported numbers come from one training run; more seeds are future work.</li></ul>')
)
P2 = (
 T(70, LX, W, 196, '<h4>Results &amp; Discussion</h4><p>On Pew fold 0 (46,209 answers) the <b>fine-tuned model reaches 38.0%</b>, against 30.4% for always guessing the majority answer, 38.5% for logistic regression and <b>40.0% for gradient boosting</b>. Zero-shot Qwen2.5-7B scores <b>16.1%</b>.</p><p>Fine-tuning closes <b>79%</b> of the gap from the majority guess to gradient boosting, but stays 2.1 points below it.</p><p>Per question, the fine-tuned model beats the majority guess on all 8 items, by 1.3 to 24.8 points, and trails gradient boosting on every one.</p>')
 + at(img(6, "Accuracy on 46,209 test answers; dashed line is the majority guess", fig_acc(), "lg"), 76, RX)
 + T(274, LX, W, 242, '<h4>Calibration and confidence</h4><p>Calibration error is <b>0.009</b>: when the model says 60%, it is right about 60% of the time.</p><p>Above 70% confidence it is right <b>87.3%</b> of the time, on 1,338 answers, nearly all from the number-of-children question. The driver is marital status: unmarried respondents are 98.8% confident and 96.6% accurate.</p><p>A classifier finds the same rule, so this shows a learned pattern, not superiority. The model is strongest where the profile decides the answer and weakest on opinions.</p>')
 + at(img(7, "Stated confidence against accuracy", fig_cal()), 274, RX) + at(img(8, "Accuracy by question against majority", fig_items()), 274, RX + 249)
 + T(538, LX, W, 240, '<h4>Profile use and transfer</h4><p><b>Zero-shot, the profile explains 29%</b> of the model\'s prediction variance against 4% for real answers, yet removing it raises accuracy by 3.6 points: the model reacts strongly and in the wrong direction.</p><p>Group differences correlate 0.22 (zero-shot), 0.58 (fine-tuned) and 0.64 (gradient boosting) with the truth.</p><p>On 8 unseen items the adapter scores 28.9% against 28.5% for the base model: <b>no detectable transfer</b>.</p>')
 + at(img(9, "Profile share of variance, direction", fig_profile()), 538, RX) + at(img(10, "Transfer, a larger LLM, data needs", fig_transfer()), 538, RX + 249)
 + T(800, 41, 227, 236, '<h4>Discussion</h4><p>Prompting alone does not make a faithful Indian respondent. Fine-tuning gives a calibrated simulator close to, but not above, matched classifiers, at far higher cost.</p><p><b>Limits.</b> One fold, one seed, one epoch; a small transfer test; survey weights not applied.</p><p><b>Next.</b> All five folds and seeds, more held-out items, larger models on the same rows.</p>')
 + T(800, 285, 227, 236, '<h4>References</h4><ul style="padding-left:9px;font-size:7pt;line-height:1.3"><li>Argyle et al., Political Analysis 31(3), 2023.</li><li>Santurkar et al., ICML 2023.</li><li>Chen, Zhu, Zheng, arXiv:2607.26348, 2026.</li><li>Miranda &amp; Balbi, Entropy 27(9), 2025.</li><li>Hu &amp; Collier, ACL 2024.</li><li>Suh et al., ACL 2025.</li><li>Dettmers et al., QLoRA, NeurIPS 2023.</li><li>Pew Research Center, Religion in India: Tolerance and Segregation, 2021.</li><li>World Values Survey, Wave 7, India.</li></ul>')
 + T(800, 529, 227, 236, '<h4>Acknowledgements</h4><p>Ms. Khushbu Chauhan, project mentor; the university AI Lab for GPU access; Kaggle for free T4 compute.</p><p>Accepted at Hinweis AIDE-2026 (paper ID 591).</p><p>Capstone project, Group 55, MPSTME, NMIMS Mumbai, A.Y. 2026-2027.</p><div class="link">project hyperlink : https://<wbr>github.com/<wbr>DevilHunter695/<wbr>silicon-sampling-capstone</div>')
)
doc = f'<!doctype html><html><head><meta charset="utf-8"><title>55</title><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@1,700&family=Roboto:wght@400;500;700&family=Roboto+Condensed:wght@700&family=Roboto+Mono:wght@400&display=swap"><style>{CSS}</style></head><body>{page(P1)}{page(P2)}</body></html>'
open(__file__.replace("build.py", "poster_college.html"), "w").write(doc)
print("ok", len(doc))
