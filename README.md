# TakeMeter — Discourse-Quality Classifier for r/nba

A fine-tuned `distilbert-base-uncased` text classifier that sorts r/nba comments into three discourse-quality categories, compared against a zero-shot `llama-3.3-70b-versatile` baseline.

> Full design rationale lives in [`planning.md`](./planning.md). This README is the standalone final report.

**🎥 Demo video (3–5 min):** https://drive.google.com/file/d/1mV_N9bRY2QF1i11nf9hZ6GNCPCcMFsoY/view?usp=sharing

---

## 1. Community

**r/nba.** It's one of the largest, most active sports communities on Reddit, and the same event (a game, a trade, a highlight) simultaneously produces reflexive emotional venting, confident-but-unsupported opinions, and genuinely evidenced breakdowns. That natural range — the labels are contested in the wild rather than artificially separated — makes it an ideal corpus for a discourse-quality classifier.

## 2. Label Taxonomy

Classification is based on the **form and evidentiary backing** of a comment, not whether the take is correct or popular.

| Label | One-sentence definition |
|---|---|
| `emotional_reaction` | A comment whose primary content is the expression of a feeling or visceral reaction rather than an argued claim. |
| `hot_take` | A confident, debatable claim about players/teams/the league offered with little to no supporting evidence. |
| `structured_analysis` | A claim backed by specific evidence, statistics, observed mechanics, or step-by-step reasoning a reader could evaluate. |

**Decision order (enforces mutual exclusivity):** Is it primarily a feeling with no argued claim? → `emotional_reaction`. Otherwise, does it support its claim with specific, load-bearing evidence? → yes: `structured_analysis`; no: `hot_take`.

**Examples**

- `emotional_reaction` — *"I genuinely cannot do this anymore. Every single year same heartbreak man."* · *"LETS GOOOOO!!! I was screaming so loud my dog ran out of the room."*
- `hot_take` — *"The most athletic player ever."* · *"He's the greatest pure athlete this sport has ever seen."*
- `structured_analysis` — *"In 2009 he was listed at 6'9 250. That's the average weight of an NFL linebacker. Was still the fastest, highest flying player on the floor and did it for 39 minutes a night."* · *"Lebron has the best career transition volume all time. His fastbreak is about 1.25 points per possession."*

## 3. Dataset

- **Source:** Comments collected from a single high-traffic r/nba thread (a LeBron James career-athleticism highlight discussion), copied from the public web. Stored in [`dataset.csv`](./dataset.csv) with columns `text`, `label`, `notes`.
- **Size:** 286 labeled comments.
- **Labeling process:** Each comment was read individually and labeled by hand against the planning.md definitions, using an LLM as a first-pass pre-labeler that was then reviewed and corrected (see AI Usage). The `notes` column records the reasoning for every comment that sat on a label boundary.

**Label distribution**

| Label | Count | Share |
|---|---|---|
| `emotional_reaction` | 196 | 68.5% |
| `hot_take` | 55 | 19.2% |
| `structured_analysis` | 35 | 12.2% |
| **Total** | **286** | **100%** |

No single label exceeds the 70% ceiling. The set is imbalanced toward `emotional_reaction` because the source thread was a celebratory highlight post — a real and intentional finding discussed in §7.

**Three genuinely difficult examples**

1. *"He shot like 38 percent last series so he's a complete fraud who disappears when it matters, classic empty stats merchant."* — Cites a stat, so it *looks* analytical. **Decided `hot_take`:** the number is vague and decorative, not load-bearing; removing it wouldn't weaken any argument because there is none.
2. *"For sure. He is a perfect combination of size, speed, explosiveness, and basketball IQ! The only thing I can think of was that he was not a great shooter early on. He only developed his 3 late in his career."* — Opens with unsupported praise but pivots to a specific, checkable developmental claim. **Decided `structured_analysis`:** the load-bearing content is the concrete observation.
3. *"He averaged 30.3 at 37 years old. Prime LeBron is probably averaging 34-35 in this era."* — The conclusion is a speculative projection. **Decided `structured_analysis`:** the projection is explicitly anchored to and reasoned from a real, specific stat (30.3 PPG at age 37).

## 4. Fine-Tuning Approach

- **Base model:** `distilbert-base-uncased` (HuggingFace) with a 3-class sequence-classification head.
- **Split:** 70% train / 15% validation / 15% test, stratified by label (handled by the starter notebook).
- **Training setup (final):** 12 epochs, learning rate `3e-5`, batch size 8, weight decay `0.01`, 20 warmup steps, **inverse-frequency class weights** in the loss, best model selected on validation **macro-F1**.
- **Key hyperparameter decision — class weighting.** The default run (3 epochs, LR 2e-5, batch 16, unweighted loss) **collapsed to the majority class**: validation accuracy was frozen at exactly 0.674 (= the `emotional_reaction` share of the val set) across all epochs, and on the test set the model predicted `emotional_reaction` for all 43 comments — F1 = 0.00 on both minority classes. Increasing epochs (12) and learning rate (3e-5) *alone* did not fix it; the model still collapsed. The fix was adding **inverse-frequency class weights** to the cross-entropy loss (`emotional_reaction` 0.49, `hot_take` 1.73, `structured_analysis` 2.72), which penalizes majority-class bias. This broke the collapse — the model began predicting all three classes and macro-F1 jumped from **0.27 → 0.67**. Best-model selection was also switched from `accuracy` to `f1_macro`, since accuracy *rewards* majority collapse under imbalance.

## 5. Baseline Approach

A zero-shot baseline using Groq's `llama-3.3-70b-versatile` at `temperature=0`. The prompt names the community and task, gives the three label definitions and one example each (lifted from planning.md), and instructs the model to **output only the label name**. The notebook classifies every test comment, matches the response to a known label (longest-label-first), and flags unparseable outputs. Prompt used:

```
You are classifying comments from the r/nba subreddit by discourse quality.
Assign each comment to exactly one of the following categories.

emotional_reaction: the comment is primarily a feeling or visceral reaction (excitement, frustration, awe, nostalgia) with no argued claim about basketball.
Example: "LETS GOOOOO!!! I was screaming so loud my dog ran out of the room"

hot_take: a confident, debatable claim about players, teams, or the league, offered with little or no supporting evidence.
Example: "The most athletic player ever"

structured_analysis: a claim backed by specific evidence, statistics, observed mechanics, or step-by-step reasoning a reader could evaluate.
Example: "Lebron has the best career transition volume all time. His fastbreak is about 1.25 points per possession."

Respond with ONLY the label name, lowercase, exactly as written.
Do not explain your reasoning.

Valid labels:
emotional_reaction
hot_take
structured_analysis
```

## 6. Evaluation Report

> ⚠️ FILL FROM YOUR COLAB RUN. Replace every `<<…>>` below with the numbers your notebook prints (Sections 4, 5, 6) and the committed `confusion_matrix.png`.

### Overall accuracy

| Model | Accuracy |
|---|---|
| Zero-shot baseline (Groq) | **0.791** (43/43 parseable) |
| Fine-tuned DistilBERT | **0.744** |

**Headline:** the zero-shot baseline edges out the fine-tuned model on overall accuracy (0.791 vs 0.744) and macro-F1 (0.70 vs 0.67). Fine-tuning was *not* a net win here — but the per-class breakdown below shows it reshaped *where* the errors live (much better `structured_analysis`, much worse `hot_take`).

### Per-class metrics (fine-tuned model)

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `emotional_reaction` | 0.81 | 0.83 | 0.82 | 30 |
| `hot_take` | 0.33 | 0.25 | 0.29 | 8 |
| `structured_analysis` | 0.83 | 1.00 | 0.91 | 5 |
| **macro avg** | 0.66 | 0.69 | 0.67 | 43 |

### Per-class metrics (baseline)

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `emotional_reaction` | 1.00 | 0.80 | 0.89 | 30 |
| `hot_take` | 0.47 | 1.00 | 0.64 | 8 |
| `structured_analysis` | 1.00 | 0.40 | 0.57 | 5 |
| **macro avg** | 0.82 | 0.73 | 0.70 | 43 |

**Baseline read:** The zero-shot LLM is precise but timid on the two hard classes. It *over-predicts* `hot_take` (recall 1.00 but precision 0.47 — it dumps borderline comments here), and it *under-catches* `structured_analysis` (recall 0.40 — it misses 3 of 5, likely because short analytical comments look like hot takes). This is the gap fine-tuning needs to close.

### Confusion matrix (fine-tuned, test set) — rows = true, cols = predicted

|  | pred: emotional | pred: hot_take | pred: structured |
|---|---|---|---|
| **true: emotional** | 25 | 4 | 1 |
| **true: hot_take** | 6 | 2 | 0 |
| **true: structured** | 0 | 0 | 5 |

(Image copy committed as `confusion_matrix.png`.)

**The dominant error is directional: `hot_take` → `emotional_reaction`.** 6 of 8 true hot takes were misclassified, all but none of them landing in `emotional_reaction`. `structured_analysis` is perfectly recalled (5/5), and `emotional_reaction` is mostly clean. So the one boundary the model never learned is **hot_take vs. emotional_reaction** — the two classes that share the most surface language (short, enthusiastic, superlative-laden) and differ only in whether a *claim* is being asserted.

### Three wrong predictions, analyzed

1. **Text:** *"The best to ever do it"* — **True:** `hot_take` **Pred:** `emotional_reaction` (conf 0.55).
   This is a confident, debatable claim (a ranking assertion) with zero evidence — textbook `hot_take` by our rule. The model misread it as `emotional_reaction` because the *surface form* is a short, punchy hype phrase indistinguishable from a celebratory reaction. The model learned to associate brevity + enthusiasm with `emotional_reaction`, and a 5-word hot take carries no lexical signal of an argued claim. This is a **boundary the model genuinely hasn't learned**, not a labeling slip.

2. **Text:** *"In the replays he clearly got fouled but IDC you have to be extremely stupid to try that shit and deserve to get hacked w no call"* — **True:** `hot_take` **Pred:** `emotional_reaction` (conf 0.68).
   The underlying content is an opinionated judgment ("you have to be stupid to try that"), which is a hot take, but it's wrapped in profanity and venting affect ("IDC", "that shit"). The model latched onto the emotional register and ignored the claim underneath. This is a **data-distribution problem, not annotation inconsistency**: I labeled it consistently with the definition, but the training set has too few hot takes phrased in emotional language for the model to learn that affect ≠ absence of a claim.

3. **Text:** *"Bigger, stronger, faster, jumps higher than everyone else on the floor."* — **True:** `hot_take` **Pred:** `emotional_reaction` (conf 0.69).
   A confident, unsupported superlative claim — hot take. But phrased as a string of admiring descriptors, it reads like hype/awe to the model. Same root cause as #1 and #2: `hot_take` is both the **least-represented distinct class** (55 examples, ~8 in test) and the one that **overlaps most lexically** with `emotional_reaction`. The fix is more `hot_take` training examples — specifically short, superlative, emotionally-phrased ones — so the model can learn that an asserted claim, however brief or hyped, is not the same as a pure feeling.

### Sample classifications

All five examples below were predicted **correctly** by the fine-tuned model.

| Comment | Predicted | Confidence |
|---|---|---|
| "Lebron has the best career transition volume all time. His fastbreak is about 1.25 points per possession." | `structured_analysis` | 0.88 |
| "He averaged 30.3 at 37 years old. Prime LeBron is probably averaging 34-35 in this era." | `structured_analysis` | 0.89 |
| "The most athletic player ever" | `hot_take` | 0.60 |
| "LETS GOOOOO!!! I was screaming so loud my dog ran out of the room" | `emotional_reaction` | 0.81 |
| "I miss headband lebron." | `emotional_reaction` | 0.82 |

*Why a correct prediction is reasonable:* For *"Lebron has the best career transition volume all time. His fastbreak is about 1.25 points per possession,"* the model predicted `structured_analysis` at 0.88 — correctly, because the comment backs its claim with a specific, checkable statistic (1.25 PPP), which is exactly the load-bearing evidence that defines the class. Note also the confidence gradient that matches difficulty: the two evidence-backed comments score ~0.88, the pure reactions ~0.81, while the bare `hot_take` ("the most athletic player ever") is the model's least-confident correct call at 0.60 — consistent with `hot_take` being the boundary it learned least well.

## 7. Reflection — what the model learned vs. what I intended

I intended the model to learn the **evidentiary structure** of a comment — does it argue from specific checkable evidence (`structured_analysis`), merely assert (`hot_take`), or just emote (`emotional_reaction`)? The confusion matrix shows it learned **two of the three boundaries and one strong lexical shortcut.**

What it captured well: **`structured_analysis` (5/5 recall).** Despite being the rarest class, comments with explicit evidence — stats, player comparisons, mechanism ("his fastbreak is about 1.25 points per possession") — carry an unmistakable lexical signature (numbers, named players, causal connectives), and the class weighting gave the model enough incentive to key on it. This is genuinely the "evidence detector" I wanted.

What it missed: **the `hot_take` ↔ `emotional_reaction` boundary.** This is precisely the distinction I cared most about — *a confident claim vs. a pure feeling* — and it's the one the model never learned (hot_take recall 0.25). The reason is instructive: that boundary is **semantic, not lexical.** "The best to ever do it" (hot take) and "Best night of my life" (reaction) look nearly identical at the token level; telling them apart requires recognizing that one *asserts a debatable claim about basketball* and the other *expresses a feeling*. With only ~38 training hot takes, many lexically overlapping with hype, the model fell back on the surface shortcut "short + enthusiastic → emotional_reaction." 

In short: the model became a decent **evidence detector** but not a **claim detector** — it captured the part of my taxonomy that has a lexical fingerprint and missed the part that requires genuine semantic judgment. That gap, plus the fact that a well-prompted zero-shot LLM beat it overall, is the honest takeaway: 286 examples (heavily skewed) is enough to teach the easy, lexically-marked distinction but not the subtle one.

## 8. Spec Reflection

- **One way the spec helped:** Forcing the decision-order rule and explicit edge-case handling in planning.md *before* annotating meant the 286 comments were labeled against a consistent boundary, so the `notes` column doubles as a documented rationale rather than after-the-fact justification.
- **One way the implementation diverged:** The plan targeted a roughly even ~33% per-class split, but a single real thread yielded only 12% `structured_analysis`. Rather than synthesize fake balance, I kept the authentic distribution (still under the 70% cap) and treat the imbalance as a documented limitation and a driver of the failure analysis. *(Revise if you collect more data.)*

## 9. AI Usage

1. **Label stress-testing & taxonomy design.** I directed Claude to generate boundary comments between the three labels and pressure-test the definitions; it surfaced the "evidence-flavored hot take" failure mode, which became the explicit load-bearing-evidence decision rule in planning.md §3. I kept the rule and rewrote the wording in my own terms.
2. **Annotation assistance.** I used Claude to pre-label the raw thread comments against my definitions; I then reviewed and corrected every row and wrote the `notes` reasoning for each edge case myself. *(Disclosure: every label in `dataset.csv` was human-reviewed; the LLM provided a first pass only.)*
3. **Failure-pattern analysis.** *(After your run)* I will paste the misclassified test examples to Claude to cluster them into named failure modes, then re-read each example to verify before writing §6.

## 10. Files

- `planning.md` — full design spec
- `dataset.csv` — 286 labeled comments
- `confusion_matrix.png` — committed from Colab
- `evaluation_results.json` — committed from Colab
