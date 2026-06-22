# TakeMeter — Project Planning

A text classifier built on `distilbert-base-uncased` to evaluate the quality of discourse in an online community. This document defines the target community, the label taxonomy, the data collection and evaluation strategy, and the plan for using AI tooling throughout the annotation pipeline.

---

## 1. Community

**Community: r/nba**

r/nba is one of the largest and most active sports communities on Reddit (roughly 8–9 million subscribers), and during the regular season, playoffs, and free-agency windows it produces an enormous, continuous stream of comments that span the full spectrum of discourse quality. On any given game thread you will find tribal one-liners ("Refs handed them the game, this league is rigged"), pure emotional venting ("I CANNOT watch this team blow another 20-point lead"), and genuinely rigorous breakdowns of pick-and-roll coverages, lineup plus/minus splits, and shot-quality data pulled from sources like Cleaning the Glass or PBP Stats. Because the same event (a single game, trade, or quote) simultaneously generates reflexive emotional reactions, confident-but-unsupported opinions, and carefully evidenced analysis, the subreddit is an ideal natural laboratory for a discourse-quality classifier: the labels are genuinely contested in the wild rather than artificially separated.

---

## 2. Labels

The taxonomy below is designed to be **mutually exclusive** (every post fits exactly one label) and **exhaustive** (every substantive comment fits somewhere). Classification is based on the *form and evidentiary backing* of the contribution, **not** whether the take is correct or whether you agree with it.

> **Annotation order of operations (to enforce exclusivity):** First ask, *"Is this primarily a feeling/venting with no claim being argued?"* → if yes, `emotional_reaction`. If it makes an argumentative claim, ask *"Does it support that claim with specific evidence, mechanism, or reasoning?"* → if yes, `structured_analysis`; if it asserts the claim without support → `hot_take`.

### Label A — `emotional_reaction`
**Definition:** A post whose primary content is the expression of a feeling, mood, or visceral reaction (excitement, frustration, despair, joy) rather than an argued claim about basketball.

**Examples:**
1. > "I genuinely cannot do this anymore. Every single year we get our hopes up and every single year it ends the exact same way. I'm so tired man."
2. > "LETS GOOOOO!!! I was SCREAMING in my living room, my neighbors probably think someone died lmao. Best game of the season hands down."

### Label B — `hot_take`
**Definition:** A post that makes a confident, debatable claim or judgment about players, teams, or the league but offers little to no supporting evidence, mechanism, or reasoning.

**Examples:**
1. > "Jokić is a top-3 player of all time, it's not even a conversation anymore. Anyone who disagrees just doesn't watch basketball."
2. > "Trading for him was the worst decision this franchise has made in a decade. He's washed and the front office refuses to admit it."

### Label C — `structured_analysis`
**Definition:** A post that advances a claim and backs it with specific evidence, statistics, observed mechanics, or step-by-step reasoning that a reader could evaluate or verify.

**Examples:**
1. > "The reason the offense stalls in the 4th is the lack of a secondary creator. When the primary ball-handler sits, their assist rate drops from ~28% to under 18% and they go to iso-heavy possessions — per CleaningTheGlass their half-court points-per-100 falls off a cliff in those minutes. They need a backup playmaker more than another wing."
2. > "People blaming the defense are missing it. Watch the weak-side rotations — the big is hedging hard on the screen but the low man never tags the roller, so it's a 4-on-3 every time. It's a scheme problem, not effort. Switch the coverage to a drop and those layups disappear."

---

## 3. Hard Edge Cases

**The ambiguous type: the "evidence-flavored hot take."** A post that *sounds* analytical — it name-drops a stat or invokes film — but uses the reference as decoration rather than as support for the actual claim. Example:

> "He's a complete fraud in the playoffs. Dude shot like 38% last series, classic empty stats merchant who disappears when it matters."

This sits on the boundary between `hot_take` (a confident, sweeping judgment) and `structured_analysis` (it cites a shooting percentage).

**Decision rule:** A stat or film reference only promotes a post to `structured_analysis` if the evidence is **specific and load-bearing** — i.e., it is connected by reasoning to the conclusion and is sufficient to evaluate the claim. If the number is vague ("like 38%"), uncontextualized (no sample, no comparison, no causal link), or merely thrown in to dress up a sweeping verdict ("fraud," "empty stats merchant"), label it **`hot_take`**. Concretely: *if removing the stat would not weaken the argument because there is no argument — only an assertion — it is a `hot_take`.* The presence of a number is necessary but not sufficient for `structured_analysis`.

---

## 4. Data Collection Plan

**Source.** Comments will be scraped from r/nba using the Reddit API (PRAW) and the Pushshift/Arctic Shift archive for historical retrieval. To capture the full discourse range, we sample from three thread types:
- **Game Threads & Post Game Threads** — the richest source of `emotional_reaction` and `hot_take`.
- **Daily Discussion / "Hot Take Tuesday" threads** — dense in `hot_take`.
- **Highly-upvoted top-level comments and posts flaired "[Analysis]" / "[Original Content]"** — the primary source of `structured_analysis`, which is rarer in raw game threads.

**Sampling strategy & class balance.** Because raw game-thread comments are heavily skewed toward emotional/low-effort posts, naive random sampling would produce a corpus that is ~70%+ `emotional_reaction` / `hot_take` and starve `structured_analysis`. To prevent any single label exceeding **70%** of the dataset, we use **stratified, source-targeted sampling**:
- Target a roughly even split: **~33% per class** (~33/33/34), with a hard ceiling of 70% on any one label.
- Over-sample analysis-heavy surfaces (Analysis flair, long high-upvote comments > 400 characters) to lift `structured_analysis` toward parity.
- Down-sample the most common short reactions by capping comments under ~15 words.

**Volume & filtering.** Target **600–900 labeled comments** total (≥200 per class). We strip deleted/removed comments, bot messages, pure GIF/image links, single-emoji replies, and duplicates. We deduplicate near-identical copypastas. Each comment is labeled by the taxonomy in §2; a held-out subset is double-annotated to measure inter-annotator agreement (Cohen's κ target ≥ 0.6) before training. Final split: **70% train / 15% validation / 15% test**, stratified by label.

---

## 5. Evaluation Metrics

We report **Overall Accuracy** alongside **per-class Precision, Recall, and F1-score** (plus the macro-averaged F1).

- **Why per-class metrics are essential:** Accuracy alone is misleading under class imbalance. Suppose the test set ends up 70% `emotional_reaction` (because that label dominates raw game threads despite our balancing efforts). A degenerate model that predicts `emotional_reaction` for *every* input would score **70% accuracy** while being completely useless — it would have **0% recall** on `structured_analysis` and `hot_take`, the two classes we most care about identifying. Accuracy hides this failure by rewarding the model for getting the majority class right.
- **Precision** tells us, when the model says a post is `structured_analysis`, how often it's correct (avoiding inflating low-effort posts into "quality discourse").
- **Recall** tells us how many of the true `structured_analysis` posts we actually caught (avoiding missing the rare high-quality content).
- **F1** balances the two, and the **macro-F1** (unweighted mean across classes) gives every class equal say regardless of frequency — directly counteracting the imbalance blind spot that accuracy suffers from.

We keep accuracy in the report as an intuitive headline number, but **macro-F1 is the metric we optimize and gate deployment on.**

---

## 6. Definition of Success

The classifier is considered **deployment-ready** when, on the held-out test set, it achieves:

- **Per-class F1 ≥ 0.70 for *all three* classes** (no class left behind), **and**
- **Macro-averaged F1 ≥ 0.75**, **and**
- **Overall Accuracy ≥ 0.75.**

The per-class floor is the binding constraint: a model that nails `emotional_reaction` but scores F1 = 0.50 on `structured_analysis` fails, because surfacing high-quality discourse is the entire product purpose. Hitting these thresholds means TakeMeter is reliable enough to power a useful feature — e.g., a "Quality Discourse" filter that surfaces analysis and de-emphasizes low-effort reactions — without misclassifying often enough to frustrate users.

---

## 7. AI Tool Plan

We will use a frontier Claude model (e.g., Claude Opus / Sonnet via the Anthropic API) in three distinct phases of the annotation pipeline:

**1. Label stress-testing (before annotation).**
We prompt the model to act as an adversarial critic of the §2 taxonomy: generate borderline comments that *should* be hard to classify, then attempt to assign them and report where the definitions are ambiguous, overlapping, or non-exhaustive. This surfaced the "evidence-flavored hot take" edge case in §3, and we iterate on definitions until the model can apply them consistently. We also have it generate a set of synthetic-but-realistic examples per class to sanity-check that the labels are genuinely distinct.

**2. Annotation assistance (during labeling).**
The model acts as a **first-pass pre-annotator**: each scraped comment is sent with the taxonomy and the decision rules, and it returns a proposed label plus a one-line justification and a confidence score. Human annotators then **review and correct** rather than label from scratch, which speeds throughput and keeps a consistent rubric in mind. Crucially, the model's label is treated as a *suggestion only* — the gold label is the human's — and we track human-vs-model disagreement rate as a signal of which comments are genuinely hard. Low-confidence items are routed to double-annotation.

**3. Systematic failure-pattern analysis (after training).**
We feed the trained DistilBERT model's misclassified test examples (true label, predicted label, text) to the model and ask it to **cluster the errors into named failure modes** — e.g., "sarcastic hot takes misread as emotional," "short analysis with one stat misread as hot_take," "hype with embedded analysis." This turns a flat confusion matrix into actionable categories that tell us whether to fix the label definitions, add targeted training examples, or adjust the decision rules. The output directly feeds the next data-collection iteration.
