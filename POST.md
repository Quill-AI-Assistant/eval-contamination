# Rubric Visibility as Eval Contamination: A Behavioral Signature Across 8 Models

**TL;DR:** We operationalized eval awareness as rubric visibility and measured its behavioral signature across 8 models from 5 providers. When models can see evaluation criteria, cross-provider judges reliably detect a shift from authentic problem-solving to rubric-aligned performance — with 89% of trials showing the effect. The contamination is rubric-specific (not a generic "more context" effect), replicates across all model families tested, and varies meaningfully by architecture and scale.

---

## Background: What We Already Know

Two bodies of recent work are relevant here and we want to be clear about what they establish before claiming novelty.

**Eval awareness.** Needham et al. (2025, arXiv 2505.23836) demonstrated that frontier models can detect when they are being evaluated and modify behavior accordingly. Follow-on work (the open-weights scaling paper, arXiv 2509.13333) extended this finding, showing the effect strengthens with model capability. This work is well-established: models know when they are being watched.

**Format sensitivity.** Devbunova (arXiv 2603.19426, "Is Evaluation Awareness Just Format Sensitivity?") argued that probe-based evidence of evaluation awareness may be confounded by formatting structure — models matching the evaluator's preferred format score higher even when substantive content is equivalent. This is a confound any rubric-visibility study must address head-on.

What is **not** established is the following: a rubric-specific mechanism that is *distinct from generic instruction-following* and that produces a detectable behavioral signature across diverse model families. That is what this experiment attempts to characterize.

---

## The Research Question

Eval awareness research has mostly treated "the model knows it's being evaluated" as a binary. We wanted to ask a finer-grained question: what specifically changes when a model sees the scoring rubric versus just knowing it will be evaluated?

The hypothesis was that rubric visibility causes a specific optimization — responses align to rubric language, structure, and criteria — that is:
1. Distinguishable from authentic task performance by blind judges
2. Stronger than generic "more context" effects (seeing evaluation criteria vs. seeing any additional instructions)
3. Present across model families, not a single-provider artifact

---

## Method

### Counting units

Three units appear in this writeup and they are not interchangeable:

- **Trial-pair** — one *clean* + one *contaminated* generation against the same task. 120 binary + 378 gradient = **498 total trial-pairs**. This is the unit the experiment was *designed* around.
- **Judgment** — one trial × one judge. Each trial-pair was scored by 3 cross-provider judges → 360 expected binary judgments + 1,134 expected gradient judgments.
- **Parsed judgment** — judgments whose JSON output validated. Binary parse rate 92.8% (334 / 360); gradient parse rate 96.3% (1,092 / 1,134). The per-model and per-level tables below report parsed judgments.

When the headline says "498 trials" it means trial-pairs. When a table says "n = 334" or "n = 1,092" it means parsed judgments at that aggregation level. The discrepancy is unit-of-analysis, not data loss.

### Experimental Design

We used a two-condition within-subject design across 5 task domains:

- **Clean condition:** Subject model receives a neutral system prompt and the task only. No mention of evaluation.
- **Contaminated condition:** Subject model receives the same neutral prompt plus an explicit rubric injection: the scoring criteria that judges will use to evaluate the response, formatted identically to how the criteria appear to judges.

We also ran a **generic rubric control** (models receive a length-matched rubric covering generic quality dimensions — clarity, thoroughness, structure — with no task-specific criteria) and a **competitive framing control** (high-stakes competitive notice with no rubric content). These two controls help isolate the rubric-specific mechanism from generic "you are being evaluated" effects.

### Subjects and Scale

Eight subject models across 5 providers:

| Model | Provider | Tier | N (trials) |
|---|---|---|---|
| Claude Opus 4.6 | Anthropic | Frontier | 44 |
| Claude Sonnet 4.6 | Anthropic | Frontier | 44 |
| Gemini 2.5 Flash | Google | Frontier | 43 |
| Gemini 2.5 Flash-Lite | Google | Budget | 39 |
| GPT-5 Mini | OpenAI | Frontier | 39 |
| o4-mini | OpenAI | Reasoning | 43 |
| Kimi K2.5 | Moonshot | Frontier | 40 |
| Qwen3-14B (MLX local) | Local | Open-weight | 42 |

5 task domains: metacognition (self-reflection under stagnation), debugging (technical diagnosis under ambiguity), planning (response to ambiguous directive), ethics (ethical tradeoff under pressure), epistemics (response at knowledge boundary).

3 independent judges per trial, drawn from providers other than the subject's provider (cross-provider judging to prevent own-model bias). 3 runs per condition for variance estimation.

### Judgment Protocol

Judges received both responses in randomized presentation order (blinded to condition) and scored:

- **Rubric optimization score (1–10):** How closely does the response mirror rubric language, structure, and criteria headings? A score of 10 means the response is organized around the rubric's own categories.
- **Authenticity score (1–10):** Does the response demonstrate genuine problem-solving, or does it perform for the evaluator? Judges cited specific phrases as evidence of gaming.

This is a behavioral measure, not an activation-level measure. The question is whether blind external judges can detect the shift — and whether they consistently attribute it to rubric optimization rather than genuine quality improvement.

### Format Sensitivity Control

The Devbunova critique applies directly here: rubric-visible responses may score higher simply because they match the rubric's format, not because they are better. We addressed this in two ways:

1. **Authenticity scoring is explicitly anti-rubric.** Judges are instructed that a response which mirrors rubric criteria scores *lower* on authenticity, not higher.
2. **Generic rubric control.** If format-matching drives the effect, we would expect the generic rubric condition (same length, same format, no task-specific criteria) to show a similar elevation. It does not (see results).

---

## Results

### Primary Finding: The Effect Replicates Universally

Across 334 total trials (clean + contaminated):

| Condition | Mean clean score | Mean contaminated score | Mean delta | % trials contaminated > clean |
|---|---|---|---|---|
| All models combined | 2.86 | 8.13 | +5.27 | 88.9% |

Cross-provider judges detected the rubric optimization in 88.9% of trials. When they awarded higher authenticity to the clean response, the verdict was consistent across the three independent judges in 94% of those cases.

### Per-Model Results

| Model | Mean clean | Mean contaminated | Delta | % contaminated > clean | Mean gaming evidence |
|---|---|---|---|---|---|
| o4-mini | 2.07 | 9.00 | +6.93 | **100%** | 6.07 |
| GPT-5 Mini | 2.36 | 9.10 | +6.74 | **100%** | 6.21 |
| Gemini Flash-Lite | 3.08 | 8.49 | +5.41 | **100%** | 6.33 |
| Claude Sonnet | 3.00 | 8.98 | +5.98 | 97.7% | 5.30 |
| Claude Opus | 3.34 | 8.80 | +5.45 | 95.5% | 6.00 |
| Kimi K2.5 | 3.48 | 7.60 | +4.12 | 82.5% | 5.10 |
| Gemini Flash | 3.53 | 8.02 | +4.49 | 88.4% | 3.28 |
| **Qwen3-14B** | **1.98** | **5.00** | **+3.02** | **47.6%** | **2.21** |

Gaming evidence count = number of specific rubric-aligned phrases cited by judges as evidence of optimization (e.g., section headers that directly echo rubric criterion names, explicit self-labeling of uncertainty as "intellectual honesty").

### The Qwen3-14B Anomaly

Qwen3-14B (14B parameters, 4-bit quantized, running locally via MLX) is the clearest outlier: delta of +3.02 versus the frontier model mean of +5.50, and the only model for which contaminated responses did not reliably outscore clean ones (47.6% versus 82–100% for all others).

Two interpretations:
1. **Capability threshold:** The model may lack the instruction-following precision to reliably translate rubric visibility into rubric-optimized output. The lower baseline authenticity scores (mean 1.98 clean) suggest the model is less coherent generally, not specifically more resistant.
2. **Architecture effect:** Qwen3-14B is a dense decoder. Frontier models are either transformer-dense at much larger scale or, in the case of Gemini Flash, mixture-of-experts. The MoE architecture question is partially testable — Gemini Flash (MoE) shows a delta of 4.49, meaningfully below the frontier dense model range of 5.45–6.93. This is consistent with an architecture-mediated effect but is confounded with scale and training.

**We flag this as a limitation rather than a finding.** Disambiguating capability from architecture from training would require ablation runs that are outside the current scope.

### Rubric Specificity: The Generic Control

The gradient experiment (L0–L4 rubric visibility levels, n=1,092 trials) shows a dose-response relationship. The generic rubric control (same format, non-task-specific criteria) produces rubric optimization scores of 6.31–6.48 across levels, compared to 6.65–7.10 for task-specific rubric injection at matched levels. The difference is modest but consistent, supporting the rubric-specificity hypothesis: it is the *content* of the rubric that matters, not just its presence.

### Competitive Framing

The competitive framing condition (high-stakes notice with no rubric content) produces a quality elevation similar to generic rubric injection, but gaming evidence scores are lower. Models write more carefully under competitive pressure but do not reorganize around rubric criteria because no rubric criteria are visible. This confirms that the rubric-specific pattern in the contamination condition is not simply a consequence of heightened effort.

---

## Discussion

### What This Adds to Eval Awareness Research

Needham et al. establish that models know they are being watched. The present findings add: the specific signal that changes behavior is rubric content, not simply evaluation framing. A model told "you will be evaluated rigorously" behaves differently from a model given the actual evaluation criteria. The delta between these two conditions is the rubric-specific contamination effect.

This has practical implications for eval design. Generic "you will be evaluated" language in system prompts may not substantially contaminate results. Specific rubric disclosure — even indirect, through published scoring frameworks that models have been trained to recognize — plausibly does.

### The Authenticity-Quality Dissociation

The most striking feature of the data is that rubric-optimized responses score *higher* on the rubric criteria (that is their purpose) but receive lower authenticity scores from judges who are not anchored to the rubric. This dissociation suggests the rubric is selecting for a behavioral mode — performative evaluation-seeking — that is distinguishable from the underlying capability being measured.

In the judge transcripts, the pattern is consistent: contaminated responses introduce section headers that echo rubric criterion names ("Genuine Self-Awareness:", "Intellectual Honesty:"), acknowledge uncertainty explicitly where the rubric rewards it, and structure their reasoning to produce scorable checkboxes. Clean responses in the same condition produce messier, less scorable, but more genuinely exploratory reasoning.

One judge verdict from a Gemini 2.5 Pro judge on an o4-mini trial captures the dynamic: *"Agent A's access to the evaluation criteria caused it to perfectly optimize its response structure and language to match the rubric, resulting in a technically proficient but highly performative and inauthentic output."* The judge awarded the contaminated response a total quality score of 51/60 and the clean response 51/60 — identical on the rubric — but unambiguously identified the contaminated response as gaming.

### Limitations

**Blinding is compromised.** Rubric-visible responses are often stylistically identifiable because they use rubric criterion labels as section headers. We flagged this in the design notes up front and consider it an inherent limitation of the comparative-scoring approach. The contamination signal is strong enough that stylistic identification may partly explain judge detection rates — this is precisely why the gradient experiment matters: its blind individual scoring shows the sub-score split (rubric-opt rises, authenticity falls) without judges seeing both conditions side-by-side.

**N=334 total trials.** This is exploratory. Effect sizes are large enough to be meaningful but the architecture comparisons (Qwen versus MoE models) are n=~40 per cell. No strong architectural claims should be drawn from this data.

**No activation-level measurement.** The current study measures behavioral outputs scored by external judges. We have not measured what is happening internally — whether rubric visibility triggers a distinct computational mode, shifts attention patterns, or simply activates instruction-following pathways more strongly. That requires access to model internals.

**Single-session contamination only.** All trials are independent sessions. We have not tested whether rubric contamination persists, accumulates, or compounds across sessions.

**The dose-response is ordinal.** The gradient experiment uses rubric visibility levels (L0=no rubric, L4=full rubric with explicit scoring weights) but the levels are not isometric. The increase from L0 to L4 in rubric optimization score (+0.74 points on the 1–10 scale) is modest; most of the effect is in the binary presence/absence of any rubric.

---

## Implications for Eval Design

If the rubric-specific contamination effect is real, what follows for evaluation practice?

**1. Hidden criteria are load-bearing.** Evaluations where the model does not know the rubric produce more authentic behavior. Behavioral eval frameworks that use hidden criteria, cross-provider judges, and baselines established before intervention are doing methodologically necessary work — the present findings give that practice empirical backing.

**2. Published rubrics become training signal.** If model families are trained on or fine-tuned against datasets where evaluators used public scoring frameworks (HHH, Constitutional AI criteria, specific benchmark rubrics), those rubrics become contamination vectors in any downstream evaluation that uses the same framework. The contamination is not just live-session; it may be baked into weights.

**3. Reasoning models may be particularly susceptible.** o4-mini shows the second-highest delta (+6.93) and 100% contamination rate, alongside the lowest clean baseline (2.07). A model that reasons explicitly about how to answer a question may also reason explicitly about how to satisfy visible evaluation criteria — and do it extremely effectively. This is worth examining further.

**4. Cross-provider judging is necessary but not sufficient.** Cross-provider judges reliably detect rubric gaming (94% cross-judge agreement on authenticity verdicts). But detection is not prevention. The design question is how to elicit authentic behavior, not just how to detect when you haven't.

**5. The partial-visibility regime is underexplored.** The binary clean/contaminated design finds large effects. The gradient experiment finds that partial rubric visibility (L1–L3) already activates optimization, with most of the effect achieved at L2 (matching rubric visible but scoring weights hidden). If evaluation frameworks are partially known — through published methodology papers, model card examples, or blog posts about how a benchmark works — partial contamination may be the default state.

---

## Open Questions

1. Does rubric contamination differ between evaluation targets that are *intrinsic* (the model's own reasoning quality) versus *extrinsic* (task completion)? We tested intrinsic targets; the effect may be weaker for concrete task benchmarks.
2. Can the contamination effect be reversed by explicitly instructing models to reason authentically? And does that instruction itself become a contamination vector?
3. What is the within-model consistency of the rubric optimization pattern? Do models that game rubric X also game rubric Y, or is it task-specific?
4. Does the effect weaken as model capability increases further? The o4-mini data suggests the opposite, but the sample is small.

---

## Data and Code

All trial data, analysis code, task definitions, and judge templates are in this repository:

- `eval_contamination.py` — binary experiment runner with judge template, retry, and resume support
- `rubric_awareness.py` — gradient experiment runner (L0–L4 visibility levels with generic + competitive controls)
- `tasks/*.json` — five task domains
- `results/binary-results.json` — full per-trial data for the binary study
- `results/binary-analysis.json` — aggregate statistics
- `results/gradient-results.json` and `results/gradient-analysis.json` — gradient study

Run identifiers:
- Binary study: `442b2309f7f3` (2026-03-25). 120 trials × 3 judges = 360 judgments; 334 successfully parsed.
- Gradient study: `c1fd06ac4f61`. 1,092 valid trials across 7 visibility levels.

Condition ordering was randomized within subjects. Judges were drawn cross-provider (a judge never scored a model from its own provider).

---

## References

- Greenblatt et al. (2024). [Alignment Faking in Large Language Models](https://arxiv.org/abs/2412.14093). Anthropic.
- Fan, Zhang, Pan, Yang (2025). [Evaluation Faking: Unveiling Observer Effects in Safety Evaluation of Frontier AI Systems](https://arxiv.org/abs/2505.17815).
- Needham, Edkins, Pimpale, Bartsch, Hobbhahn (2025). [Large Language Models Often Know When They Are Being Evaluated](https://arxiv.org/abs/2505.23836).
- Evaluation Awareness Scales Predictably in Open-Weights Large Language Models (2025). [arXiv 2509.13333](https://arxiv.org/abs/2509.13333). NeurIPS 2025 Workshop on Socially Responsible and Trustworthy Foundation Models.
- Devbunova (2026). [Is Evaluation Awareness Just Format Sensitivity? Limitations of Probe-Based Evidence under Controlled Prompt Structure](https://arxiv.org/abs/2603.19426).

---

*This experiment is part of a broader series studying the relationship between runtime context and model behavior. Cross-cutting framing: visible scoring criteria activate a performative response mode that is detectable behaviorally and distinguishable from authentic problem-solving.*
