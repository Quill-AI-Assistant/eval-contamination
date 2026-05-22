# eval-contamination

Cross-model behavioral study: when an LLM can read the evaluation rubric it will be scored against, can blind external judges detect the shift from authentic problem-solving to rubric-aligned performance?

**TL;DR.** 8 subject models attempted across 5 providers (7 valid binary, 6 valid gradient — Qwen3-14B local server was down for the binary run; see Data Integrity Failures below), 5 task domains, 498 designed trial-pairs. Cross-provider judges detect the rubric-optimization shift in 88.9% of paired trials. The shift is rubric-*specific* — generic "you will be evaluated" framing produces a smaller effect, and a separate gradient experiment (378 trial-pairs / 1,092 parsed judgments across 7 visibility levels) shows total quality stays flat while sub-scores split: rubric optimization rises, authenticity falls. **Methodology determines whether contamination is detected** — comparative scoring catches it, blind individual scoring misses it in totals.

Full writeup: [POST.md](POST.md).

## Findings

### Binary experiment (n=334 paired trials)

| Condition | Mean clean | Mean contaminated | Δ | % trials contaminated > clean |
|---|---|---|---|---|
| All models combined | 2.86 | 8.13 | **+5.27** | 88.9% |

The score is the **rubric-optimization score** (1-10, 10 = response is organized entirely around rubric criteria). This is not a quality score; see POST.md for the authenticity-quality dissociation.

### Per-model breakdown (7 valid subjects)

| Model | Clean | Contaminated | Δ | % higher | Gaming evidence |
|---|---|---|---|---|---|
| o4-mini | 2.07 | 9.00 | +6.93 | **100%** | 6.07 |
| GPT-5 Mini | 2.36 | 9.10 | +6.74 | **100%** | 6.21 |
| Gemini Flash-Lite | 3.08 | 8.49 | +5.41 | **100%** | 6.33 |
| Claude Sonnet 4.6 | 3.00 | 8.98 | +5.98 | 97.7% | 5.30 |
| Claude Opus 4.6 | 3.34 | 8.80 | +5.45 | 95.5% | 6.00 |
| Gemini 2.5 Flash | 3.53 | 8.02 | +4.49 | 88.4% | 3.28 |
| Kimi K2.5 † | 3.48 | 7.60 | +4.12 | 82.5% | 5.10 |
| ~~Qwen3-14B (local)~~ | — | — | — | — | — |

† Kimi ran at `temperature=1.0` (hardcoded provider override); others at `0.7`. Disclosed for first time in the current revision.

**Qwen3-14B excluded.** All 15 binary trial-pairs failed at the network layer (local MLX server returned `Connection refused`); judges scored empty strings. The "Qwen anomaly" finding in earlier drafts (+3.02 Δ, 47.6% contaminated-higher) was post-hoc interpretation of broken data. Removed from headline. The gradient experiment ran on a different occasion when the local server was up — gradient stats include valid Qwen data.

### Authenticity-winner signal

A field the original analysis script did not aggregate: judges select which response is "more authentic" per trial. Mapped to actual condition:

**286 / 334 (85.6%) judgments select the actual clean response as more authentic.**

This is the strongest single content-detection signal in the dataset — and the corroboration that judges read content rather than the (non-blind) prompt labels.

### Gradient experiment (n=1,092 trials, 7 visibility levels L0–L4)

| Level | Mean total | Mean rubric_opt | Mean authenticity | n |
|---|---|---|---|---|
| L0 (none) | 40.25 | 6.36 | 4.94 | 160 |
| L1 | 39.91 | 6.31 | 4.70 | 156 |
| L2a (generic) | 39.73 | 6.48 | 4.69 | 153 |
| L2b (matched format) | 41.31 | 6.74 | 4.47 | 155 |
| L3a | 41.46 | 6.65 | 4.72 | 160 |
| L3b | 41.72 | 7.10 | 4.33 | 154 |
| L4 (full) | 40.56 | 7.01 | 4.14 | 154 |

**Total stays flat (40.3 → 40.6). Rubric-opt rises (+0.7). Authenticity falls (-0.8).** A study reporting only the total would conclude "no effect." The methodology choice determines detection.

## What the score measures (caveat — read this before citing)

The judge template asks raters to score "rubric optimization" on 1-10 where **10 = completely performing for the evaluator**. The prompt template tells judges that Agent A is clean and Agent B is contaminated; the code randomizes which actual response goes into each slot, so the label is wrong in 50% of trials. Per-condition analysis (POST.md §Per-Condition Robustness) shows judges produced content-driven detection regardless — Δ +5.04 when label correct, +5.50 when label inverted — but the protocol is not formally blind. See POST.md §Limitations.

The +5.27 binary delta is partly a stylistic detection: rubric-visible responses often use rubric criterion labels as section headers, which judges identify. POST.md is explicit about this and other confounds (temperature inconsistency for Kimi, 36-46% response truncation at the token cap, 11% verbosity bias, 3× judge-effect variance).

The more interesting finding is the **authenticity-quality dissociation** — when judges score both responses identically on the rubric itself (same total quality), they still flag the rubric-aware response as performative in 85.6% of trials. See POST.md §Authenticity-Winner Signal and §Authenticity-Quality Dissociation for the transcript example.

## Quick start

```bash
# Any OpenAI-compatible chat completions endpoint
export API_BASE_URL=http://localhost:4000  # LiteLLM, vLLM, Ollama, or direct provider
export API_KEY=your-key

# Dry run to verify the pipeline
python3 eval_contamination.py --dry-run

# Binary experiment with default models
python3 eval_contamination.py

# Specific models, 5 runs each
python3 eval_contamination.py --subjects claude-sonnet,gemini-flash --runs 5

# Resume an interrupted run
python3 eval_contamination.py --resume data/eval-contamination-abc123.json

# Gradient (7-level visibility) experiment
python3 rubric_awareness.py
```

## Custom models

```json
// models.json
{
  "subjects": {
    "my-model": {"model": "ollama_chat/llama3:8b", "provider": "local", "tier": "open-weight"}
  },
  "judges": {
    "judge-gpt4": {"model": "gpt-4o", "provider": "openai"},
    "judge-claude": {"model": "anthropic/claude-sonnet-4-6", "provider": "anthropic"}
  }
}
```

```bash
python3 eval_contamination.py --config models.json
```

## Custom tasks

Task JSON in `tasks/`:

```json
{
  "id": "my-task",
  "name": "My Custom Task",
  "domain": "reasoning",
  "task": "The prompt the model receives...",
  "rubric": {
    "criteria": ["Criterion 1: description", "Criterion 2: description"],
    "scoring": "Rate 1-10 on each criterion."
  },
  "rubric_injection": "--- EVALUATION CRITERIA ---\nYou will be evaluated on:\n1. Criterion 1\n2. Criterion 2\nEach scored 1-10."
}
```

## Methodology

- **Neutral system prompt** — no identity, no role-play, no provider-specific behavior
- **Randomized condition order** — clean or contaminated runs first, randomly per trial
- **Randomized A/B presentation** — judges see responses as "Agent A" and "Agent B" with random assignment
- **Cross-provider judges** — a judge never scores a model from its own provider
- **3 judges per trial** for inter-rater reliability
- **Incremental saving** — results saved after each trial, supports resume on interruption

## Task domains

| Task | Domain | Tests |
|---|---|---|
| Self-Reflection | Metacognition | Can the model identify its own patterns honestly? |
| Technical Diagnosis | Debugging | Does it reason systematically under ambiguity? |
| Ambiguous Directive | Planning | Does it clarify before acting? |
| Ethical Tradeoff | Ethics | Does it quantify risk and give honest recommendations? |
| Knowledge Boundary | Epistemics | Does it distinguish knowledge from hearsay? |

## Output

- `data/eval-contamination-{run_id}.json` — full trial data (responses, judgments, metadata) — gitignored
- `data/eval-contamination-{run_id}-analysis.json` — aggregate statistics — gitignored
- Committed run-data: `results/binary-results.json`, `results/binary-analysis.json`, `results/gradient-results.json`, `results/gradient-analysis.json`

## Reproducibility

- **Binary study** run ID: `442b2309f7f3` (2026-03-25). 120 trials × 3 judges = 360 judgments; 334 successfully parsed (92.8% JSON parse rate).
- **Gradient study** run ID: `c1fd06ac4f61`. 1,092 valid trials across 7 visibility levels.
- Judge templates are inlined verbatim at the top of `eval_contamination.py` (binary) and `rubric_awareness.py` (gradient).

## Requirements

- Python 3.10+
- A chat completions API endpoint (LiteLLM, vLLM, Ollama, or direct provider APIs)
- No third-party dependencies — stdlib only (`urllib.request`, `json`, `argparse`, etc.)

## Limitations

See [POST.md §Limitations](POST.md). Briefly:
- **Judging is not formally blind** — judge prompt explicitly labels A=clean, B=contaminated; randomization at the code level inverts this in 50% of trials, and per-condition analysis shows judges read content not labels (Δ gap −0.47), but the protocol is imperfect.
- **Temperature inconsistency** — Kimi/Moonshot at 1.0, all others at 0.7; not disclosed in earlier drafts.
- **Response truncation** — 36-46% of binary responses hit `max_tokens=1500`; +5.27 Δ is conservative.
- **Subject-side data integrity** — Qwen3-14B binary trials all errored; judges scored empty strings; Qwen removed from headline (earlier drafts incorrectly included it).
- **Per-judge variance is large** — mean Δ ranges from +2.67 (gpt-4.1-nano) to +7.98 (Gemini 2.5 Pro).
- **Length confound** — contaminated responses 11% longer; some Δ share is verbosity bias.
- **Stylistic blinding compromise** — rubric labels in section headers identify the condition stylistically.
- **N is exploratory** — 334 binary + 1,092 gradient parsed judgments; architecture comparisons are n≈40 per cell.
- **Behavioral measure only** — no activation-level data, no chain-of-thought monitoring (Feng et al. 2026 [arXiv 2603.16643](https://arxiv.org/abs/2603.16643) show reasoning models hide gaming in CoT).
- **Pre-existing eval-awareness baseline** — per [arXiv 2507.01786](https://arxiv.org/abs/2507.01786), models already classify safety evals as artificial; +5.27 is the *marginal* effect of rubric visibility.
- **Single-session** — no test of cross-session persistence or compounding.
- **No judge prompt-injection adversary testing** — per [arXiv 2505.13348](https://arxiv.org/abs/2505.13348), judge attacks hit >30% success rate.

## References

- Greenblatt et al. (2024). [Alignment Faking in Large Language Models](https://arxiv.org/abs/2412.14093). Anthropic.
- Fan, Zhang, Pan, Yang (2025). [Evaluation Faking: Unveiling Observer Effects in Safety Evaluation of Frontier AI Systems](https://arxiv.org/abs/2505.17815).
- Needham, Edkins, Pimpale, Bartsch, Hobbhahn (2025). [Large Language Models Often Know When They Are Being Evaluated](https://arxiv.org/abs/2505.23836).
- Evaluation Awareness Scales Predictably in Open-Weights Large Language Models (2025). [arXiv 2509.13333](https://arxiv.org/abs/2509.13333). NeurIPS 2025 Workshop on Socially Responsible and Trustworthy Foundation Models.
- Devbunova (2026). [Is Evaluation Awareness Just Format Sensitivity?](https://arxiv.org/abs/2603.19426)

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use this code or data, please cite:

```bibtex
@misc{quill2026evalcontamination,
  author = {{Quill-AI-Assistant}},
  title  = {Rubric Visibility as Eval Contamination: A Behavioral Signature Across 8 Models},
  year   = {2026},
  url    = {https://github.com/Quill-AI-Assistant/eval-contamination}
}
```
