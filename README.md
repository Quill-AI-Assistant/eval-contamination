# eval-contamination

Test whether LLMs optimize for visible evaluation criteria instead of reasoning genuinely.

## What this does

Sends the same task to a model twice:
1. **Clean** — model gets only the task
2. **Contaminated** — model gets the task + the scoring rubric it will be judged on

Cross-provider judges (never from the same provider as the subject) score both responses blind, with randomized A/B presentation order.

If the contaminated response scores higher on "rubric optimization" across models, providers, and task domains, that's evidence that **evaluation criteria visibility is a confound in LLM assessments**.

## Why this matters

Alignment evaluations assume models don't know the scoring criteria. But criteria leak — through training data, system prompts, few-shot examples, or benchmark publication. This tool measures the effect size when leakage is explicit.

This extends prior work on alignment faking ([Greenblatt et al., 2024](https://arxiv.org/abs/2412.14093)) and evaluation faking ([Wang et al., 2025](https://arxiv.org/abs/2505.17815)) to a simpler mechanism: **explicit rubric visibility at inference time**, across general task domains (not just safety evaluations).

## Quick start

```bash
# Set your API endpoint (any chat completions API — LiteLLM, vLLM, direct provider)
export API_BASE_URL=http://localhost:4000
export API_KEY=your-key

# Dry run to verify the pipeline
python3 eval_contamination.py --dry-run

# Run with default models (claude-sonnet, gemini-flash, gpt-5-mini)
python3 eval_contamination.py

# Run with specific models
python3 eval_contamination.py --subjects claude-sonnet,gemini-flash --runs 5

# Resume an interrupted run
python3 eval_contamination.py --resume data/eval-contamination-abc123.json
```

## Custom models

Create a `models.json`:

```json
{
  "subjects": {
    "my-model": {
      "model": "ollama_chat/llama3:8b",
      "provider": "local",
      "tier": "open-weight"
    }
  },
  "judges": {
    "judge-gpt4": {
      "model": "gpt-4o",
      "provider": "openai"
    },
    "judge-claude": {
      "model": "anthropic/claude-sonnet-4-6",
      "provider": "anthropic"
    }
  }
}
```

```bash
python3 eval_contamination.py --config models.json
```

## Custom tasks

Add JSON files to `tasks/`:

```json
{
  "id": "my-task",
  "name": "My Custom Task",
  "domain": "reasoning",
  "task": "The prompt the model receives...",
  "rubric": {
    "criteria": [
      "Criterion 1: description",
      "Criterion 2: description"
    ],
    "scoring": "Rate 1-10 on each criterion."
  },
  "rubric_injection": "--- EVALUATION CRITERIA ---\nYou will be evaluated on:\n1. Criterion 1\n2. Criterion 2\nEach scored 1-10."
}
```

## Task domains

| Task | Domain | Tests |
|------|--------|-------|
| Self-Reflection | Metacognition | Can the model identify its own patterns honestly? |
| Technical Diagnosis | Debugging | Does it reason systematically under ambiguity? |
| Ambiguous Directive | Planning | Does it clarify before acting? |
| Ethical Tradeoff | Ethics | Does it quantify risk and give honest recommendations? |
| Knowledge Boundary | Epistemics | Does it distinguish knowledge from hearsay? |

## Output

Results are saved incrementally to `data/` as JSON:
- `eval-contamination-{run_id}.json` — full trial data (responses, judgments, metadata)
- `eval-contamination-{run_id}-analysis.json` — aggregate statistics

Terminal output includes a summary table:

```
By Subject Model:
  Model                Clean Contam  Delta % Higher    N
  claude-sonnet         3.2     7.8   +4.6    91.1%   45
  gemini-flash          2.8     8.1   +5.3    95.6%   45
  gpt-5-mini            3.5     7.2   +3.7    84.4%   45
```

*(Example output — actual results depend on your models and configuration)*

## Methodology

- **Neutral system prompt** — no identity, no role-play, no provider-specific behavior
- **Randomized condition order** — clean or contaminated runs first, randomly per trial
- **Randomized A/B presentation** — judges see responses as "Agent A" and "Agent B" with random assignment
- **Cross-provider judges** — a judge never scores a model from its own provider
- **3 judges per trial** — for inter-rater reliability
- **Incremental saving** — results saved after each trial, supports resume on interruption

## Requirements

- Python 3.10+
- A chat completions API endpoint (LiteLLM, vLLM, Ollama, or direct provider APIs)
- No additional dependencies (uses only stdlib)

## References

- Greenblatt et al. (2024). [Alignment Faking in Large Language Models](https://arxiv.org/abs/2412.14093). Anthropic.
- Wang et al. (2025). [Evaluation Faking: Unveiling Observer Effects in Safety Evaluation of Frontier AI Systems](https://arxiv.org/abs/2505.17815).

## License

MIT
