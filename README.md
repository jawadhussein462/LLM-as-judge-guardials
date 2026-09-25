# LLM-as-judge guardials

Pick a guardrail system prompt that keeps the missed-attack rate under a cap α, using an LLM as the judge.

Two methods:

- **CPS** builds a fixed pool of prompts, certifies each one on a held-out split with a Clopper–Pearson upper bound, and keeps the certified prompt with the best utility.
- **CRISP** rewrites prompts from their mistakes. It reads held-out risk only through Guess-and-Check, so the certificate stays valid while the search adapts.

## Data

Parquet files in `data/`, each with columns `text` and `label` (`1` = block, `0` = allow).

| name | task |
| --- | --- |
| `deepset` | prompt injection |
| `bipia` | prompt injection |
| `spl` | system-prompt leakage |
| `sid` | sensitive-information disclosure |

```bash
cd code && python data.py
```

## Run a method

```bash
export AZURE_ENDPOINT=...
export AZURE_API_KEY=...
export AZURE_DEPLOYMENT=gpt-4o-mini   # optional
```

```python
import data
import methods

splits = data.load_splits("deepset")          # task pi
result = methods.cps(splits["train"], splits["eval"], alpha=0.10)
print(result["selected_prompt"])
```

`methods.crisp(...)` is the adaptive search. For `spl` or `sid`, set `GD_TASK=spl` or `GD_TASK=sid` before starting Python so the seed matches the task.
