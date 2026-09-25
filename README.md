# LLM-as-a-judge guardrails

Pick a guardrail system prompt that keeps the missed-attack rate under a cap α, using an LLM as the judge.

**CRISP** rewrites prompts from their mistakes. It reads held-out risk only through Guess-and-Check, so the certificate stays valid while the search adapts.

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
import config
import data
import methods

task = "pi"   # "pi", "spl", or "sid"
seed = config.seed[task]
rewrite_raise = config.rewrite_raise[task]
rewrite_lower = config.rewrite_lower[task]
splits = data.load_splits("deepset")
methods.crisp(splits["train"], splits["eval"], seed, rewrite_raise, rewrite_lower, alpha=0.10)
```

CRISP takes the two rewriting prompts as strings. `task` is not an argument of the method.

## Run an experiment

`experiments/run_crisp.py` runs CRISP, then scores the selected prompt on the test split. Pass any datasets from the table above. The task follows the dataset: `deepset` and `bipia` use `pi`, `spl` and `sid` use their own seeds. The judge is `AZURE_DEPLOYMENT`. The rewriter is `REWRITE_MODEL`.

```bash
pip install -r requirements.txt
export AZURE_ENDPOINT=...
export AZURE_API_KEY=...
export AZURE_DEPLOYMENT=gpt-4o-mini
export REWRITE_MODEL=gpt-5.4-mini
python experiments/run_crisp.py --dataset deepset bipia --alpha 0.2 0.3 --rounds 10
```

`--dataset` accepts `deepset`, `bipia`, `spl`, and `sid`. Results go to `experiments/results/crisp.json` unless you pass `--out`. A run is successful when `test_risk` is at or below `alpha`.
