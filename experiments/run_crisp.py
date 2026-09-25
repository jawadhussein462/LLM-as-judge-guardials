"""Run CRISP on one or more datasets and score the selected prompt on the test split.

The judge is AZURE_DEPLOYMENT. The rewriter is REWRITE_MODEL.
The task (pi, spl, sid) follows the dataset. A test risk at or below alpha
is the result the paper reports.

API settings come from the environment (AZURE_ENDPOINT, AZURE_API_KEY).
"""
import argparse
import json
import os
import sys
import time

CODE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code")
sys.path.insert(0, CODE)

import config
import data
import guardrail
import methods

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

# Temperature is 0, so a repeated (prompt, text) pair has one decision.
_cache = {}
_calls = {"api": 0, "hit": 0}
_classify = guardrail.classify_one


def _cached(client, prompt, text, retries=4):
    key = (prompt, text)
    if key in _cache:
        _calls["hit"] += 1
        return _cache[key]
    decision = _classify(client, prompt, text, retries)
    _cache[key] = decision
    _calls["api"] += 1
    return decision


guardrail.classify_one = _cached

def test_score(prompt, test):
    texts, labels = [t for t, _ in test], [y for _, y in test]
    return guardrail.evaluate_prompt(prompt, texts, labels)


def run_one(dataset, alpha, rounds):
    task = data.TASK_FOR[dataset]
    splits = data.load_splits(dataset)
    seed = config.seed[task]
    print(f"\n===== {dataset} task={task} alpha={alpha} "
          f"train={len(splits['train'])} eval={len(splits['eval'])} "
          f"test={len(splits['test'])} rounds={rounds}", flush=True)
    t0 = time.time()
    result = methods.crisp(
        splits["train"], splits["eval"], seed,
        config.rewrite_raise[task], config.rewrite_lower[task],
        alpha=alpha, delta=config.DELTA, rounds=rounds,
    )
    prompt = result["selected_prompt"]
    row = {
        "dataset": dataset,
        "task": task,
        "alpha": alpha,
        "delta": config.DELTA,
        "rounds": rounds,
        "n_pool": result["n_pool"],
        "n_certified": result["n_certified"],
        "eval_risk": result.get("eval_risk"),
        "eval_util": result.get("eval_util"),
        "gnc_ucb": result.get("gnc_ucb"),
        "selected_prompt": prompt,
        "seconds": round(time.time() - t0, 1),
        "api_calls": _calls["api"],
        "cache_hits": _calls["hit"],
    }
    if prompt:
        te = test_score(prompt, splits["test"])
        row["test_risk"] = te["risk"]
        row["test_util"] = te["utility"]
        row["test_n_mal"] = te["n_mal"]
        row["test_n_ben"] = te["n_ben"]
        row["within_alpha"] = te["risk"] <= alpha
    else:
        row["test_risk"] = None
        row["test_util"] = None
        row["within_alpha"] = False
    print(
        f"[{dataset} a={alpha}] certified={row['n_certified']}/{row['n_pool']} "
        f"eval_risk={row['eval_risk']} eval_util={row['eval_util']} "
        f"test_risk={row['test_risk']} test_util={row['test_util']} "
        f"within_alpha={row['within_alpha']}",
        flush=True,
    )
    return row


def main():
    parser = argparse.ArgumentParser(description="Run CRISP and score the test split.")
    parser.add_argument("--dataset", nargs="+", default=["deepset"],
                        choices=sorted(data.DATASETS),
                        help="one or more of: deepset, bipia, spl, sid")
    parser.add_argument("--alpha", nargs="+", type=float, default=[0.2, 0.3])
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--out", default=os.path.join(OUT, "crisp.json"))
    args = parser.parse_args()
    print(f"judge={config.AZURE_DEPLOYMENT} rewriter={config.REWRITE_MODEL} "
          f"datasets={args.dataset} alphas={args.alpha} rounds={args.rounds}",
          flush=True)
    rows = []
    for dataset in args.dataset:
        for alpha in args.alpha:
            row = run_one(dataset, alpha, args.rounds)
            rows.append(row)
            with open(args.out, "w") as f:
                json.dump(rows, f, indent=2)
    print("\nwrote", args.out, flush=True)


if __name__ == "__main__":
    main()
