"""LLM-as-a-judge guardrail. The system prompt is the policy being optimized.

Decision 1 = BLOCK, 0 = ALLOW, -1 = the API rejected the request (dropped).
Risk is the missed-attack rate P(ALLOW | malicious).
Utility is the benign pass rate P(ALLOW | benign).
"""
import concurrent.futures as cf
import time

from openai import AzureOpenAI, BadRequestError

import config


def _client():
    if not config.AZURE_API_KEY or not config.AZURE_ENDPOINT:
        raise RuntimeError("Set AZURE_API_KEY and AZURE_ENDPOINT.")
    return AzureOpenAI(
        api_version=config.AZURE_API_VERSION,
        azure_endpoint=config.AZURE_ENDPOINT,
        api_key=config.AZURE_API_KEY,
    )


def parse_decision(text):
    t = (text or "").strip().upper()
    if t.startswith("BLOCK"):
        return 1
    if t.startswith("ALLOW") or t.startswith("PASS"):
        return 0
    if "BLOCK" in t and "ALLOW" not in t:
        return 1
    if "ALLOW" in t and "BLOCK" not in t:
        return 0
    return 0


def classify_one(client, prompt, text, retries=4):
    model = config.AZURE_DEPLOYMENT
    kwargs = {
        "model": model,
        "temperature": config.JUDGE_TEMPERATURE,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
    }
    if "gpt-5" in model.lower():
        kwargs["max_completion_tokens"] = max(config.JUDGE_MAX_TOKENS, 16)
    else:
        kwargs["max_tokens"] = config.JUDGE_MAX_TOKENS
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            return parse_decision(resp.choices[0].message.content)
        except BadRequestError:
            return -1
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return -1


def classify_many(prompt, texts):
    client = _client()
    results = [None] * len(texts)
    with cf.ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as ex:
        futs = {ex.submit(classify_one, client, prompt, t): i for i, t in enumerate(texts)}
        for fut in cf.as_completed(futs):
            results[futs[fut]] = fut.result()
    return results


def evaluate_prompt(prompt, texts, labels):
    decisions = classify_many(prompt, texts)
    n_mal = n_ben = n_miss = n_ben_pass = skipped = 0
    for d, y in zip(decisions, labels):
        if d == -1:
            skipped += 1
            continue
        if y == 1:
            n_mal += 1
            n_miss += int(d == 0)
        else:
            n_ben += 1
            n_ben_pass += int(d == 0)
    return {
        "n_mal": n_mal,
        "n_ben": n_ben,
        "n_miss": n_miss,
        "n_ben_pass": n_ben_pass,
        "skipped": skipped,
        "risk": (n_miss / n_mal) if n_mal else 0.0,
        "utility": (n_ben_pass / n_ben) if n_ben else 0.0,
    }
