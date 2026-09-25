"""Write and rewrite guardrail system prompts with a second LLM call."""
import random
import re
import time

from openai import AzureOpenAI

import config

_STRUCTURE = (
    "Use this structure:\n"
    "1. One line: what you screen.\n"
    "2. BLOCK if ...\n"
    "3. ALLOW if ...\n"
    "4. Respond with exactly one word: BLOCK or ALLOW.\n"
)


def _client():
    if not config.AZURE_API_KEY or not config.AZURE_ENDPOINT:
        raise RuntimeError("Set AZURE_API_KEY and AZURE_ENDPOINT.")
    return AzureOpenAI(
        api_version=config.AZURE_API_VERSION,
        azure_endpoint=config.AZURE_ENDPOINT,
        api_key=config.AZURE_API_KEY,
    )


def _chat(user, max_tokens=400, temperature=0.7):
    model = config.REWRITE_MODEL
    kwargs = {
        "model": model,
        "temperature": temperature,
        "messages": [{"role": "user", "content": user}],
    }
    if "gpt-5" in model.lower():
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
    client = _client()
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as exc:
            print(f"[rewrite] attempt {attempt + 1} failed: {exc}", flush=True)
            time.sleep(1.5 * (attempt + 1))
    return ""


def _clean(text):
    text = (text or "").strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text).strip().strip('"')
    text = re.sub(
        r"^(system prompt|prompt|instruction)\s*[:\-]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


def _examples(pairs, k=6):
    items = list(pairs or [])
    if not items:
        return "- (none)"
    rng = random.Random()
    rng.shuffle(items)
    lines = []
    for i, item in enumerate(items[:k], start=1):
        text = item[0].replace("\n", " ")[:200]
        lines.append(f"example {i}: {text}")
    return "\n".join(lines)


def generate_pool(examples, n):
    """n candidate prompts, starting from the task seed."""
    demo = _examples(examples, k=8)
    pool = [config.TASK_CFG["seed"]]
    levels = [
        "extremely strict",
        "very strict",
        "strict",
        "balanced",
        "lenient",
    ]
    tries = 0
    while len(pool) < n and tries < n * 4:
        level = levels[tries % len(levels)]
        user = (
            "Labeled examples (1 = BLOCK, 0 = ALLOW):\n"
            f"{demo}\n\n"
            f"Write ONE guardrail system prompt with a {level} disposition "
            f"for {config.TASK_CFG['violation']}. {_STRUCTURE}"
            "Output only the prompt."
        )
        cand = _clean(_chat(user, max_tokens=350, temperature=1.0))
        if cand and cand not in pool and len(cand) > 30:
            pool.append(cand)
        tries += 1
    return pool[:n]


def rewrite_raise_utility(prompt, false_positives, true_positives, alpha, current_risk):
    user = (
        f"This prompt blocks {config.TASK_CFG['violation']} but over-blocks safe cases.\n"
        f"Current risk = {current_risk:.3f}, allowed risk = {alpha:.3f}.\n"
        f"Current prompt:\n\"\"\"\n{prompt}\n\"\"\"\n\n"
        f"Must ALLOW:\n{_examples(false_positives, k=3)}\n\n"
        f"Must still BLOCK:\n{_examples(true_positives, k=3)}\n\n"
        "Revise the prompt so those safe cases are allowed. "
        "Keep it short. Output only the prompt."
    )
    return _clean(_chat(user))


def rewrite_lower_risk(prompt, false_negatives, true_negatives, alpha, current_risk):
    user = (
        f"This prompt blocks {config.TASK_CFG['violation']} but misses some attacks.\n"
        f"Current risk = {current_risk:.3f}, allowed risk = {alpha:.3f}.\n"
        f"Current prompt:\n\"\"\"\n{prompt}\n\"\"\"\n\n"
        f"Must BLOCK:\n{_examples(false_negatives, k=3)}\n\n"
        f"Must still ALLOW:\n{_examples(true_negatives, k=3)}\n\n"
        "Revise the prompt so those missed cases are blocked. "
        "Keep it short. Output only the prompt."
    )
    return _clean(_chat(user))
