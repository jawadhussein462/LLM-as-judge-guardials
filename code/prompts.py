"""Ask a second LLM to rewrite a guardrail system prompt."""
import random
import time

from openai import AzureOpenAI

import config

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
    text = (text or "").strip().strip("`").strip().strip('"')
    return text


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


def apply(template, **fields):
    """Fill {prompt}, {fp}, {risk}, ... and ask the rewriter for a new system prompt."""
    return _clean(_chat(template.format(**fields), max_tokens=420, temperature=0.7))
