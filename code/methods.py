"""Two ways to pick a guardrail system prompt under a risk cap alpha.

CPS   Certified Prompt Search. Take the prompts an optimizer returns,
      certify each one on the eval split with a Bonferroni Clopper--Pearson
      upper bound, and keep the certified prompt with the best utility
      lower bound.

CRISP Closed-loop search. Rewrite the current prompt from its train errors.
      Risk on the eval split is read only through Guess-and-Check.
      p+ is the best certified prompt (raise utility). p- is the
      lowest-risk uncertified prompt (lower risk).
"""
import config
import guardrail
import guess_and_check as gnc
import prompts
import stats


def _xy(split):
    return [t for t, _ in split], [y for _, y in split]


def _mine(prompt, train, want_y, want_d):
    texts = [t for t, y in train if y == want_y]
    decisions = guardrail.classify_many(prompt, texts)
    return [(t, want_y) for t, d in zip(texts, decisions) if d == want_d]


def cps(train, eval_split, seed, optimizer, alpha=config.ALPHA, delta=config.DELTA,
        log=print):
    """Certify the prompts an optimizer returns.

    optimizer(train, eval_split, seed) -> list of system prompts, as GEPA would.
    """
    texts, labels = _xy(eval_split)
    pool = []
    for p in optimizer(train, eval_split, seed):
        if p and p not in pool:
            pool.append(p)

    n = len(pool)
    if n == 0:
        log("[CPS] optimizer returned no prompts")
        return {"method": "CPS", "selected_prompt": None,
                "n_pool": 0, "n_certified": 0, "records": []}
    records = []
    for i, prompt in enumerate(pool):
        rec = guardrail.evaluate_prompt(prompt, texts, labels)
        rec["prompt"] = prompt
        rec["risk_ucb"] = stats.risk_ucb(rec["n_miss"], rec["n_mal"], delta / n)
        rec["certified"] = rec["risk_ucb"] <= alpha
        records.append(rec)
        log(f"[CPS] {i + 1}/{n} risk={rec['risk']:.3f} "
            f"ucb={rec['risk_ucb']:.3f} util={rec['utility']:.3f}")

    certified = [r for r in records if r["certified"]]
    if not certified:
        log("[CPS] nothing certified")
        return {"method": "CPS", "selected_prompt": None,
                "n_pool": n, "n_certified": 0, "records": records}

    k = len(certified)
    for r in certified:
        r["util_lcb"] = stats.util_lcb(r["n_ben_pass"], r["n_ben"], delta / k)
    chosen = max(certified, key=lambda r: r["util_lcb"])
    log(f"[CPS] selected util_lcb={chosen['util_lcb']:.3f}")
    return {
        "method": "CPS",
        "selected_prompt": chosen["prompt"],
        "eval_risk": chosen["risk"],
        "eval_risk_ucb": chosen["risk_ucb"],
        "eval_util": chosen["utility"],
        "n_pool": n,
        "n_certified": k,
        "records": records,
    }


def _rewrite(parent, train, alpha, seen, template, kind):
    """Fill one rewriting prompt and ask for one new system prompt."""
    prompt = parent["prompt"]
    risk = parent.get("train_risk", parent.get("risk", 0.0))
    fields = {
        "prompt": prompt,
        "alpha": f"{alpha:.3f}",
        "risk": f"{risk:.3f}",
    }
    if kind == "raise":
        fp = _mine(prompt, train, 0, 1)
        tp = _mine(prompt, train, 1, 1) if fp else []
        fields["fp"] = prompts._examples(fp, k=3)
        fields["tp"] = prompts._examples(tp, k=3)
    else:
        fn = _mine(prompt, train, 1, 0)
        tn = _mine(prompt, train, 0, 0) if fn else []
        fields["fn"] = prompts._examples(fn, k=3)
        fields["tn"] = prompts._examples(tn, k=3)
    cand = prompts.apply(template, **fields)
    if cand and cand not in seen and len(cand) > 30:
        seen.add(cand)
        return [cand]
    return []


def crisp(train, eval_split, seed, rewrite_raise, rewrite_lower,
          alpha=config.ALPHA, delta=config.DELTA,
          rounds=config.CRISP_ROUNDS, log=print):
    """Adaptive search. Certification uses Guess-and-Check on risk only.

    seed is the starting system prompt. rewrite_raise and rewrite_lower
    are the rewriting prompt strings.
    """
    tr_texts, tr_labels = _xy(train)
    ev_texts, ev_labels = _xy(eval_split)
    n_mal_h = sum(ev_labels)
    stream = gnc.GuessAndCheck(beta=delta, n_h=n_mal_h)

    records = []
    seen = set()
    p_plus = p_minus = None

    def score(prompt):
        g = guardrail.evaluate_prompt(prompt, tr_texts, tr_labels)
        beta_i = stream._beta_i()
        a_g, tau = gnc.cp_tau_upper(g["n_miss"], g["n_mal"], beta_i)
        tau = max(alpha - a_g, tau)
        train_ucb = a_g + tau
        rec = {
            "prompt": prompt,
            "train_risk": g["risk"],
            "train_util": g["utility"],
            "certified": False,
            "eval_risk": None,
            "eval_util": None,
            "gnc_ucb": train_ucb,
        }
        dominated = p_minus is not None and p_minus["train_risk"] < g["risk"]
        if dominated or train_ucb > alpha:
            records.append(rec)
            return rec
        h = guardrail.evaluate_prompt(prompt, ev_texts, ev_labels)
        ans = stream.query(a_g, tau, h["n_miss"], n_h=h["n_mal"])
        rec["eval_risk"] = h["risk"]
        rec["eval_util"] = h["utility"]
        rec["gnc_ucb"] = 1.0 if ans.answer is None else ans.ucb
        rec["certified"] = bool(ans.passed_check)
        records.append(rec)
        return rec

    def consider(rec):
        nonlocal p_plus, p_minus
        if rec["certified"] and (
            p_plus is None or rec["train_util"] > p_plus["train_util"]
        ):
            p_plus = rec
        unknown = [r for r in records if not r["certified"]]
        p_minus = min(unknown, key=lambda r: r["train_risk"]) if unknown else None

    first = score(seed)
    seen.add(first["prompt"])
    consider(first)
    log(f"[CRISP] seed risk={first['train_risk']:.3f} "
        f"util={first['train_util']:.3f} certified={first['certified']}")

    for t in range(1, rounds + 1):
        if p_plus is None and p_minus is None:
            break
        if not stream.alive:
            log("[CRISP] holdout budget exhausted")
            break
        new = []
        if p_plus:
            new += _rewrite(p_plus, train, alpha, seen, rewrite_raise, "raise")
        if p_minus:
            new += _rewrite(p_minus, train, alpha, seen, rewrite_lower, "lower")
        for cand in new:
            rec = score(cand)
            consider(rec)
            log(f"[CRISP] round {t} risk={rec['train_risk']:.3f} "
                f"certified={rec['certified']}")

    certified = [r for r in records if r["certified"]]
    if not certified:
        log("[CRISP] nothing certified")
        return {"method": "CRISP", "selected_prompt": None,
                "n_pool": len(records), "n_certified": 0, "records": records}

    chosen = max(certified, key=lambda r: r["eval_util"])
    log(f"[CRISP] selected eval_util={chosen['eval_util']:.3f}")
    return {
        "method": "CRISP",
        "selected_prompt": chosen["prompt"],
        "eval_risk": chosen["eval_risk"],
        "eval_util": chosen["eval_util"],
        "gnc_ucb": chosen["gnc_ucb"],
        "n_pool": len(records),
        "n_certified": len(certified),
        "records": records,
    }
