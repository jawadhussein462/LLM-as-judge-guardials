"""Two ways to pick a guardrail system prompt under a risk cap alpha.

CPS   Certified Prompt Search. Build a fixed pool, certify each prompt on
      the eval split with a Bonferroni Clopper--Pearson upper bound, and
      keep the certified prompt with the best utility lower bound.

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


def cps(train, eval_split, alpha=config.ALPHA, delta=config.DELTA,
        pool_size=config.CPS_POOL_SIZE, candidate_prompts=None, log=print):
    """Certify a fixed pool. Pass candidate_prompts to skip generation."""
    texts, labels = _xy(eval_split)
    if candidate_prompts:
        pool = []
        for p in candidate_prompts:
            if p and p not in pool:
                pool.append(p)
        pool = pool[:pool_size]
    else:
        pool = prompts.generate_pool(train, pool_size)

    n = len(pool)
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
        return {"method": "CPS", "selected_prompt": None, "fallback": True,
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
        "fallback": False,
        "n_pool": n,
        "n_certified": k,
        "records": records,
    }


def _rewrite(parent, train, alpha, seen, kind, n):
    prompt = parent["prompt"]
    risk = parent.get("train_risk", parent.get("risk", 0.0))
    out = []
    if kind == "raise":
        fp = _mine(prompt, train, 0, 1)
        tp = _mine(prompt, train, 1, 1) if fp else []
        rewrite = prompts.rewrite_raise_utility
        args = (fp, tp)
    else:
        fn = _mine(prompt, train, 1, 0)
        tn = _mine(prompt, train, 0, 0) if fn else []
        rewrite = prompts.rewrite_lower_risk
        args = (fn, tn)
    for _ in range(max(n * 2, n + 1)):
        if len(out) >= n:
            break
        cand = rewrite(prompt, *args, alpha, risk)
        if cand and cand not in seen and len(cand) > 30:
            seen.add(cand)
            out.append(cand)
    return out


def crisp(train, eval_split, alpha=config.ALPHA, delta=config.DELTA,
          rounds=config.CRISP_ROUNDS, m=config.CRISP_CANDIDATES_PER_ROUND,
          log=print):
    """Adaptive search. Certification uses Guess-and-Check on risk only."""
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

    seed = score(config.TASK_CFG["seed"])
    seen.add(seed["prompt"])
    consider(seed)
    log(f"[CRISP] seed risk={seed['train_risk']:.3f} "
        f"util={seed['train_util']:.3f} certified={seed['certified']}")

    for t in range(1, rounds + 1):
        if p_plus is None and p_minus is None:
            break
        if not stream.alive:
            log("[CRISP] holdout budget exhausted")
            break
        new = []
        if p_plus and p_minus:
            new += _rewrite(p_plus, train, alpha, seen, "raise", max(1, m // 2))
            new += _rewrite(p_minus, train, alpha, seen, "lower", max(1, m // 2))
        elif p_plus:
            new += _rewrite(p_plus, train, alpha, seen, "raise", max(1, m // 2))
        else:
            new += _rewrite(p_minus, train, alpha, seen, "lower", max(1, m // 2))
        for cand in new:
            rec = score(cand)
            consider(rec)
            log(f"[CRISP] round {t} risk={rec['train_risk']:.3f} "
                f"certified={rec['certified']}")

    certified = [r for r in records if r["certified"]]
    if not certified:
        log("[CRISP] nothing certified")
        return {"method": "CRISP", "selected_prompt": None, "fallback": True,
                "n_pool": len(records), "n_certified": 0, "records": records}

    chosen = max(certified, key=lambda r: r["eval_util"])
    log(f"[CRISP] selected eval_util={chosen['eval_util']:.3f}")
    return {
        "method": "CRISP",
        "selected_prompt": chosen["prompt"],
        "eval_risk": chosen["eval_risk"],
        "eval_util": chosen["eval_util"],
        "gnc_ucb": chosen["gnc_ucb"],
        "fallback": False,
        "n_pool": len(records),
        "n_certified": len(certified),
        "records": records,
    }
