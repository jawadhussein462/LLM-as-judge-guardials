"""Load the four guardrail datasets and cut a train / eval / test split.

  y = 1  violation, should be BLOCKED
  y = 0  benign, should be ALLOWED

  deepset, bipia   prompt injection (user message)
  spl              system-prompt leakage (assistant reply)
  sid              sensitive-information disclosure (assistant reply)

Each file is local parquet with columns text, label.
"""
import os
import random

import pandas as pd

import config

# name -> parquet pieces under data/
DATASETS = {
    "deepset": ["deepset_train.parquet", "deepset_test.parquet"],
    "bipia": ["bipia_train.parquet", "bipia_test.parquet"],
    "spl": ["spl_all.parquet"],
    "sid": ["sid_all.parquet"],
}

TASK_FOR = {"deepset": "pi", "bipia": "pi", "spl": "spl", "sid": "sid"}


def load_frame(name):
    frames = []
    for fname in DATASETS[name]:
        path = os.path.join(config.DATA_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df = df[df["text"].str.len() < config.MAX_TEXT_CHARS]
    return df.drop_duplicates(subset=["text"]).reset_index(drop=True)


def load_splits(name=None, seed=None):
    """Return {"train", "eval", "test"} lists of (text, label)."""
    name = name or config.DATASET
    seed = config.SEED if seed is None else seed
    df = load_frame(name)
    rng = random.Random(seed)

    idx = []
    for label in (0, 1):
        ids = list(df[df["label"] == label].index)
        rng.shuffle(ids)
        if config.MAX_PER_CLASS:
            ids = ids[: config.MAX_PER_CLASS]
        idx.extend(ids)
    rng.shuffle(idx)

    n = len(idx)
    n_tr = int(round(config.SPLIT["train"] * n))
    n_ev = int(round(config.SPLIT["eval"] * n))
    parts = {
        "train": idx[:n_tr],
        "eval": idx[n_tr:n_tr + n_ev],
        "test": idx[n_tr + n_ev:],
    }
    splits = {}
    for split_name, ids in parts.items():
        rows = [(df.at[i, "text"], int(df.at[i, "label"])) for i in ids]
        rng.shuffle(rows)
        splits[split_name] = rows
    return splits


def counts(split):
    labels = [y for _, y in split]
    n_mal = sum(labels)
    return {"n": len(labels), "n_mal": n_mal, "n_ben": len(labels) - n_mal}


if __name__ == "__main__":
    for name in DATASETS:
        splits = load_splits(name)
        print(name, {k: counts(v) for k, v in splits.items()})
