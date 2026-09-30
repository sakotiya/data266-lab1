"""Yelp polarity data pipeline: EDA, cleaning, tokenisation, vocabulary, slices.

Constraint (workplan 2.1): embeddings are learned from scratch later in models.py.
Nothing here loads a pretrained vector table or language model.

Negation policy: NLTK's English stopword list contains 39 words that INVERT
sentiment ("not", "no", "nor", and every "n't" contraction form). Removing them
turns "the food was not good" into "food good". Contractions are therefore
expanded BEFORE stopword removal, and the negation words are kept.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

NEGATIONS = {"not", "no", "nor", "never", "none", "neither", "cannot", "without"}
PAD, UNK = "<pad>", "<unk>"


def get_stopwords() -> set:
    """NLTK English stopwords minus every sentiment-inverting word."""
    from nltk.corpus import stopwords

    sw = set(stopwords.words("english"))
    keep = {w for w in sw if w in NEGATIONS or w.endswith("n't")
            or w in {"ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn",
                     "haven", "isn", "mightn", "mustn", "needn", "shan",
                     "shouldn", "wasn", "weren", "won", "wouldn", "don"}}
    return sw - keep


def slice_features(raw_text: str) -> dict:
    """Robustness-slice features, computed on the RAW review before cleaning -
    punctuation counts do not survive preprocessing."""
    low = raw_text.lower()
    return {
        "exclamation_count": raw_text.count("!"),
        "has_negation": int(bool(re.search(r"\b(not|no|never|n't)\b", low))),
        "raw_len_chars": len(raw_text),
    }


def clean_text(text: str, pp: dict, stop: set, lemmatizer=None) -> list:
    """lowercase -> expand contractions -> strip punctuation/specials ->
    stopword removal -> lemmatise -> tokens."""
    t = text.lower() if pp["lowercase"] else text
    if pp["normalize_contractions"]:
        t = t.replace("won't", "will not").replace("can't", "can not")
        t = re.sub(r"n't\b", " not", t)
        for k, v in (("'re", " are"), ("'s", " is"), ("'d", " would"),
                     ("'ll", " will"), ("'ve", " have"), ("'m", " am")):
            t = t.replace(k, v)
    t = t.replace("\\n", " ").replace("\\\"", " ")
    if pp["strip_punctuation"] or pp["strip_special_chars"]:
        t = re.sub(r"[^a-z0-9\s]", " ", t)
    toks = t.split()
    if stop:
        toks = [w for w in toks if w not in stop]
    if pp["lemmatize"] and lemmatizer is not None:
        toks = [lemmatizer.lemmatize(w) for w in toks]
    return toks


def is_malformed(text: str, label) -> bool:
    return (text is None or label is None or not str(text).strip()
            or len(str(text).strip()) < 3 or label not in (0, 1))


def build_dataset(cfg: dict, log=None) -> dict:
    """Load, EDA, clean, split, index. Validation is held out of TRAIN; the
    official test split is only touched at final evaluation."""
    from datasets import load_dataset
    from nltk.stem import WordNetLemmatizer

    d, pp = cfg["data"], cfg["preprocess"]
    rng = np.random.default_rng(cfg["run"]["seed"])

    raw_train = load_dataset(d["dataset"], split="train")
    raw_test = load_dataset(d["dataset"], split="test")
    if d["train_size"]:
        idx = rng.permutation(len(raw_train))[:d["train_size"]]
        raw_train = raw_train.select(idx.tolist())
    if d["test_size"]:
        raw_test = raw_test.select(range(min(d["test_size"], len(raw_test))))

    stop = get_stopwords()
    lem = WordNetLemmatizer() if pp["lemmatize"] else None

    def prepare(ds, name):
        texts, labels, feats, raws, dropped = [], [], [], [], 0
        for row in ds:
            text, label = row["text"], row["label"]
            if is_malformed(text, label):
                dropped += 1
                continue
            toks = clean_text(text, pp, stop, lem)
            if not toks:                      # empty after cleaning
                dropped += 1
                continue
            texts.append(toks)
            labels.append(int(label))
            raws.append(text)
            f = slice_features(text)
            f["len_tokens"] = len(toks)
            feats.append(f)
        if log:
            log.event("prepare", split=name, kept=len(texts), dropped=dropped)
        return texts, np.array(labels), feats, raws

    tr_toks, tr_y, tr_f, _ = prepare(raw_train, "train_full")
    te_toks, te_y, te_f, te_raw = prepare(raw_test, "test")

    # validation carved out of TRAIN, stratified by label
    n_val = int(len(tr_toks) * d["val_fraction"])
    order = rng.permutation(len(tr_toks))
    val_idx, train_idx = set(order[:n_val].tolist()), order[n_val:]
    va_toks = [tr_toks[i] for i in sorted(val_idx)]
    va_y = tr_y[sorted(val_idx)]
    va_f = [tr_f[i] for i in sorted(val_idx)]
    tr2_toks = [tr_toks[i] for i in train_idx]
    tr2_y = tr_y[train_idx]
    tr2_f = [tr_f[i] for i in train_idx]

    # vocabulary from the TRAINING split only
    counts = Counter(w for toks in tr2_toks for w in toks)
    vocab_words = [w for w, c in counts.most_common(pp["max_vocab"] - 2)
                   if c >= pp["min_token_freq"]]
    stoi = {PAD: 0, UNK: 1}
    stoi.update({w: i + 2 for i, w in enumerate(vocab_words)})

    def encode(toks_list):
        L = pp["max_len"]
        out = np.zeros((len(toks_list), L), dtype=np.int64)
        for i, toks in enumerate(toks_list):
            ids = [stoi.get(w, 1) for w in toks[:L]]
            out[i, :len(ids)] = ids
        return out

    Xtr, Xva, Xte = encode(tr2_toks), encode(va_toks), encode(te_toks)

    eda = {
        "train_rows": len(tr2_toks), "val_rows": len(va_toks), "test_rows": len(te_toks),
        "train_pos_frac": float(tr2_y.mean()), "test_pos_frac": float(te_y.mean()),
        "vocab_size": len(stoi),
        "token_len_mean": float(np.mean([f["len_tokens"] for f in tr2_f])),
        "token_len_median": float(np.median([f["len_tokens"] for f in tr2_f])),
        "token_len_p95": float(np.percentile([f["len_tokens"] for f in tr2_f], 95)),
        "token_len_max": int(np.max([f["len_tokens"] for f in tr2_f])),
        "truncated_frac": float(np.mean([f["len_tokens"] > pp["max_len"] for f in tr2_f])),
        "stopwords_removed": len(stop),
        "negation_words_kept": sorted(NEGATIONS),
        "oov_rate_test": float(np.mean(Xte[Xte > 0] == 1)),
    }
    return {
        "stoi": stoi, "eda": eda,
        "train": (Xtr, tr2_y), "val": (Xva, va_y), "test": (Xte, te_y),
        "feats": {"val": va_f, "test": te_f},
        "test_texts": te_raw,   # raw reviews, in the order the arrays hold them
    }


def compute_slices(feats: list, rules: list) -> dict:
    """Boolean mask per configured robustness slice."""
    masks = {}
    for r in rules:
        name = r["name"]
        if name == "short_reviews":
            m = [f["len_tokens"] <= 50 for f in feats]
        elif name == "long_reviews":
            m = [f["len_tokens"] > 200 for f in feats]
        elif name == "contains_negation":
            m = [bool(f["has_negation"]) for f in feats]
        elif name == "high_punctuation":
            m = [f["exclamation_count"] >= 3 for f in feats]
        else:
            raise ValueError(f"unknown slice rule: {name}")
        masks[name] = np.array(m)
    return masks


def save_processed(bundle: dict, cache_dir: str) -> None:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        X, y = bundle[split]
        np.savez_compressed(cache / f"{split}.npz", X=X, y=y)
    (cache / "vocab.json").write_text(json.dumps(bundle["stoi"]))
    (cache / "eda.json").write_text(json.dumps(bundle["eda"], indent=2))
    (cache / "feats.json").write_text(json.dumps(bundle["feats"]))
    (cache / "test_texts.json").write_text(json.dumps(bundle["test_texts"]))
