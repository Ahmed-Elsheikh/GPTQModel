"""Calibration windows for EXP1/EXP2.

draw_windows(source, tokenizer, n, L, seed) -> list[{"input_ids": [...L ints], "attention_mask": [1]*L}]
This is the pre-tokenised dict format accepted by GPTQModel.quantize(); with L > 10 and
L <= max_position_embeddings GPTQModel neither drops nor trims any sample (see
gptqmodel/utils/calibration.py: calibration_data_min_length=10 filter, max_position trim).

Sources
  wikitext2 : HF datasets Salesforce/wikitext wikitext-2-raw-v1 train, docs joined with "\n\n", tokenised once,
              window starts ~ random.Random(seed).randint(0, T-L-1)                (GPTQ-paper style)
  c4        : allenai/c4 en/c4-train.00000-of-01024.json.gz; sample a doc index with the same RNG
              until a doc has > L tokens, then a window start inside it                 (GPTQ-paper style)
  synthetic : OFFLINE STAND-IN used in this feasibility study because huggingface.co is blocked.
              A fixed Zipf(1.1) token stream (corpus_seed fixed), same window-start logic as wikitext2.
eval_windows(source, tokenizer, n=40, L=2048) -> first n non-overlapping windows of the *test* stream.
"""
import hashlib, json, os, random
import numpy as np

SYN_TRAIN_SEED, SYN_TEST_SEED, SYN_TOKENS = 1234, 5678, 400_000


def _synthetic_stream(vocab, corpus_seed, n_tokens=SYN_TOKENS):
    rng = np.random.default_rng(corpus_seed)
    ranks = rng.zipf(1.1, size=n_tokens * 2)
    ranks = ranks[ranks < vocab][:n_tokens]
    perm = np.random.default_rng(0).permutation(vocab)  # fixed rank->id map shared by train/test
    return perm[ranks].tolist()


TOKEN_CACHE_DIR = os.environ.get("TOKEN_CACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_tok"))


def _cache_path(tokenizer, split):
    name = getattr(tokenizer, "name_or_path", "tok").replace("/", "--")
    return os.path.join(TOKEN_CACHE_DIR, f"wikitext2_{split}_{name}.npy")


def _wikitext_stream(tokenizer, split):
    """Token ids of the joined split, cached as .npy (int32) so every venv uses identical ids.
    The cache is filled by the main venv (tokenizers 0.23.x); other venvs (wanda: tokenizers 0.21.4, which
    tokenises WikiText-2 differently) must only read it."""
    p = _cache_path(tokenizer, split)
    if os.path.exists(p):
        return np.load(p).tolist()
    ids = _wikitext_stream_uncached(tokenizer, split)
    os.makedirs(TOKEN_CACHE_DIR, exist_ok=True)
    np.save(p, np.asarray(ids, dtype=np.int32))
    return ids


def cache_info():
    """sha256 (16 hex) of every cached stream file, for provenance in result JSONs."""
    out = {}
    if os.path.isdir(TOKEN_CACHE_DIR):
        for f in sorted(os.listdir(TOKEN_CACHE_DIR)):
            if f.endswith(".npy"):
                out[f] = hashlib.sha256(open(os.path.join(TOKEN_CACHE_DIR, f), "rb").read()).hexdigest()[:16]
    return out


def _wikitext_stream_uncached(tokenizer, split):
    from datasets import load_dataset
    # "Salesforce/wikitext" is the canonical Hub id (the bare "wikitext" alias redirects to it)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    return tokenizer("\n\n".join(ds["text"]), add_special_tokens=False)["input_ids"]


def _windows_from_stream(stream, n, L, rng):
    out = []
    for _ in range(n):
        s = rng.randint(0, len(stream) - L - 1)
        out.append(stream[s:s + L])
    return out


def draw_windows(source, tokenizer, n=128, L=512, seed=0, vocab=None):
    rng = random.Random(seed)
    if source == "wikitext2":
        wins = _windows_from_stream(_wikitext_stream(tokenizer, "train"), n, L, rng)
    elif source == "c4":
        from datasets import load_dataset
        ds = load_dataset("allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train")
        wins = []
        while len(wins) < n:
            ids = tokenizer(ds[rng.randint(0, len(ds) - 1)]["text"], add_special_tokens=False)["input_ids"]
            if len(ids) > L:
                s = rng.randint(0, len(ids) - L - 1)
                wins.append(ids[s:s + L])
    elif source == "synthetic":
        wins = _windows_from_stream(_synthetic_stream(vocab or len(tokenizer), SYN_TRAIN_SEED), n, L, rng)
    else:
        raise ValueError(source)
    assert len(wins) == n and all(len(w) == L for w in wins)
    return [{"input_ids": w, "attention_mask": [1] * L} for w in wins]


def eval_windows(source, tokenizer, n=40, L=2048, vocab=None):
    if source == "wikitext2":
        stream = _wikitext_stream(tokenizer, "test")
    elif source == "synthetic":
        stream = _synthetic_stream(vocab or len(tokenizer), SYN_TEST_SEED)
    else:
        raise ValueError(source)
    assert len(stream) >= n * L, (len(stream), n * L)
    return [stream[i * L:(i + 1) * L] for i in range(n)]


def fingerprint(samples):
    h = hashlib.sha256()
    for s in samples:
        ids = s["input_ids"] if isinstance(s, dict) else s
        h.update(json.dumps(ids).encode())
    return h.hexdigest()[:16]
