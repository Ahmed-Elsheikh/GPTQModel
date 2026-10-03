"""S3: tokenize once and cache calibration windows (128 x 512, seeds 0-2) for both sources, with fingerprints.
wikitext2: windows drawn with random.Random(seed) from the cached main-venv WikiText-2 train stream (wikitext2_train_smollm2.npy)
c4:        calib.draw_windows("c4", ...) = GPTQ-paper style sampling from allenai/c4 en/c4-train.00000-of-01024.json.gz
Writes $FINAL_CACHE/s3_calib_{source}_n128_L512_s{seed}.npy and appends one JSON line per file to RESULTS."""
import json, os, random, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, calib, transformers, tokenizers, datasets
from transformers import AutoTokenizer
cache, out = os.environ["FINAL_CACHE"], sys.argv[1]
M, N, L = "HuggingFaceTB/SmolLM2-135M", 128, 512
tok = AutoTokenizer.from_pretrained(M)
train = np.load(os.path.join(cache, "wikitext2_train_smollm2.npy")).tolist()
for src in ("wikitext2", "c4"):
    for seed in (0, 1, 2):
        t = time.time()
        if src == "wikitext2":
            wins = calib._windows_from_stream(train, N, L, random.Random(seed))
            samples = [{"input_ids": w} for w in wins]
        else:
            samples = calib.draw_windows("c4", tok, n=N, L=L, seed=seed)
        arr = np.array([s["input_ids"] for s in samples], dtype=np.int32)
        f = os.path.join(cache, f"s3_calib_{src}_n{N}_L{L}_s{seed}.npy"); np.save(f, arr)
        rec = {"run_id": f"cache_{src}_s{seed}", "kind": "calib_cache", "source": src, "seed": seed, "n": N, "L": L,
               "file": os.path.basename(f), "shape": list(arr.shape), "fingerprint": calib.fingerprint(arr.tolist()),
               "wall_s": round(time.time() - t, 1), "tokenizers": tokenizers.__version__,
               "transformers": transformers.__version__, "datasets": datasets.__version__,
               "c4_file": "en/c4-train.00000-of-01024.json.gz" if src == "c4" else None, "rc": 0}
        open(out, "a").write(json.dumps(rec) + "\n"); print(rec, flush=True)
