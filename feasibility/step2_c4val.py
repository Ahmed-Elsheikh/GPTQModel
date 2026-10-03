"""C4 validation perplexity, GPTQ convention (IST-DASLab/gptq datautils.get_c4, the "c4" eval set):
en/c4-validation.00000-of-00008.json.gz; random.seed(0); 256 times: draw random docs until one has >= 2048 tokens,
then a random start in it (randint(0, len - 2048 - 1)); keep the 2048-token window.

usage:
  step2_c4val.py build CACHE.npy OUT.json            # tokenise + cache ids (256 x 2048 int32) with the main-venv tokenizer
  step2_c4val.py eval  CACHE.npy OUT.json dense      # dense SmolLM2-135M, fp32 (and bf16)
  step2_c4val.py eval  CACHE.npy OUT.json QDIR       # GPTQModel checkpoint, BACKEND.TORCH, bf16 (same reload as quant_eval.py)
"""
import json, os, random, resource, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
torch.set_num_threads(4)
import calib
from quant_eval import perplexity

M, N, L = "HuggingFaceTB/SmolLM2-135M", 256, 2048
mode, cache, out = sys.argv[1:4]
res = {"mode": mode, "cache": cache, "n": N, "L": L}
t0 = time.time()
if mode == "build":
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import tokenizers, transformers
    tok = AutoTokenizer.from_pretrained(M)
    val = load_dataset("allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"}, split="validation")
    random.seed(0)
    wins, tries, docs = [], 0, set()
    for _ in range(N):
        while True:
            i = random.randint(0, len(val) - 1); tries += 1
            ids = tok(val[i]["text"])["input_ids"]  # GPTQ calls tokenizer() with defaults; SmolLM2 adds no BOS
            if len(ids) >= L:
                break
        s = random.randint(0, len(ids) - L - 1)
        wins.append(ids[s:s + L]); docs.add(i)
    arr = np.array(wins, dtype=np.int32)
    np.save(cache, arr)
    res.update(n_val_docs=len(val), docs_drawn=tries, distinct_docs=len(docs), shape=list(arr.shape),
               fingerprint=calib.fingerprint(arr.tolist()), transformers=transformers.__version__,
               tokenizers=tokenizers.__version__)
else:
    target = sys.argv[4]
    wins = np.load(cache).tolist()
    res["fingerprint"] = calib.fingerprint(wins)
    if target == "dense":
        from transformers import AutoModelForCausalLM
        for dt in ("float32", "bfloat16"):
            t = time.time()
            m = AutoModelForCausalLM.from_pretrained(M, dtype=getattr(torch, dt)).eval()
            res[f"ppl_c4val_dense_{dt}"] = perplexity(m, wins); res[f"eval_{dt}_s"] = round(time.time() - t, 1)
            del m
            print(res, flush=True)
    else:
        from gptqmodel import GPTQModel, BACKEND
        t = time.time()
        qm = GPTQModel.load(target, device="cpu", backend=BACKEND.TORCH, dtype=torch.bfloat16)
        inner = qm.model
        res["qlinear_classes"] = sorted({type(x).__name__ for x in inner.modules() if "Linear" in type(x).__name__})
        res["load_s"] = round(time.time() - t, 1)
        t = time.time()
        res["ppl_c4val_quant"] = perplexity(inner, wins); res["eval_s"] = round(time.time() - t, 1)
    res["target"] = target
res["wall_s"] = round(time.time() - t0, 1)
res["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)
json.dump(res, open(out, "w"), indent=2)
print(json.dumps(res), flush=True)
