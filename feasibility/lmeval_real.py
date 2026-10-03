"""Real-task lm-eval runs (lm_eval 0.4.13) for the real-data rerun.

usage: lmeval_real.py {dense|quant} MODEL_OR_QDIR OUT.json TASKS LIMIT BATCH [MAX_LENGTH]
dense: HFLM loads the HF model id in float32.
quant: GPTQModel.load(BACKEND.TORCH, bf16) and the in-memory model is handed to HFLM (the CLI gptqmodel=True path
       cannot select a kernel that accepts SmolLM2's K=576; see REPORT.md 6.7).
"""
import json, sys, time, resource
import torch
torch.set_num_threads(4)
import lm_eval
from lm_eval.models.huggingface import HFLM

mode, src, out, tasks, limit, bs = sys.argv[1:7]
maxlen = int(sys.argv[7]) if len(sys.argv) > 7 else None
limit = None if limit in ("none", "0") else int(limit)
extra = {} if maxlen is None else {"max_length": maxlen}
t = time.time()
if mode == "dense":
    lm = HFLM(pretrained=src, dtype="float32", batch_size=int(bs), device="cpu", **extra)
else:
    from gptqmodel import GPTQModel, BACKEND
    qm = GPTQModel.load(src, device="cpu", backend=BACKEND.TORCH, dtype=torch.bfloat16)
    lm = HFLM(pretrained=qm.model, tokenizer=qm.tokenizer, batch_size=int(bs), device="cpu", **extra)
t_load = time.time() - t
res = lm_eval.simple_evaluate(model=lm, tasks=tasks.split(","), num_fewshot=0, limit=limit,
                              random_seed=0, numpy_random_seed=0, torch_random_seed=0, fewshot_random_seed=0)
summary = {"mode": mode, "model": src, "tasks": tasks, "limit": limit, "batch_size": int(bs), "max_length": maxlen,
           "load_s": round(t_load, 1), "wall_s": round(time.time() - t, 1),
           "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
           "results": res["results"]}
json.dump(summary, open(out, "w"), indent=2, default=str)
print(json.dumps(summary, indent=1, default=str))
