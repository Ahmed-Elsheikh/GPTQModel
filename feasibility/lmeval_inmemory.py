"""Test 2b fallback: load a GPTQ checkpoint with GPTQModel (explicit BACKEND.TORCH, since AUTO picks an int4pack
kernel that rejects K=576) and hand the in-memory HF model object to lm-eval's HFLM."""
import json, sys, time, resource, os
import torch
torch.set_num_threads(4)
from gptqmodel import GPTQModel, BACKEND
import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

qdir, out = sys.argv[1], sys.argv[2]
t = time.time()
qm = GPTQModel.load(qdir, device="cpu", backend=BACKEND.TORCH)
lm = HFLM(pretrained=qm.model, tokenizer=qm.tokenizer, batch_size=8, device="cpu")
t_load = time.time() - t
tm = TaskManager(include_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "lm_eval_tasks"))
res = lm_eval.simple_evaluate(model=lm, tasks=["synth_arc_easy", "synth_hellaswag"], num_fewshot=0, limit=200,
                              task_manager=tm, random_seed=0, numpy_random_seed=0, torch_random_seed=0, fewshot_random_seed=0)
summary = {"load_s": round(t_load, 1), "wall_s": round(time.time() - t, 1),
           "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
           "results": res["results"]}
json.dump(summary, open(out, "w"), indent=2, default=str)
print(json.dumps(summary, indent=1, default=str))
