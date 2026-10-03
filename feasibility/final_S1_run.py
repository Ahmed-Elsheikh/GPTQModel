"""S1 final runs: one run = (dense | GPTQ quantize) + WT-ppl [+ C4-ppl] [+ lm-eval], one JSON line appended to RESULTS.

Deterministic configuration (REPORT_step2.md §8, option 2 chosen by the user): attn_implementation="eager" for every
model load (quantization, quantized reload, dense), default thread count (torch.set_num_threads(4)).
Quantization path mirrors quant_eval.py exactly (seeding, prepare_dataset check, quantize(batch_size=1,
backend=AUTO), save, reload with BACKEND.TORCH in bf16) so the anchor reproduces step2_det.py DET_SETTING=eager.
Token ids come ONLY from the fingerprinted caches in $FINAL_CACHE:
  wikitext2_{train,test}_smollm2.npy (main-venv tokenizers 0.23.2; test windows fingerprint 7da3b34bdebd1923)
  c4val_256x2048_s0.npy (GPTQ convention, fingerprint cbe443ebebeeceeb)

usage: final_S1_run.py RESULTS.jsonl ITEMS.jsonl RUN_ID {dense|gptq} --bits B --seed S --n N --L L --evals wt,c4,lmeval
"""
import argparse, hashlib, glob, json, os, random, resource, shutil, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, torch

MODEL = "HuggingFaceTB/SmolLM2-135M"
WT_FP, C4_FP = "7da3b34bdebd1923", "cbe443ebebeeceeb"
ATTN = "eager"
T0 = time.time()


def peak_gb(): return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)


def env_info():
    lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout.splitlines()
    get = lambda k: next((l.split(":", 1)[1].strip() for l in lscpu if l.startswith(k)), None)
    import transformers, tokenizers, datasets, lm_eval, gptqmodel
    cfg = torch.__config__.show()
    pick = lambda key: next((l.strip(" -") for l in cfg.splitlines() if key in l), None)
    return {"cpu_model": get("Model name"), "cpu_flags": get("Flags"), "nproc": os.cpu_count(),
            "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "mkl": pick("Math Kernel Library"),
                         "onednn": pick("MKL-DNN"), "gptqmodel": getattr(gptqmodel, "__version__", None),
                         "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
                         "datasets": datasets.__version__, "lm_eval": lm_eval.__version__, "numpy": np.__version__}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results"); ap.add_argument("items"); ap.add_argument("run_id"); ap.add_argument("kind", choices=["dense", "gptq"])
    ap.add_argument("--bits", type=int, default=None); ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n", type=int, default=128); ap.add_argument("--L", type=int, default=2048)
    ap.add_argument("--evals", default="wt"); ap.add_argument("--lm_limit", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    cache = os.environ["FINAL_CACHE"]; work = os.environ.get("FINAL_WORK", cache)
    evals = a.evals.split(",")
    torch.set_num_threads(a.threads)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    import calib
    from quant_eval import perplexity

    rec = {"run_id": a.run_id, "model": MODEL, "kind": a.kind,
           "method": "dense" if a.kind == "dense" else "gptq", "bits": a.bits, "sparsity": None,
           "group_size": a.group_size if a.kind == "gptq" else None,
           "calibration": None, "seed": a.seed if a.kind == "gptq" else None,
           "config": {"attn_implementation": ATTN, "torch_threads": a.threads, "quant_dtype": "float32",
                      "quant_backend": "auto", "eval_backend": "BACKEND.TORCH", "eval_dtype_quant": "bfloat16",
                      "eval_dtype_dense": "float32", "batch_size_quant": 1},
           **env_info(), "phases": {}, "metrics": {}}
    def phase(k, t): rec["phases"][k] = {"wall_s": round(time.time() - t, 2), "peak_rss_gb": peak_gb()}; print(k, rec["phases"][k], flush=True)

    t = time.time()
    test = np.load(os.path.join(cache, "wikitext2_test_smollm2.npy")).tolist()
    wt_wins = [test[i * 2048:(i + 1) * 2048] for i in range(40)]
    rec["wt_eval_fingerprint"] = calib.fingerprint(wt_wins); assert rec["wt_eval_fingerprint"] == WT_FP, rec["wt_eval_fingerprint"]
    if "c4" in evals:
        c4_wins = np.load(os.path.join(cache, "c4val_256x2048_s0.npy")).tolist()
        rec["c4_eval_fingerprint"] = calib.fingerprint(c4_wins); assert rec["c4_eval_fingerprint"] == C4_FP, rec["c4_eval_fingerprint"]
    if a.kind == "gptq":
        train = np.load(os.path.join(cache, "wikitext2_train_smollm2.npy")).tolist()
        wins = calib._windows_from_stream(train, a.n, a.L, random.Random(a.seed))  # == calib.draw_windows("wikitext2", ...)
        samples = [{"input_ids": w, "attention_mask": [1] * a.L} for w in wins]
        rec["calibration"] = {"source": "wikitext2-raw-v1 train (cached ids)", "n": a.n, "L": a.L,
                              "fingerprint": calib.fingerprint(samples), "rng": "random.Random(seed).randint"}
    phase("data", t)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL)
    if a.kind == "dense":
        t = time.time()
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation=ATTN).eval()
        phase("load", t)
    else:
        from gptqmodel import GPTQModel, GPTQConfig, BACKEND
        t = time.time()
        qcfg = GPTQConfig(bits=a.bits, group_size=a.group_size, device="cpu")
        rec["quant_config"] = repr(qcfg)[:2000]
        qmodel = GPTQModel.load(MODEL, qcfg, dtype=torch.float32, attn_implementation=ATTN)
        rec["attn_at_quant"] = qmodel.model.config._attn_implementation
        phase("load_for_quant", t)
        prepared = qmodel.prepare_dataset(samples, calibration_dataset_sort="desc", batch_size=1)
        flat = [row for b in prepared for row in (b["input_ids"].tolist() if torch.is_tensor(b["input_ids"]) else b["input_ids"])]
        rec["calib_check"] = {"passed": len(samples), "after_prepare": len(flat), "lengths": sorted({len(r) for r in flat}),
                              "same_multiset": sorted(map(tuple, flat)) == sorted(tuple(s["input_ids"]) for s in samples)}
        t = time.time()
        qmodel.quantize(samples, batch_size=1, backend=BACKEND("auto"))
        phase("quantize", t)
        save_dir = os.path.join(work, "ckpt_" + a.run_id)
        shutil.rmtree(save_dir, ignore_errors=True)
        t = time.time(); qmodel.save(save_dir); del qmodel; phase("save", t)
        h = hashlib.sha256()
        for f in sorted(glob.glob(os.path.join(save_dir, "*.safetensors"))): h.update(open(f, "rb").read())
        rec["ckpt_sha256"] = h.hexdigest()
        t = time.time()
        qm = GPTQModel.load(save_dir, device="cpu", backend=BACKEND.TORCH, dtype=torch.bfloat16, attn_implementation=ATTN)
        model = qm.model
        rec["qlinear_classes"] = sorted({type(m).__name__ for m in model.modules() if "Linear" in type(m).__name__})
        phase("load_quantized", t)
    rec["attn_at_eval"] = model.config._attn_implementation
    rec["eval_dtype"] = str(next(model.parameters()).dtype)

    t = time.time(); rec["metrics"]["wt_ppl"] = perplexity(model, wt_wins); phase("eval_wt", t)
    print("wt_ppl", repr(rec["metrics"]["wt_ppl"]), flush=True)
    if "c4" in evals:
        t = time.time(); rec["metrics"]["c4_ppl"] = perplexity(model, c4_wins); phase("eval_c4", t)
    if "lmeval" in evals:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        t = time.time()
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=8, device="cpu")
        res = lm_eval.simple_evaluate(model=lm, tasks=["arc_easy", "hellaswag"], num_fewshot=0, limit=a.lm_limit,
                                      log_samples=True, random_seed=0, numpy_random_seed=0, torch_random_seed=0,
                                      fewshot_random_seed=0)
        phase("eval_lmeval", t)
        rec["lmeval"] = {"tasks": ["arc_easy", "hellaswag"], "limit": a.lm_limit, "num_fewshot": 0, "batch_size": 8,
                         "results": res["results"], "n_samples": {k: len(v) for k, v in res["samples"].items()}}
        for task, d in res["results"].items():
            for m in ("acc,none", "acc_norm,none"):
                if m in d: rec["metrics"][f"{task}_{m.split(',')[0]}"] = d[m]
        items = {"run_id": a.run_id}
        for task, ss in res["samples"].items():
            ss = sorted(ss, key=lambda s: s["doc_id"])
            items[task] = {"doc_id": [s["doc_id"] for s in ss], "acc": [s.get("acc") for s in ss],
                           "acc_norm": [s.get("acc_norm") for s in ss]}
        with open(a.items, "a") as f: f.write(json.dumps(items) + "\n")
    if a.kind == "gptq":
        shutil.rmtree(save_dir, ignore_errors=True)
    rec["rc"] = 0; rec["total_wall_s"] = round(time.time() - T0, 1); rec["peak_rss_gb"] = peak_gb()
    with open(a.results, "a") as f: f.write(json.dumps(rec, default=str) + "\n")
    print("DONE", json.dumps(rec["metrics"]), flush=True)


if __name__ == "__main__":
    main()
