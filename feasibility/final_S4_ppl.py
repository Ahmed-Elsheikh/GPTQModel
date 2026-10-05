"""S4 (trimmed): WT-ppl of SmolLM2-135M at context {512, 1024, 2048} x prepend-BOS {no, yes}, for dense fp32 and
GPTQ W4 g128 seed 0 (128x512 WikiText calibration), both with attn_implementation="eager", 4 threads.

Token ids come only from the fingerprinted cache ($FINAL_CACHE/wikitext2_{train,test}_smollm2.npy).
Every context scores the SAME 81,920 test tokens (= the 40 x 2048 Step 1 windows, fingerprint 7da3b34bdebd1923):
160 x 512, 80 x 1024 or 40 x 2048 non-overlapping windows. Without BOS the first token of each window is not
predicted (L-1 targets per window, as in quant_eval.perplexity); with BOS the window is [BOS] + w and all L tokens are
targets. Token counts are recorded. One JSON line per (model, context, bos); lines already present are skipped.
usage: final_S4_ppl.py RESULTS.jsonl {dense|w4}"""
import glob, hashlib, json, os, random, resource, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
torch.set_num_threads(4)
import calib

MODEL, ATTN = "HuggingFaceTB/SmolLM2-135M", "eager"
out, which = sys.argv[1], sys.argv[2]
cache, work = os.environ["FINAL_CACHE"], os.environ["FINAL_WORK"]
done = set()
if os.path.exists(out):
    for l in open(out):
        r = json.loads(l)
        if r.get("rc") == 0: done.add(r["run_id"])
todo = [(L, b) for L in (512, 1024, 2048) for b in (False, True) if f"{which}_ctx{L}_bos{int(b)}" not in done]
if not todo:
    print("nothing to do"); sys.exit(0)

def env():
    cpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout.splitlines()
    g = lambda k: next((l.split(":", 1)[1].strip() for l in cpu if l.startswith(k)), None)
    import transformers, gptqmodel, tokenizers
    return {"cpu_model": g("Model name"), "cpu_flags": g("Flags"),
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                         "gptqmodel": getattr(gptqmodel, "__version__", None), "tokenizers": tokenizers.__version__}}
ENV = env()
test = np.load(os.path.join(cache, "wikitext2_test_smollm2.npy")).tolist()
base = [test[i * 2048:(i + 1) * 2048] for i in range(40)]
assert calib.fingerprint(base) == "7da3b34bdebd1923"
stream = [t for w in base for t in w]  # 81,920 tokens

from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(MODEL)
BOS = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
meta = {}
t = time.time()
if which == "dense":
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation=ATTN).eval()
    meta.update(kind="dense", eval_dtype="float32")
else:
    from gptqmodel import GPTQModel, GPTQConfig, BACKEND
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    ck = os.path.join(work, "ckpt_s4_w4_s0")
    if not glob.glob(os.path.join(ck, "*.safetensors")):
        train = np.load(os.path.join(cache, "wikitext2_train_smollm2.npy")).tolist()
        wins = calib._windows_from_stream(train, 128, 512, random.Random(0))
        samples = [{"input_ids": w, "attention_mask": [1] * 512} for w in wins]
        meta["calib_fingerprint"] = calib.fingerprint(samples)
        qm = GPTQModel.load(MODEL, GPTQConfig(bits=4, group_size=128, device="cpu"), dtype=torch.float32, attn_implementation=ATTN)
        qm.prepare_dataset(samples, calibration_dataset_sort="desc", batch_size=1)  # same call order as quant_eval.py
        tq = time.time(); qm.quantize(samples, batch_size=1, backend=BACKEND("auto")); meta["quantize_s"] = round(time.time() - tq, 1)
        qm.save(ck); del qm
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(ck, "*.safetensors"))): h.update(open(f, "rb").read())
    meta["ckpt_sha256"] = h.hexdigest()
    model = GPTQModel.load(ck, device="cpu", backend=BACKEND.TORCH, dtype=torch.bfloat16, attn_implementation=ATTN).model
    meta.update(kind="gptq", bits=4, group_size=128, seed=0, calibration={"source": "wikitext2", "n": 128, "L": 512},
                eval_dtype="bfloat16", eval_backend="BACKEND.TORCH")
meta["load_s"] = round(time.time() - t, 1)
meta["attn"] = model.config._attn_implementation

@torch.no_grad()
def ppl(L, bos):
    nll, n = 0.0, 0
    for i in range(0, len(stream), L):
        w = stream[i:i + L]
        ids = torch.tensor([([BOS] if bos else []) + w])
        logits = model(input_ids=ids).logits.float()[0]
        tgt = ids[0, 1:]
        nll += torch.nn.functional.cross_entropy(logits[:-1], tgt, reduction="sum").item(); n += tgt.numel()
    return float(np.exp(nll / n)), n

for L, b in todo:
    t = time.time(); v, n = ppl(L, b)
    rec = {"run_id": f"{which}_ctx{L}_bos{int(b)}", "rc": 0, "model": MODEL, "method": "dense" if which == "dense" else "gptq",
           "context": L, "prepend_bos": b, "bos_id": BOS, "n_windows": len(stream) // L, "n_targets": n, "wt_ppl": v,
           "eval_s": round(time.time() - t, 1), "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3),
           "config": {"attn_implementation": ATTN, "threads": 4}, **meta, **ENV}
    open(out, "a").write(json.dumps(rec) + "\n"); print(rec["run_id"], v, n, rec["eval_s"], flush=True)
