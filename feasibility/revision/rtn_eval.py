"""Revision step 9: round-to-nearest (RTN) W4/W3 g128 fake-quantization of SmolLM2-135M, no calibration, deterministic.

Quantizer = GPTQ's symmetric min-max quantizer (gptq/quant.py Quantizer, sym=True, no MSE search), applied per output row
and per group of 128 consecutive input columns (last partial group as GPTQ: K=576 -> 4 x 128 + 1 x 64):
  xmax = max|w_group|; scale = 2*xmax / (2^bits - 1); zero = 2^(bits-1); q = clamp(round(w/scale) + zero, 0, 2^bits - 1);
  w_hat = scale * (q - zero).
Applied to every nn.Linear inside model.model.layers (q/k/v/o, gate/up/down); embeddings, norms and lm_head untouched
(GPTQ runs here used lm_head=False). Weights are fake-quantized in fp32, then the model is cast to bf16 and evaluated with
attn_implementation="eager" (the GPTQ checkpoints are evaluated in bf16 too, via BACKEND.TORCH dequantization).
Evals (cached ids only): WT-ppl (40 x 2048, fp 7da3b34bdebd1923), C4-ppl (256 x 2048, fp cbe443ebebeeceeb), and the six S4
protocols (context {512,1024,2048} x prepend-BOS {no,yes}, same 81,920 tokens; ctx2048/no-BOS == WT-ppl by construction).
usage: rtn_eval.py OUT.jsonl BITS     (lines already present are skipped)"""
import hashlib, json, os, resource, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
torch.set_num_threads(4)
import calib

out, bits = sys.argv[1], int(sys.argv[2])
cache = os.environ["FINAL_CACHE"]; M = "HuggingFaceTB/SmolLM2-135M"
done = {json.loads(l)["run_id"] for l in open(out) if json.loads(l).get("rc") == 0} if os.path.exists(out) else set()
evals = [("wt", None, None), ("c4", None, None)] + [("s4", L, b) for L in (512, 1024, 2048) for b in (0, 1)]
rid = lambda e: f"rtn_w{bits}_" + (e[0] if e[0] != "s4" else f"ctx{e[1]}_bos{e[2]}")
todo = [e for e in evals if rid(e) not in done]
if not todo: print("nothing to do"); sys.exit(0)

def rtn_(w, bits, gs=128):
    maxq = 2 ** bits - 1; zero = 2 ** (bits - 1)
    out = torch.empty_like(w)
    for c in range(0, w.shape[1], gs):
        g = w[:, c:c + gs]
        xmax = g.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        scale = 2 * xmax / maxq
        q = torch.clamp(torch.round(g / scale) + zero, 0, maxq)
        out[:, c:c + gs] = scale * (q - zero)
    return out

from transformers import AutoModelForCausalLM, AutoTokenizer
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32, attn_implementation="eager").eval()
n_lin, h = 0, hashlib.sha256()
with torch.no_grad():
    for name, mod in model.model.layers.named_modules():
        if isinstance(mod, torch.nn.Linear):
            mod.weight.copy_(rtn_(mod.weight.data, bits)); n_lin += 1
            h.update(name.encode()); h.update(mod.weight.data.numpy().tobytes())
model = model.to(torch.bfloat16)
quant_s = round(time.time() - t0, 1)
tok = AutoTokenizer.from_pretrained(M); BOS = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
test = np.load(os.path.join(cache, "wikitext2_test_smollm2.npy")).tolist()
wt = [test[i * 2048:(i + 1) * 2048] for i in range(40)]; assert calib.fingerprint(wt) == "7da3b34bdebd1923"
stream = [t for w in wt for t in w]
c4 = np.load(os.path.join(cache, "c4val_256x2048_s0.npy")).tolist(); assert calib.fingerprint(c4) == "cbe443ebebeeceeb"

@torch.no_grad()
def ppl(wins, bos=False):
    nll, n = 0.0, 0
    for w in wins:
        ids = torch.tensor([([BOS] if bos else []) + list(w)])
        lg = model(input_ids=ids).logits.float()[0]
        nll += torch.nn.functional.cross_entropy(lg[:-1], ids[0, 1:], reduction="sum").item(); n += ids.shape[1] - 1
    return float(np.exp(nll / n)), n

cpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout.splitlines()
g = lambda k: next((l.split(":", 1)[1].strip() for l in cpu if l.startswith(k)), None)
import transformers
for e in todo:
    t = time.time()
    if e[0] == "wt": v, n = ppl(wt); ctx, bos = 2048, 0
    elif e[0] == "c4": v, n = ppl(c4); ctx, bos = 2048, 0
    else:
        ctx, bos = e[1], e[2]; v, n = ppl([stream[i:i + ctx] for i in range(0, len(stream), ctx)], bool(bos))
    rec = {"run_id": rid(e), "rc": 0, "model": M, "method": "rtn", "bits": bits, "group_size": 128, "sym": True,
           "calibration": None, "seed": None, "eval_set": "c4" if e[0] == "c4" else "wikitext2", "context": ctx, "prepend_bos": bool(bos),
           "n_targets": n, "ppl": v, "eval_s": round(time.time() - t, 1), "fakequant_s": quant_s, "n_linear_quantized": n_lin,
           "weights_sha256": h.hexdigest(), "config": {"attn_implementation": "eager", "threads": 4, "eval_dtype": "bfloat16"},
           "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3), "cpu_model": g("Model name"),
           "cpu_flags": g("Flags"), "versions": {"torch": torch.__version__, "transformers": transformers.__version__}}
    open(out, "a").write(json.dumps(rec) + "\n"); print(rec["run_id"], v, n, rec["eval_s"], flush=True)
