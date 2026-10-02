"""Test 1: quantize with GPTQModel on CPU, then evaluate fixed-budget perplexity.

Writes a JSON with per-phase wall-clock times, peak RSS and perplexities.
"""
import argparse, json, os, random, resource, shutil, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, psutil, torch

T0 = time.time()
def stamp(): return round(time.time() - T0, 2)
def rss_gb(): return round(psutil.Process().memory_info().rss / 2**30, 3)
def peak_gb(): return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)


@torch.no_grad()
def perplexity(model, windows, device="cpu"):
    nll, ntok = 0.0, 0
    for w in windows:
        ids = torch.tensor([w], device=device)
        logits = model(input_ids=ids).logits.float()
        loss = torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="sum")
        nll += loss.item(); ntok += ids.shape[1] - 1
    return float(np.exp(nll / ntok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", default="gptq", choices=["gptq", "awq"])
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--L", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", default="synthetic")
    ap.add_argument("--eval_n", type=int, default=40)
    ap.add_argument("--eval_L", type=int, default=2048)
    ap.add_argument("--eval_dense", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_dir", default=None)
    ap.add_argument("--backend", default="auto", help="GPTQModel BACKEND for the post-quant reload (e.g. auto, torch, torch_fused)")
    ap.add_argument("--eval_only", default=None, help="skip quantization; load this saved quantized dir and evaluate")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import calib

    res = {"args": vars(a), "phases": {}}
    def phase(name, t_start):
        res["phases"][name] = {"wall_s": round(time.time() - t_start, 2), "rss_gb": rss_gb(), "peak_rss_gb": peak_gb()}
        print(f"[{stamp()}s] {name}: {res['phases'][name]}", flush=True)
        json.dump(res, open(a.out, "w"), indent=2)

    t = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    from transformers import AutoConfig
    vocab = AutoConfig.from_pretrained(a.model).vocab_size
    samples = calib.draw_windows(a.source, tok, n=a.n, L=a.L, seed=a.seed, vocab=vocab)
    evalw = calib.eval_windows(a.source, tok, n=a.eval_n, L=a.eval_L, vocab=vocab)
    res["calib_fingerprint"] = calib.fingerprint(samples)
    res["eval_fingerprint"] = calib.fingerprint(evalw)
    phase("data", t)

    if a.eval_only:
        return eval_quantized(a, res, phase, evalw, a.eval_only)

    if a.eval_dense:
        t = time.time()
        dense = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
        res["ppl_dense_fp32"] = perplexity(dense, evalw)
        del dense
        phase("eval_dense_fp32", t)

    from gptqmodel import GPTQModel, GPTQConfig, AWQConfig
    t = time.time()
    if a.method == "gptq":
        qcfg = GPTQConfig(bits=a.bits, group_size=a.group_size, device="cpu")
    else:
        qcfg = AWQConfig(bits=a.bits, group_size=a.group_size, device="cpu")
    res["quant_config"] = repr(qcfg)[:2000]
    model = GPTQModel.load(a.model, qcfg, dtype=a.dtype if a.dtype == "auto" else getattr(torch, a.dtype))  # QuantizeConfig.device="cpu"; passing device= here is rejected
    phase("load_for_quant", t)

    # Verify GPTQModel keeps exactly our samples: run the same normaliser quantize() uses.
    prepared = model.prepare_dataset(samples, calibration_dataset_sort="desc", batch_size=1)
    flat = [row for b in prepared for row in (b["input_ids"].tolist() if torch.is_tensor(b["input_ids"]) else b["input_ids"])]
    res["calib_check"] = {
        "passed": len(samples), "after_prepare": len(flat),
        "same_multiset": sorted(map(tuple, flat)) == sorted(tuple(s["input_ids"]) for s in samples),
        "lengths": sorted({len(r) for r in flat}),
    }
    print("calib_check", res["calib_check"], flush=True)

    t = time.time()
    model.quantize(samples, batch_size=1)
    phase("quantize", t)

    save_dir = a.save_dir or os.path.join(os.path.dirname(a.out), "_tmp_q_" + os.path.basename(a.out).replace(".json", ""))
    t = time.time()
    model.save(save_dir)
    del model
    phase("save", t)

    eval_quantized(a, res, phase, evalw, save_dir)
    if a.save_dir is None:
        shutil.rmtree(save_dir, ignore_errors=True)


def eval_quantized(a, res, phase, evalw, save_dir):
    from gptqmodel import GPTQModel, BACKEND
    t = time.time()
    # TorchLinear (the only GPTQModel CPU kernel that accepts K=576/960) supports only fp16/bf16 at inference,
    # so a model quantized in float32 is reloaded in bfloat16 for evaluation.
    rdt = {"auto": None, "float32": "bfloat16"}.get(a.dtype, a.dtype)
    res["reload_dtype"] = rdt or "auto"
    qm = GPTQModel.load(save_dir, device="cpu", backend=BACKEND(a.backend),
                        **({} if rdt is None else {"dtype": getattr(torch, rdt)}))
    res["quant_backend"] = str(getattr(qm, "backend", None))
    inner = getattr(qm, "model", qm)
    res["quant_qlinear_classes"] = sorted({type(m).__name__ for m in inner.modules() if "Quant" in type(m).__name__ or "Linear" in type(m).__name__})
    res["quant_dtype"] = str(next(inner.parameters()).dtype)
    phase("load_quantized", t)

    t = time.time()
    res["ppl_quant"] = perplexity(inner, evalw)
    phase("eval_quant", t)
    res["total_wall_s"] = stamp()
    res["peak_rss_gb"] = peak_gb()
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k.startswith(("ppl", "total", "peak", "calib"))}), flush=True)


if __name__ == "__main__":
    main()
