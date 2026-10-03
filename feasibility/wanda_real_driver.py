"""Run locuslab/wanda main.py (patched only by patches/wanda_cpu.patch) on real data, with two driver-side swaps:
- calibration: wanda hard-codes C4; here get_loaders returns calib.draw_windows("wikitext2", n, L=model.seqlen=512, seed),
  i.e. the same WikiText-2 train windows GPTQ uses in Step 1 (target = -100 except last, as wanda's loaders do);
- perplexity: main.eval_ppl is replaced by quant_eval.perplexity on calib.eval_windows("wikitext2", 40, 2048),
  i.e. the Step 1 protocol (first 40 non-overlapping 2048-token windows of the test split).
usage: python wanda_real_driver.py OUT.json <wanda main.py args...>   (--sparsity_ratio 0 = dense baseline)"""
import json, os, random, sys, time, resource
HERE = os.path.dirname(os.path.abspath(__file__))
WANDA = os.path.join(os.path.dirname(HERE), "third_party", "wanda")
sys.path[:0] = [WANDA, HERE]
os.chdir(WANDA)
import torch
torch.set_num_threads(4)
import transformers
import calib
import lib.data, lib.prune, lib.eval
from quant_eval import perplexity

out = sys.argv[1]
res = {"argv": sys.argv[2:], "transformers": transformers.__version__, "torch": torch.__version__, "phases": {}}

def real_get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    tok = transformers.AutoTokenizer.from_pretrained(res["model"])  # fast tokenizer (same ids as Step 1)
    wins = calib.draw_windows("wikitext2", tok, n=nsamples, L=seqlen, seed=seed)
    res["calib_requested_by_wanda"] = name
    res["calib_source"] = "wikitext2"
    res["calib_fingerprint"] = calib.fingerprint(wins)
    loader = []
    for w in wins:
        inp = torch.tensor([w["input_ids"]]); tar = inp.clone(); tar[:, :-1] = -100
        loader.append((inp, tar))
    print(f"[driver] wikitext2 calib {nsamples}x{seqlen} seed {seed} fp {res['calib_fingerprint']}", flush=True)
    return loader, None

def real_eval_ppl(args, model, tokenizer, device):
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    evalw = calib.eval_windows("wikitext2", tok, n=40, L=2048)
    res["eval_fingerprint"] = calib.fingerprint(evalw)
    t = time.time()
    ppl = perplexity(model, evalw)
    res["phases"]["eval"] = round(time.time() - t, 1)
    res["ppl"] = ppl
    return ppl

lib.data.get_loaders = lib.prune.get_loaders = lib.eval.get_loaders = real_get_loaders
import main as wanda_main
wanda_main.eval_ppl = real_eval_ppl
_prune = wanda_main.prune_wanda
def timed_prune(*a, **k):
    t = time.time(); r = _prune(*a, **k); res["phases"]["prune"] = round(time.time() - t, 1); return r
wanda_main.prune_wanda = timed_prune
_check = wanda_main.check_sparsity
def rec_sparsity(m):
    s = _check(m); res["sparsity"] = float(s); return s
wanda_main.check_sparsity = rec_sparsity

argv = sys.argv[2:]
res["model"] = argv[argv.index("--model") + 1]
t = time.time()
sys.argv = ["main.py"] + argv
try:
    wanda_main.main()
finally:
    res["wall_s"] = round(time.time() - t, 1)
    res["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
    json.dump(res, open(out, "w"), indent=2)
    print("[driver]", json.dumps({k: res.get(k) for k in ("ppl", "sparsity", "phases", "wall_s", "peak_rss_gb")}), flush=True)
