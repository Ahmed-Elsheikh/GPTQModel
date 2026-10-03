"""S2 run wrapper: apply the determinism configuration, run quant_eval.py unchanged, then record the configuration,
the checkpoint SHA-256 and the effective attention implementation in the OUT JSON.

FINAL_CFG (env): "eager"          -> attn_implementation="eager" for every model load (GPTQModel.load for quantize and
                                     reload, and AutoModelForCausalLM.from_pretrained for dense evals), default threads;
                 "eager_threads1" -> eager + torch.set_num_threads(1) (pass --threads 1 to quant_eval as well).
                 "default"        -> nothing changed (SDPA attention, 4 threads).
Same mechanism as Session C's step2_det.py (eager is passed through GPTQModel.load's kwargs into the HF config).
The quantized checkpoint (--save_dir) is hashed (sha256 over *.safetensors, sorted) and then deleted to save disk."""
import glob, hashlib, json, os, runpy, shutil, sys
import torch
cfg = os.environ.get("FINAL_CFG", "default")
assert cfg in ("default", "eager", "eager_threads1"), cfg
eager = cfg.startswith("eager")
seen = []
import gptqmodel, transformers
_load = gptqmodel.GPTQModel.load
def _load_cfg(*a, **k):
    if eager:
        k.setdefault("attn_implementation", "eager")
    m = _load(*a, **k)
    seen.append(getattr(getattr(m, "model", m).config, "_attn_implementation", None))
    return m
gptqmodel.GPTQModel.load = staticmethod(_load_cfg)
_fp = transformers.AutoModelForCausalLM.from_pretrained
def _fp_cfg(*a, **k):
    if eager:
        k.setdefault("attn_implementation", "eager")
    m = _fp(*a, **k)
    seen.append(getattr(m.config, "_attn_implementation", None))
    return m
transformers.AutoModelForCausalLM.from_pretrained = _fp_cfg

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1]
save_dir = argv[argv.index("--save_dir") + 1] if "--save_dir" in argv else None
sys.argv = ["quant_eval.py"] + argv
try:
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "quant_eval.py"), run_name="__main__")
finally:
    info = {"final_cfg": cfg, "attn_implementation": "eager" if eager else "sdpa(default)", "attn_impl_seen": seen,
            "torch_threads_end": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    if save_dir and os.path.isdir(save_dir):
        files = sorted(glob.glob(os.path.join(save_dir, "*.safetensors")))
        h = hashlib.sha256()
        for f in files: h.update(open(f, "rb").read())
        info["ckpt_sha256"] = h.hexdigest() if files else None
        info["ckpt_files"] = [os.path.basename(f) for f in files]
        shutil.rmtree(save_dir, ignore_errors=True)
    try:
        r = json.load(open(out))
    except Exception:
        r = {}
    r["config"] = info
    json.dump(r, open(out, "w"), indent=2)
