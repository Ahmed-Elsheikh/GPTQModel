"""Determinism probe wrapper: apply one setting, then run quant_eval.py unchanged with the given argv.
usage: DET_SETTING={none,deterministic,threads1,eager} step2_det.py <quant_eval.py args>
Records the setting, the effective attention implementation, CPU flags and torch build config in OUT.json["det"]."""
import json, os, runpy, subprocess, sys
import torch
setting = os.environ.get("DET_SETTING", "none")
info = {"setting": setting}
if setting == "deterministic":
    torch.use_deterministic_algorithms(True)
seen = []
import gptqmodel
_load = gptqmodel.GPTQModel.load
def _load_rec(*a, **k):
    if setting == "eager":  # GPTQModel's loader writes this into the HF config (loader.py _override_attn_implementation)
        k.setdefault("attn_implementation", "eager")
    m = _load(*a, **k)
    inner = getattr(m, "model", m)
    seen.append(getattr(inner.config, "_attn_implementation", None))
    return m
gptqmodel.GPTQModel.load = staticmethod(_load_rec)
lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
info.update(cpu_model=next(l for l in lscpu.splitlines() if l.startswith("Model name")).split(":", 1)[1].strip(),
            cpu_flags=next(l for l in lscpu.splitlines() if l.startswith("Flags")).split(":", 1)[1].strip(),
            torch=torch.__version__, torch_config=torch.__config__.show(), mkldnn=torch.backends.mkldnn.is_available(),
            deterministic=torch.are_deterministic_algorithms_enabled(),
            gptqmodel=gptqmodel.__version__ if hasattr(gptqmodel, "__version__") else None)
import transformers; info["transformers"] = transformers.__version__
out = sys.argv[sys.argv.index("--out") + 1]
sys.argv = ["quant_eval.py"] + sys.argv[1:]
try:
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "quant_eval.py"), run_name="__main__")
finally:
    info["attn_impl_at_load"] = seen
    info["torch_threads_end"] = torch.get_num_threads()
    try:
        r = json.load(open(out))
    except Exception:
        r = {}
    r["det"] = info
    json.dump(r, open(out, "w"), indent=2)
