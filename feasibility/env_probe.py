"""Test 0: record environment facts to feasibility/env.json."""
import json, os, platform, shutil, subprocess, sys, importlib

def ver(mod):
    try:
        m = importlib.import_module(mod); return getattr(m, "__version__", "unknown")
    except Exception as e:
        return f"MISSING ({type(e).__name__})"

lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
info = {l.split(":", 1)[0].strip(): l.split(":", 1)[1].strip() for l in lscpu.splitlines() if ":" in l}
flags = info.get("Flags", "").split()
mem = open("/proc/meminfo").read().splitlines()
du = shutil.disk_usage("/home/user")
env = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "torch": ver("torch"), "transformers": ver("transformers"), "numpy": ver("numpy"),
    "cpu_model": info.get("Model name"), "cpu_count_logical": os.cpu_count(),
    "cores_per_socket": info.get("Core(s) per socket"), "threads_per_core": info.get("Thread(s) per core"),
    "simd_flags": [f for f in flags if f.startswith(("avx", "amx", "fma"))],
    "mem_total": mem[0], "mem_available": mem[2],
    "disk_total_gb": round(du.total / 1e9, 1), "disk_free_gb": round(du.free / 1e9, 1),
    "note": "df reports the host fs; the per-session writable allowance is ~30 GB",
}
try:
    import torch
    env["torch_cuda_available"] = torch.cuda.is_available()
    env["torch_num_threads_default"] = torch.get_num_threads()
    env["torch_config"] = torch.__config__.show().splitlines()[:12]
except Exception:
    pass
json.dump(env, open(os.path.join(os.path.dirname(__file__), "env.json"), "w"), indent=2)
print(json.dumps(env, indent=2))
