"""Run one experiment command, then append one JSON line to a results/real_*.jsonl file.

usage: runlog.py JSONL RUN_ID [--json OUT.json] -- cmd args...
Records wall time and peak RSS of the child process tree (RUSAGE_CHILDREN; /usr/bin/time is absent on this VM),
the lscpu model name and flags, package versions of the command venv, the exit code, the tail of stderr/stdout on failure, and the contents of OUT.json if the command wrote one.
"""
import json, os, resource, subprocess, sys, time, datetime

jsonl, run_id = sys.argv[1], sys.argv[2]
rest = sys.argv[3:]
out_json = None
if rest[0] == "--json":
    out_json, rest = rest[1], rest[2:]
assert rest[0] == "--"
cmd = rest[1:]
log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", f"real_{run_id}.log")
start = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
t = time.time()
with open(log, "w") as f:
    rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
wall = round(time.time() - t, 1)
peak = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20, 3)
def _lscpu():
    try:
        d = dict(l.split(":", 1) for l in subprocess.run(["lscpu"], capture_output=True, text=True).stdout.splitlines() if ":" in l)
        return {"cpu_model": d.get("Model name", "").strip(), "cpu_flags": d.get("Flags", "").strip()}
    except Exception as e:
        return {"cpu_model": None, "cpu_flags": None, "lscpu_error": repr(e)}

def _versions(py):
    """Versions of the key packages in the venv that runs the command (cmd[0] is that venv's python)."""
    code = ("import importlib.metadata as m, json, sys; out={'python': sys.version.split()[0]}\n"
            "for p in ('torch','transformers','tokenizers','gptqmodel','datasets','numpy','accelerate','lm_eval','optimum-benchmark'):\n"
            "    try: out[p]=m.version(p)\n"
            "    except Exception: pass\n"
            "print(json.dumps(out))")
    try:
        return json.loads(subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=120).stdout)
    except Exception as e:
        return {"error": repr(e)}

rec = {"run_id": run_id, **_lscpu(), "versions": next((_versions(c) for c in cmd if c.endswith(("/python", "/python3"))), None), "start_utc": start, "wall_s": wall, "peak_rss_gb": peak, "rc": rc, "log": os.path.relpath(log),
       "cmd": " ".join(cmd)}
if rc != 0:
    rec["error_tail"] = open(log, errors="replace").read()[-1500:]
if out_json and os.path.exists(out_json):
    try:
        rec["result"] = json.load(open(out_json))
    except Exception as e:
        rec["result_error"] = repr(e)
with open(jsonl, "a") as f:
    f.write(json.dumps(rec, default=str) + "\n")
print(json.dumps({k: rec[k] for k in ("run_id", "wall_s", "peak_rss_gb", "rc")}), flush=True)
