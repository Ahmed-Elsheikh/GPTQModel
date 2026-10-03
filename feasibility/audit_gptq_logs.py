"""Diagnosis step 1: audit GPTQModel's per-module logs (logs/gptq_log_*.log) of the W3 512- vs 2048-token runs.

Each gptq_log file is a stream of JSON objects {layer, module, loss, samples, damp, ...}. Files are matched to runs
by start time (run start_utc from the results JSONL; GPTQModel creates its log ~15-60 s later, at quantize start).

Loss normalisation (gptqmodel/quantization/gptq.py): H = (2/N) * sum x x^T (a per-token mean, independent of N) and
the logged avg_loss = sum(Losses) / N with N = calibration tokens. So the logged loss scales as 1/N, and runs with
different token counts must be compared on loss * samples (= sum(Losses)), which is what this script reports.
"""
import glob, json, os, re, statistics as st, sys, datetime as dt
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

def parse_log(path):
    txt = open(path).read()
    return [json.loads(m) for m in re.findall(r"\{[^{}]*\}", txt)]

def log_time(path):
    m = re.search(r"time_(\d\d)_(\d\d)_(\d{4})_(\d\d)h_(\d\d)m_(\d\d)s", path)
    mo, d, y, H, M, S = map(int, m.groups())
    return dt.datetime(y, mo, d, H, M, S)

def runs_from(jsonl, pattern):
    out = {}
    for l in open(os.path.join(HERE, jsonl)):
        r = json.loads(l)
        if re.fullmatch(pattern, r["run_id"]):
            out[r["run_id"]] = r
    return out

def match_logs(runs):
    logs = sorted(glob.glob(os.path.join(HERE, "logs", "gptq_log_*.log")), key=log_time)
    m = {}
    for rid, r in runs.items():
        t0 = dt.datetime.fromisoformat(r["start_utc"].rstrip("Z"))
        cands = [p for p in logs if 0 <= (log_time(p) - t0).total_seconds() <= 180]
        m[rid] = cands[0] if cands else None
    return m

def summarise(arm_runs):
    """per-run: samples/damp sets, module count, summed corrected loss; per (layer,module): list of corrected losses."""
    per_mod = defaultdict(list); per_run = {}
    for rid, path in arm_runs.items():
        recs = [x for x in parse_log(path) if x.get("process") == "gptq"]
        samples = sorted({int(x["samples"]) for x in recs}); damps = sorted({x["damp"] for x in recs})
        tot = 0.0
        for x in recs:
            c = float(x["loss"]) * int(x["samples"])
            per_mod[(int(x["layer"]), x["module"])].append(c); tot += c
        per_run[rid] = {"log": os.path.basename(path), "modules": len(recs), "samples": samples, "damps": damps,
                        "sum_corrected_loss": tot, "sum_logged_loss": sum(float(x["loss"]) for x in recs)}
    return per_run, per_mod

def main():
    arms = {
        "W3_128x512": runs_from("results/real_step1.jsonl", r"gptq_w3_135m_s\d"),
        "W3_128x2048": runs_from("results/real_step1b.jsonl", r"gptq_w3_135m_L2048_s\d"),
    }
    extra = [a.split("=", 1) for a in sys.argv[1:]]  # optional NAME=JSONL:REGEX
    for name, spec in extra:
        f, pat = spec.split(":", 1); arms[name] = runs_from(f, pat)
    out = {"note": __doc__.strip().splitlines()[0], "arms": {}}
    mods = {}
    for name, runs in arms.items():
        logs = match_logs(runs)
        missing = [k for k, v in logs.items() if v is None]
        per_run, per_mod = summarise({k: v for k, v in logs.items() if v})
        mods[name] = per_mod
        out["arms"][name] = {"runs": per_run, "missing_logs": missing}
    a, b = "W3_128x512", "W3_128x2048"
    rows = []
    for key in sorted(mods[a]):
        if key in mods[b]:
            ma, mb = st.mean(mods[a][key]), st.mean(mods[b][key])
            rows.append({"layer": key[0], "module": key[1], "loss512": ma, "loss2048": mb, "ratio_2048_512": mb / ma})
    out["per_module"] = rows
    by_layer = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        by_layer[r["layer"]][0] += r["loss512"]; by_layer[r["layer"]][1] += r["loss2048"]
    out["per_layer"] = [{"layer": L, "loss512": v[0], "loss2048": v[1], "ratio": v[1] / v[0]} for L, v in sorted(by_layer.items())]
    by_type = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        by_type[r["module"]][0] += r["loss512"]; by_type[r["module"]][1] += r["loss2048"]
    out["per_module_type"] = {k: {"loss512": v[0], "loss2048": v[1], "ratio": v[1] / v[0]} for k, v in by_type.items()}
    tot_a = sum(r["loss512"] for r in rows); tot_b = sum(r["loss2048"] for r in rows)
    out["total"] = {"loss512": tot_a, "loss2048": tot_b, "ratio": tot_b / tot_a}
    json.dump(out, open(os.path.join(HERE, "results", "real_diag_audit.json"), "w"), indent=2)
    return out

if __name__ == "__main__":
    o = main()
    for k, v in o["arms"].items():
        print(k, "missing", v["missing_logs"])
        for rid, r in v["runs"].items():
            print(" ", rid, r["log"], r["modules"], r["samples"], r["damps"], f"{r['sum_corrected_loss']:.4f}")
    print("total", o["total"])
    print("per type", json.dumps(o["per_module_type"], indent=0))
    print("per layer (ratio 2048/512):", " ".join(f"{x['layer']}:{x['ratio']:.2f}" for x in o["per_layer"]))
    top = sorted(o["per_module"], key=lambda r: -abs(r["loss2048"] - r["loss512"]))[:8]
    print("largest absolute divergence:"); [print(" ", r) for r in top]
