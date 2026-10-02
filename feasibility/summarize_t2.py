"""Collect lm-eval metrics + timings for Test 2 into results/t2/summary.json and print a table."""
import glob, json, os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "t2")
rows = {}
for line in open(os.path.join(R, "timings.tsv")):
    name, start, rc, wall, rss = line.rstrip("\n").split("\t")
    rows[name] = {"rc": int(rc), "wall": wall, "peak_rss_gb": round(int(rss) / 2**20, 2), "metrics": {}}
    files = sorted(glob.glob(os.path.join(R, name, "**", "results_*.json"), recursive=True))
    if files:
        res = json.load(open(files[-1]))
        for task, m in res["results"].items():
            for k, v in m.items():
                if isinstance(v, float) and "stderr" not in k:
                    rows[name]["metrics"][f"{task}:{k.split(',')[0]}"] = round(v, 4)
        rows[name]["n_samples"] = {t: v.get("effective", v.get("original")) if isinstance(v, dict) else v for t, v in res.get("n-samples", {}).items()}
        rows[name]["config"] = {k: res["config"].get(k) for k in ("model_args", "batch_size", "limit")}
json.dump(rows, open(os.path.join(R, "summary.json"), "w"), indent=2)
for n, r in rows.items():
    print(f"{n:18s} rc={r['rc']} wall={r['wall']:>8s} rss={r['peak_rss_gb']}GB {r['metrics']}")

# Per-sample log-likelihood deltas between protocol variants (accuracy on a random model is too coarse)
def lls(name, task):
    f = glob.glob(os.path.join(R, name, "**", f"samples_{task}_*.jsonl"), recursive=True)
    if not f: return None
    out = {}
    for line in open(f[0]):
        d = json.loads(line)
        out[d["doc_id"]] = [float(r[0][0]) if isinstance(r[0], list) else float(r[0]) for r in d["resps"]]
    return out
pairs = [("c_base", "c_bs1"), ("c_base", "c_bf16"), ("c_base", "c_fewshot5"), ("c_fewshot5", "c_maxlen512")]
deltas = {}
for a, b in pairs:
    for task in ("synth_arc_easy", "synth_hellaswag"):
        A, B = lls(a, task), lls(b, task)
        if not A or not B: continue
        d = [abs(x - y) for k in A if k in B for x, y in zip(A[k], B[k])]
        argmax_flips = sum(1 for k in A if k in B and max(range(len(A[k])), key=A[k].__getitem__) != max(range(len(B[k])), key=B[k].__getitem__))
        deltas[f"{a} vs {b} [{task}]"] = {"n_ll": len(d), "max_abs_dLL": round(max(d), 6), "mean_abs_dLL": round(sum(d) / len(d), 6), "pred_flips": argmax_flips}
json.dump(deltas, open(os.path.join(R, "ll_deltas.json"), "w"), indent=2)
for k, v in deltas.items(): print(k, v)
