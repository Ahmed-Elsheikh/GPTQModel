"""Print the Step 2 numbers used in REPORT_step2.md from results/real_step2_*.jsonl (+ Step 1 for comparison)."""
import json, glob, statistics as st
def rows(p):
    try: return [json.loads(l) for l in open(p)]
    except FileNotFoundError: return []
R = {r["run_id"]: r for p in sorted(glob.glob("results/real_step2_*.jsonl")) for r in rows(p)}
S1 = {r["run_id"]: r for r in rows("results/real_step1.jsonl")}
def res(i): return (R.get(i) or {}).get("result") or {}
print("== Wanda 50% unstructured (tf 4.47.1, fp32) ==")
d = res("s2_wanda_dense_135m").get("ppl")
ps = []
for s in range(5):
    r = R.get(f"s2_wanda50_135m_s{s}")
    if not r: continue
    x = r.get("result", {})
    print(f"seed {s}: ppl {x.get('ppl')}  sparsity {x.get('sparsity')}  prune {x.get('phases',{}).get('prune')} s  wall {r['wall_s']} s  peak {r['peak_rss_gb']} GB  calib_fp {x.get('calib_fingerprint')} eval_fp {x.get('eval_fingerprint')} rc {r['rc']}")
    if x.get("ppl"): ps.append(x["ppl"])
print("dense", d)
if len(ps) > 1:
    m, sd = st.mean(ps), st.stdev(ps)
    print(f"n={len(ps)} mean {m:.4f} sd {sd:.4f} min {min(ps):.4f} max {max(ps):.4f} range {max(ps)-min(ps):.4f} CV {100*sd/m:.3f}%")
    if d: print(f"dense->wanda increase (mean) {m-d:.4f}; range/increase {100*(max(ps)-min(ps))/(m-d):.2f}%")
for k in sorted(R):
    if k.startswith("s2_wanda"): continue
    r = R[k]; x = r.get("result") or r.get("report")
    short = {kk: x.get(kk) for kk in ("ppl_quant", "ppl_dense_fp32", "total_wall_s", "phases", "peak_rss_gb", "calib_fingerprint", "quant_qlinear_classes")} if isinstance(x, dict) and "args" in x else x
    if isinstance(x, dict) and "results" in x:
        short = {"wall_s": x["wall_s"], "load_s": x["load_s"], "peak": x["peak_rss_gb"],
                 "acc": {t: {m: v for m, v in d.items() if m.startswith(("acc", "acc_norm")) } for t, d in x["results"].items()}}
    print(k, "wall", r.get("wall_s"), "peak", r.get("peak_rss_gb"), "rc", r.get("rc"), json.dumps(short, default=str)[:900])
    if r.get("rc"): print("   ERR:", (r.get("error_tail") or "")[-400:].replace("\n", " | "))
print("step1 gptq_w4 s0:", (S1.get("gptq_w4_135m_s0") or {}).get("result", {}).get("ppl_quant"))
