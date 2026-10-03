"""Summarise results/real_step1.jsonl: per-setting mean/SD/min/max/CV of perplexity and yardsticks."""
import json, statistics as st, sys
rows = [json.loads(l) for l in open("results/real_step1.jsonl")]
last = {r["run_id"]: r for r in rows}           # latest record per run id
dense = last.get("dense_135m", {}).get("result", {})
out = {"dense_fp32": dense.get("ppl_dense_fp32"), "dense_bf16": dense.get("ppl_dense_bf16"), "settings": {}}
for b in (4, 3):
    runs = [last[f"gptq_w{b}_135m_s{s}"] for s in range(5) if f"gptq_w{b}_135m_s{s}" in last]
    ok = [r for r in runs if r["rc"] == 0 and "ppl_quant" in r.get("result", {})]
    p = [r["result"]["ppl_quant"] for r in ok]
    if not p: continue
    sd = st.stdev(p) if len(p) > 1 else float("nan")
    m = st.mean(p)
    out["settings"][f"W{b}g128"] = {
        "n": len(p), "ppl": p, "mean": m, "sd": sd, "min": min(p), "max": max(p), "cv_pct": 100 * sd / m,
        "quantize_s": [r["result"]["phases"]["quantize"]["wall_s"] for r in ok],
        "eval_s": [r["result"]["phases"]["eval_quant"]["wall_s"] for r in ok],
        "wall_s": [r["wall_s"] for r in ok], "peak_rss_gb": [r["peak_rss_gb"] for r in ok],
        "calib_fp": [r["result"]["calib_fingerprint"] for r in ok], "eval_fp": sorted({r["result"]["eval_fingerprint"] for r in ok}),
        "failed": [r["run_id"] for r in runs if r not in ok]}
S = out["settings"]
if "W4g128" in S and out["dense_fp32"]:
    out["yard_dense_to_w4_fp32"] = S["W4g128"]["mean"] - out["dense_fp32"]
    out["yard_dense_bf16_to_w4"] = S["W4g128"]["mean"] - out["dense_bf16"]
if "W4g128" in S and "W3g128" in S:
    out["yard_w4_to_w3"] = S["W3g128"]["mean"] - S["W4g128"]["mean"]
for k in ("W4g128", "W3g128"):
    if k in S:
        for y in ("yard_dense_to_w4_fp32", "yard_w4_to_w3"):
            if y in out: S[k][f"sd_over_{y[5:]}"] = S[k]["sd"] / out[y]
json.dump(out, open("results/real_step1_summary.json", "w"), indent=2)
print(json.dumps(out, indent=1))
