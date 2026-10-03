"""Extra check A: compare GPTQ W3 g128 with 512- vs 2048-token calibration windows (SmolLM2-135M, WikiText-2)."""
import json, math, statistics as st
from scipy.stats import chi2

def load(path):
    return {r["run_id"]: r for r in map(json.loads, open(path))}

def stats(p):
    n, m, sd = len(p), st.mean(p), st.stdev(p)
    lo, hi = (sd * math.sqrt((n - 1) / chi2.ppf(q, n - 1)) for q in (0.975, 0.025))
    return {"n": n, "mean": m, "sd": sd, "cv_pct": 100 * sd / m, "min": min(p), "max": max(p), "range": max(p) - min(p),
            "sd_ci95": [lo, hi]}

s1, s1b = load("results/real_step1.jsonl"), load("results/real_step1b.jsonl")
out = {}
for L, src, fmt in ((512, s1, "gptq_w3_135m_s{}"), (2048, s1b, "gptq_w3_135m_L2048_s{}")):
    runs = [src[fmt.format(s)] for s in range(5) if fmt.format(s) in src]
    ok = [r for r in runs if r["rc"] == 0 and "ppl_quant" in r.get("result", {})]
    out[L] = {**stats([r["result"]["ppl_quant"] for r in ok]),
              "runs": [{"seed": r["result"]["args"]["seed"], "ppl": r["result"]["ppl_quant"],
                        "quantize_s": r["result"]["phases"]["quantize"]["wall_s"], "eval_s": r["result"]["phases"]["eval_quant"]["wall_s"],
                        "wall_s": r["wall_s"], "peak_rss_gb": r["peak_rss_gb"], "calib_fp": r["result"]["calib_fingerprint"],
                        "eval_fp": r["result"]["eval_fingerprint"], "cpu_model": r.get("cpu_model")} for r in ok],
              "failed": [r["run_id"] for r in runs if r not in ok]}
if len(out[512]["runs"]) > 1 and len(out[2048]["runs"]) > 1:
    out["sd_ratio_2048_over_512"] = out[2048]["sd"] / out[512]["sd"]
    out["f_test_p_two_sided"] = None
    from scipy.stats import f, ttest_ind
    F = out[2048]["sd"] ** 2 / out[512]["sd"] ** 2
    d1, d2 = out[2048]["n"] - 1, out[512]["n"] - 1
    out["f_stat"] = F
    out["f_test_p_two_sided"] = 2 * min(f.cdf(F, d1, d2), f.sf(F, d1, d2))
    out["welch_t_mean_diff"] = out[2048]["mean"] - out[512]["mean"]
    out["welch_t_p"] = float(ttest_ind([r["ppl"] for r in out[2048]["runs"]], [r["ppl"] for r in out[512]["runs"]], equal_var=False).pvalue)
json.dump(out, open("results/real_step1b_summary.json", "w"), indent=2)
print(json.dumps({L: {k: v for k, v in d.items() if k != "runs"} if isinstance(d, dict) else d for L, d in out.items()}, indent=1))
