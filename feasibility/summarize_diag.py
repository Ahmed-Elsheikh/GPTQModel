"""Diagnosis summary (REPORT.md A.6): all window-length conditions in one table, from the results JSONLs.
Uses the latest successful record per run id (a failed first attempt of diag_wanda_dense_135m is kept in the JSONL)."""
import json, re, statistics as st
from scipy.stats import ttest_ind

def rows(f):
    last = {}
    for r in map(json.loads, open(f)):
        if r["rc"] == 0: last[r["run_id"]] = r
    return last

S1, S1B, D = rows("results/real_step1.jsonl"), rows("results/real_step1b.jsonl"), rows("results/real_diag.jsonl")
def ppl(r):
    x = r["result"]; return x.get("ppl_quant", x.get("ppl"))
def t_main(r):
    x = r["result"]; ph = x.get("phases", {})
    return ph["quantize"]["wall_s"] if isinstance(ph.get("quantize"), dict) else ph.get("prune")

COND = [  # (key, label, source, regex, method)
    ("smol_w3_128x512", "SmolLM2-135M GPTQ W3, 128 x 512", S1, r"gptq_w3_135m_s\d", "GPTQ W3"),
    ("smol_w3_512x512", "SmolLM2-135M GPTQ W3, 512 x 512", D, r"diag_w3_135m_n512_L512_s\d", "GPTQ W3"),
    ("smol_w3_128x2048", "SmolLM2-135M GPTQ W3, 128 x 2048", S1B, r"gptq_w3_135m_L2048_s\d", "GPTQ W3"),
    ("smol_w4_128x512", "SmolLM2-135M GPTQ W4, 128 x 512", S1, r"gptq_w4_135m_s\d", "GPTQ W4"),
    ("smol_w4_128x2048", "SmolLM2-135M GPTQ W4, 128 x 2048", D, r"diag_w4_135m_L2048_s\d", "GPTQ W4"),
    ("smol_wanda_128x512", "SmolLM2-135M Wanda 50 %, 128 x 512", D, r"diag_wanda50_135m_L512_s\d", "Wanda"),
    ("smol_wanda_128x2048", "SmolLM2-135M Wanda 50 %, 128 x 2048", D, r"diag_wanda50_135m_L2048_s\d", "Wanda"),
    ("qwen_w3_128x512", "Qwen2.5-0.5B GPTQ W3, 128 x 512", D, r"diag_w3_qwen05_L512_s\d", "GPTQ W3"),
    ("qwen_w3_128x2048", "Qwen2.5-0.5B GPTQ W3, 128 x 2048", D, r"diag_w3_qwen05_L2048_s\d", "GPTQ W3"),
]
out = {"conditions": {}, "dense": {}}
for key, label, src, pat, meth in COND:
    rs = [r for k, r in sorted(src.items()) if re.fullmatch(pat, k)]
    p = [ppl(r) for r in rs]
    if not p: continue
    out["conditions"][key] = {"label": label, "n": len(p), "ppl": p, "mean": st.mean(p), "sd": st.stdev(p) if len(p) > 1 else None,
                              "min": min(p), "max": max(p), "quant_or_prune_s": [t_main(r) for r in rs], "wall_s": [r["wall_s"] for r in rs],
                              "peak_rss_gb": [r["peak_rss_gb"] for r in rs], "seeds": [r["result"].get("args", {}).get("seed", None) for r in rs]}
out["dense"]["smol_fp32_main"] = S1["dense_135m"]["result"]["ppl_dense_fp32"]
if "diag_wanda_dense_135m" in D: out["dense"]["smol_fp32_wanda_venv"] = D["diag_wanda_dense_135m"]["result"]["ppl"]
if "diag_qwen05_dense" in D: out["dense"]["qwen_fp32"] = D["diag_qwen05_dense"]["result"]["ppl_dense_fp32"]
C = out["conditions"]
def cmp(a, b):
    if a in C and b in C:
        pa, pb = C[a]["ppl"], C[b]["ppl"]
        d = {"diff": C[b]["mean"] - C[a]["mean"], "rel_pct": 100 * (C[b]["mean"] / C[a]["mean"] - 1)}
        if len(pa) > 1 and len(pb) > 1: d["welch_p"] = float(ttest_ind(pb, pa, equal_var=False).pvalue)
        return d
out["contrasts"] = {
    "smol_w3: 2048 vs 512": cmp("smol_w3_128x512", "smol_w3_128x2048"),
    "smol_w3: 512x512 vs 128x512": cmp("smol_w3_128x512", "smol_w3_512x512"),
    "smol_w3: 512x512 vs 128x2048": cmp("smol_w3_128x2048", "smol_w3_512x512"),
    "smol_w4: 2048 vs 512": cmp("smol_w4_128x512", "smol_w4_128x2048"),
    "smol_wanda: 2048 vs 512": cmp("smol_wanda_128x512", "smol_wanda_128x2048"),
    "qwen_w3: 2048 vs 512": cmp("qwen_w3_128x512", "qwen_w3_128x2048"),
}
json.dump(out, open("results/real_diag_summary.json", "w"), indent=2)
for k, c in C.items():
    print(f"{c['label']:42s} n={c['n']} mean={c['mean']:.3f} sd={c['sd'] if c['sd'] is None else round(c['sd'],3)} "
          f"min={c['min']:.3f} max={c['max']:.3f} t={st.mean(c['quant_or_prune_s']):.0f}s wall={st.mean(c['wall_s']):.0f}s peak={max(c['peak_rss_gb']):.2f}")
print(json.dumps(out["dense"])); print(json.dumps(out["contrasts"], indent=1))
