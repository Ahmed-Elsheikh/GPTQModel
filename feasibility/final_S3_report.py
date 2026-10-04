"""Build the tables for REPORT_final_S3.md from results/final_S3_runs.jsonl (prints Markdown)."""
import json, statistics as st, sys
from scipy.stats import chi2
rows = [json.loads(l) for l in open("results/final_S3_runs.jsonl")]
R = {r["run_id"]: r for r in rows if r.get("rc") == 0}
fails = [r for r in rows if r.get("rc") != 0]
def ci_sd(sd, n):
    if n < 2: return (None, None)
    df = n - 1
    return (sd * (df / chi2.ppf(0.975, df)) ** 0.5, sd * (df / chi2.ppf(0.025, df)) ** 0.5)
def f(x, d=4): return "–" if x is None else f"{x:.{d}f}"
cells = {}
for b in (4, 3):
    for src in ("wikitext2", "c4"):
        for ev in ("wt_ppl", "c4_ppl"):
            v = [R[k]["metrics"][ev] for k in (f"w{b}_{src}_s{s}" for s in range(3)) if k in R]
            cells[(b, src, ev)] = v
out = []
out.append("| bits | calibration | eval | n | mean | SD | CV % | min | max | 95 % CI for SD (χ²) |")
out.append("|---|---|---|---|---|---|---|---|---|---|")
for (b, src, ev), v in cells.items():
    n = len(v); m = st.mean(v) if v else None; sd = st.stdev(v) if n > 1 else None
    lo, hi = ci_sd(sd, n) if sd is not None else (None, None)
    out.append(f"| W{b} | {src} | {ev.replace('_ppl','').upper().replace('WT','WikiText-2')} | {n} | {f(m)} | {f(sd)} | {f(100*sd/m if sd else None,2)} | {f(min(v) if v else None)} | {f(max(v) if v else None)} | [{f(lo)}, {f(hi)}] |")
out.append("")
for b in (4, 3):
    M = {k: st.mean(v) for k, v in cells.items() if k[0] == b and v}
    S = {k: (st.stdev(v) if len(v) > 1 else None) for k, v in cells.items() if k[0] == b}
    out.append(f"**W{b} 2×2 (mean ± SD over seeds 0–2)**\n")
    out.append("| calibration ↓ / eval → | WikiText-2 | C4 val | average of both |")
    out.append("|---|---|---|---|")
    for src in ("wikitext2", "c4"):
        a, c = M.get((b, src, "wt_ppl")), M.get((b, src, "c4_ppl"))
        out.append(f"| {src} | {f(a)} ± {f(S[(b,src,'wt_ppl')])} | {f(c)} ± {f(S[(b,src,'c4_ppl')])} | {f((a+c)/2 if a and c else None)} |")
    adv_wt = M[(b, "c4", "wt_ppl")] - M[(b, "wikitext2", "wt_ppl")]
    adv_c4 = M[(b, "wikitext2", "c4_ppl")] - M[(b, "c4", "c4_ppl")]
    avg_gap = (M[(b, "c4", "wt_ppl")] + M[(b, "c4", "c4_ppl")]) / 2 - (M[(b, "wikitext2", "wt_ppl")] + M[(b, "wikitext2", "c4_ppl")]) / 2
    pooled = lambda ev: (st.mean([S[(b, s, ev)] ** 2 for s in ("wikitext2", "c4") if S[(b, s, ev)] is not None])) ** 0.5
    out.append("")
    out.append(f"- In-domain advantage on WikiText-2 (C4-cal − WT-cal): **{adv_wt:+.4f}** = {adv_wt/pooled('wt_ppl'):.1f}× pooled seed SD ({pooled('wt_ppl'):.4f})")
    out.append(f"- In-domain advantage on C4 (WT-cal − C4-cal): **{adv_c4:+.4f}** = {adv_c4/pooled('c4_ppl'):.1f}× pooled seed SD ({pooled('c4_ppl'):.4f})")
    out.append(f"- Average over both eval sets, C4-cal − WT-cal: **{avg_gap:+.4f}**")
    out.append("")
out.append("**Per run**\n")
out.append("| run | bits | calib | seed | calib fp | WT-ppl | C4-ppl | ckpt sha (16) | quantize s | WT eval s | C4 eval s | wall s | peak RSS GB |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for k, r in R.items():
    p = r["phases"]; m = r["metrics"]
    out.append(f"| {k} | {r['bits']} | {r['calibration']['source_key']} | {r['seed']} | `{r['calibration']['fingerprint']}` | {m['wt_ppl']!r} | {m.get('c4_ppl','–')!r} | `{r['ckpt_sha256'][:16]}` | {p['quantize']['wall_s']} | {p['eval_wt']['wall_s']} | {p.get('eval_c4',{}).get('wall_s','–')} | {r['total_wall_s']} | {r['peak_rss_gb']} |")
out.append("")
out.append("**Eager-path noise (same command repeated)**\n")
out.append("| cell | run | WT-ppl | C4-ppl | ckpt sha (16) |")
out.append("|---|---|---|---|---|")
groups = {"W4 WikiText s0 (anchor + main)": ["anchor_a", "anchor_b", "w4_wikitext2_s0"]}
for b in (4, 3):
    for src in ("wikitext2", "c4"):
        groups.setdefault(f"W{b} {src} s0", []).extend([f"w{b}_{src}_s0", f"w{b}_{src}_s0_rep"])
for g, ids in groups.items():
    for k in ids:
        if k in R:
            m = R[k]["metrics"]; out.append(f"| {g} | {k} | {m['wt_ppl']!r} | {m.get('c4_ppl','–')!r} | `{R[k]['ckpt_sha256'][:16]}` |")
out.append("")
for g, ids in groups.items():
    vals = [R[k]["metrics"]["wt_ppl"] for k in ids if k in R]; shas = {R[k]["ckpt_sha256"] for k in ids if k in R}
    out.append(f"- {g}: {len(vals)} runs, {len(shas)} distinct checkpoints, WT-ppl spread {max(vals)-min(vals):.4f}" if vals else f"- {g}: no runs")
if fails:
    out.append("\n**Failed runs**\n"); out += [f"- {r['run_id']} rc={r['rc']}: {r.get('error_tail','')[-200:]!r}" for r in fails]
print("\n".join(out))
