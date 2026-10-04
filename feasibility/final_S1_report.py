"""Tables for REPORT_final_S1.md from results/final_S1_runs.jsonl + final_S1_lmeval_items.jsonl (prints Markdown).
Eager noise is pooled from every eager run of the 128x512 W4 s0 anchor command (S1, S3, S2, REPORT_step2 §8)
and from the S1 128x2048 seed-0 repeats."""
import itertools, json, os, statistics as st
from scipy.stats import chi2

def rows(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
S1 = rows("results/final_S1_runs.jsonl")
R = {r["run_id"]: r for r in S1 if r.get("rc") == 0}
fails = [r for r in S1 if r.get("rc") not in (0, None)]
items = {json.loads(l)["run_id"]: json.loads(l) for l in open("results/final_S1_lmeval_items.jsonl")} \
    if os.path.exists("results/final_S1_lmeval_items.jsonl") else {}

def ci(sd, n):
    df = n - 1
    return sd * (df / chi2.ppf(0.975, df)) ** 0.5, sd * (df / chi2.ppf(0.025, df)) ** 0.5
def stats(v):
    n = len(v); m = st.mean(v); sd = st.stdev(v) if n > 1 else float("nan")
    lo, hi = ci(sd, n) if n > 1 else (float("nan"),) * 2
    return dict(n=n, mean=m, sd=sd, cv=100 * sd / m, min=min(v), max=max(v), range=max(v) - min(v), lo=lo, hi=hi)
def f(x, d=4): return "–" if x is None or x != x else f"{x:.{d}f}"
out = []

# ---------------- eager noise ----------------
anchor = []  # (source, run, ppl)
for k in ("anchor_a", "anchor_b"):
    if k in R: anchor.append(("S1", k, R[k]["metrics"]["wt_ppl"], R[k]["ckpt_sha256"][:16]))
for r in rows("results/final_S3_runs.jsonl"):
    if r.get("rc") == 0 and r["run_id"] in ("anchor_a", "anchor_b"):
        anchor.append(("S3", r["run_id"], r["metrics"]["wt_ppl"], r["ckpt_sha256"][:16]))
for r in rows("results/final_S2_anchor.jsonl"):
    if r.get("rc") == 0 and "eager" in r["run_id"]:
        anchor.append(("S2", r["run_id"], r["result"]["ppl_quant"], "–"))
for r in rows("results/real_step2_det.jsonl"):
    if r.get("rc") == 0 and r["run_id"] in ("s2_det_eager1", "s2_det_eager2"):
        anchor.append(("step2 §8", r["run_id"], r["result"]["ppl_quant"], "9b3c5e64c541388e"))
av = [a[2] for a in anchor]
mode = max(set(av), key=av.count)
noise_sd = st.stdev(av); noise_rng = max(av) - min(av)
out.append("### Eager run-to-run noise\n")
out.append("**A. Anchor command** (W4 g128 seed 0, 128 × 512 WikiText, WT-ppl): every eager run in this project\n")
out.append("| session | run | WT-ppl | ckpt sha (16) | Δ vs modal value |")
out.append("|---|---|---|---|---|")
for s, k, v, h in anchor:
    out.append(f"| {s} | {k} | {v!r} | `{h}` | {v - mode:+.6f} |")
out.append("")
out.append(f"- {len(av)} runs, {len(set(av))} distinct values, {av.count(mode)}/{len(av)} at the modal value {mode!r}.")
out.append(f"- Noise SD (all {len(av)} runs) **{noise_sd:.4f}**, range **{noise_rng:.4f}**, max |Δ| {max(abs(x - mode) for x in av):.4f}.")
out.append("")
out.append("**B. S1 128 × 2048 seed-0 repeats** (WT-ppl + checkpoint sha)\n")
out.append("| cell | run | WT-ppl | ckpt sha (16) | quantize s |")
out.append("|---|---|---|---|---|")
rep_delta = {}
for b in (3, 4):
    pair = [k for k in (f"w{b}_s0", f"w{b}_s0_rep") if k in R]
    for k in pair:
        out.append(f"| W{b} s0 | {k} | {R[k]['metrics']['wt_ppl']!r} | `{R[k]['ckpt_sha256'][:16]}` | {R[k]['phases']['quantize']['wall_s']} |")
    if len(pair) == 2:
        a, c = (R[k] for k in pair)
        rep_delta[b] = abs(a["metrics"]["wt_ppl"] - c["metrics"]["wt_ppl"])
        out.append(f"| W{b} s0 | → | Δ = {rep_delta[b]:.6f} | {'identical' if a['ckpt_sha256'] == c['ckpt_sha256'] else 'DIFFERENT'} | |")
out.append("")

# ---------------- dense ----------------
d = R.get("dense")
out.append("### Dense fp32 (eager)\n")
if d:
    m = d["metrics"]
    out.append(f"WT-ppl **{m['wt_ppl']:.4f}**, C4-ppl **{m.get('c4_ppl', float('nan')):.4f}**, arc_easy acc {m.get('arc_easy_acc')} "
               f"(acc_norm {m.get('arc_easy_acc_norm')}), hellaswag acc {m.get('hellaswag_acc')} (acc_norm {m.get('hellaswag_acc_norm')}); "
               f"wall {d['total_wall_s']} s, peak {d['peak_rss_gb']} GB.\n")

# ---------------- seed tables ----------------
def seeds(b, metric, ss):
    return [(s, R[f"w{b}_s{s}"]["metrics"][metric]) for s in ss if f"w{b}_s{s}" in R and metric in R[f"w{b}_s{s}"]["metrics"]]
out.append("### Seed statistics (128 × 2048 WikiText calibration)\n")
out.append("| setting | metric | seeds | n | mean | SD | CV % | min | max | range | 95 % CI for SD (χ²) | SD / anchor-noise SD | range / anchor-noise range |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
S = {}
specs = [(3, "wt_ppl", range(10), "0–9"), (3, "wt_ppl", range(5), "0–4"), (4, "wt_ppl", range(5), "0–4")]
for b in (3, 4):
    for mt in ("c4_ppl", "arc_easy_acc", "arc_easy_acc_norm", "hellaswag_acc", "hellaswag_acc_norm"):
        specs.append((b, mt, range(3), "0–2"))
for b, mt, ss, lab in specs:
    v = [x for _, x in seeds(b, mt, ss)]
    if len(v) < 2: continue
    z = stats(v); S[(b, mt, lab)] = z
    ppl = mt.endswith("ppl")
    out.append(f"| W{b} | {mt} | {lab} | {z['n']} | {f(z['mean'])} | {f(z['sd'])} | {f(z['cv'], 2)} | {f(z['min'])} | {f(z['max'])} | {f(z['range'])} | "
               f"[{f(z['lo'])}, {f(z['hi'])}] | {f(z['sd'] / noise_sd, 1) if ppl else '–'} | {f(z['range'] / noise_rng, 1) if ppl else '–'} |")
out.append("")
if d and (4, "wt_ppl", "0–4") in S and (3, "wt_ppl", "0–4") in S:
    w4, w3 = S[(4, "wt_ppl", "0–4")], S[(3, "wt_ppl", "0–9")] if (3, "wt_ppl", "0–9") in S else S[(3, "wt_ppl", "0–4")]
    g1 = w4["mean"] - d["metrics"]["wt_ppl"]; g2 = S[(3, "wt_ppl", "0–4")]["mean"] - w4["mean"]
    out.append("### Seed range relative to compression gaps (WT-ppl)\n")
    out.append(f"- dense → W4 (seeds 0–4 mean): **{g1:.4f}**; W4 → W3 (seeds 0–4 means): **{g2:.4f}**")
    for lab, z in (("W4 s0–4", w4), ("W3 s0–4", S[(3, "wt_ppl", "0–4")]), ("W3 s0–9", S.get((3, "wt_ppl", "0–9")))):
        if z: out.append(f"- {lab}: range {z['range']:.4f} = {100 * z['range'] / g1:.1f} % of dense→W4, {100 * z['range'] / g2:.1f} % of W4→W3; SD {z['sd']:.4f}")
    out.append("")

# ---------------- paired lm-eval flips ----------------
out.append("### Paired per-item agreement between seeds (lm-eval, 1000 items per task)\n")
out.append("| setting | task | metric | seed pair | items | flips | flip % | Δ accuracy |")
out.append("|---|---|---|---|---|---|---|---|")
flips = {}
for b in (3, 4):
    for task in ("arc_easy", "hellaswag"):
        for mt in ("acc", "acc_norm"):
            for s1, s2 in itertools.combinations(range(3), 2):
                a, c = items.get(f"w{b}_s{s1}"), items.get(f"w{b}_s{s2}")
                if not a or not c: continue
                A = dict(zip(a[task]["doc_id"], a[task][mt])); C = dict(zip(c[task]["doc_id"], c[task][mt]))
                ids = sorted(set(A) & set(C)); n_f = sum(A[i] != C[i] for i in ids)
                flips.setdefault((b, task, mt), []).append(100 * n_f / len(ids))
                out.append(f"| W{b} | {task} | {mt} | {s1}–{s2} | {len(ids)} | {n_f} | {100 * n_f / len(ids):.1f} | {(sum(C[i] for i in ids) - sum(A[i] for i in ids)) / len(ids):+.3f} |")
if "dense" in items:
    for b in (3, 4):
        for task in ("arc_easy", "hellaswag"):
            a, c = items["dense"], items.get(f"w{b}_s0")
            if not c: continue
            A = dict(zip(a[task]["doc_id"], a[task]["acc"])); C = dict(zip(c[task]["doc_id"], c[task]["acc"]))
            ids = sorted(set(A) & set(C)); n_f = sum(A[i] != C[i] for i in ids)
            out.append(f"| dense vs W{b} s0 | {task} | acc | – | {len(ids)} | {n_f} | {100 * n_f / len(ids):.1f} | {(sum(C[i] for i in ids) - sum(A[i] for i in ids)) / len(ids):+.3f} |")
out.append("")
for k, v in flips.items():
    out.append(f"- W{k[0]} {k[1]} {k[2]}: mean flip rate between seeds {st.mean(v):.1f} %")
out.append("")

# ---------------- per run ----------------
out.append("### Per run\n")
out.append("| run | bits | seed | calib fp | WT-ppl | C4-ppl | arc_easy acc | hellaswag acc | ckpt sha (16) | quantize s | WT s | C4 s | lm-eval s | wall s | peak RSS GB |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for k, r in R.items():
    p, m = r["phases"], r["metrics"]
    g = lambda ph: p.get(ph, {}).get("wall_s", "–")
    out.append(f"| {k} | {r.get('bits') or '–'} | {r.get('seed') if r.get('seed') is not None else '–'} | `{(r.get('calibration') or {}).get('fingerprint', '–')}` | "
               f"{m['wt_ppl']!r} | {f(m.get('c4_ppl'))} | {f(m.get('arc_easy_acc'), 3)} | {f(m.get('hellaswag_acc'), 3)} | "
               f"`{r.get('ckpt_sha256', '–')[:16]}` | {g('quantize')} | {g('eval_wt')} | {g('eval_c4')} | {g('eval_lmeval')} | {r['total_wall_s']} | {r['peak_rss_gb']} |")
if fails:
    out.append("\n**Failed runs**\n"); out += [f"- {r['run_id']} rc={r['rc']}: `{r.get('error_tail', '')[-300:]!r}`" for r in fails]
print("\n".join(out))
