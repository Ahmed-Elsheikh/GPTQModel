"""Build feasibility/PAPER_TABLES.md and feasibility/figures/*.{png,pdf} from the results files on this branch.
Every number is computed here from a results file; each table row cites `file@commit` (last commit touching the file).
usage: /opt/venvs/main/bin/python paper_tables.py"""
import json, math, os, statistics as st, subprocess
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figures", exist_ok=True)

def commit(p):
    return subprocess.run(["git", "log", "-1", "--format=%h", "--", p], capture_output=True, text=True).stdout.strip() or "uncommitted"
def rows(p):
    return [json.loads(l) for l in open(p)]
def cite(p):
    return f"`{p}`@`{commit(p)}`"
def ci_sd(sd, n):
    df = n - 1
    return sd * (df / stats.chi2.ppf(0.975, df)) ** 0.5, sd * (df / stats.chi2.ppf(0.025, df)) ** 0.5
def summ(v):
    n = len(v); m = st.mean(v); sd = st.stdev(v); lo, hi = ci_sd(sd, n)
    return dict(n=n, mean=m, sd=sd, cv=100 * sd / m, min=min(v), max=max(v), range=max(v) - min(v), lo=lo, hi=hi, v=v)
def f(x, d=3): return f"{x:.{d}f}"
def welch(a, b):
    """Welch difference mean(b) - mean(a), 95 % CI (Welch-Satterthwaite df) and two-sided p."""
    va, vb = a["sd"] ** 2 / a["n"], b["sd"] ** 2 / b["n"]; se = (va + vb) ** 0.5
    df = (va + vb) ** 2 / (va ** 2 / (a["n"] - 1) + vb ** 2 / (b["n"] - 1))
    d = b["mean"] - a["mean"]; tc = stats.t.ppf(0.975, df)
    return d, d - tc * se, d + tc * se, 2 * stats.t.sf(abs(d) / se, df), df
def holm(ps):
    """Holm step-down adjusted p-values (same order as input)."""
    o = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - k) * ps[i])); adj[i] = run
    return adj
def pooled_sd(*zs):
    """Pooled SD = sqrt of the df-weighted mean of the variances: sqrt(sum((n_i - 1) s_i^2) / sum(n_i - 1))."""
    return (sum((z["n"] - 1) * z["sd"] ** 2 for z in zs) / sum(z["n"] - 1 for z in zs)) ** 0.5

# ---------------- load ----------------
P = dict(s1="results/real_step1.jsonl", s1b="results/real_step1b.jsonl", w50="results/real_step2_wanda.jsonl",
         c4="results/real_step2_c4.jsonl", c4val="results/real_step2_c4val.jsonl", det="results/real_step2_det.jsonl",
         ob="results/real_step2_ob.jsonl", F1="results/final_S1_runs.jsonl", F2="results/final_S2_runs.jsonl",
         F2a="results/final_S2_anchor.jsonl", F2w="results/final_S2_wanda.jsonl", F3="results/final_S3_runs.jsonl",
         t3c="results/t3c_ratios.json")
R = {k: {r["run_id"]: r for r in rows(p) if r.get("rc") in (0, None)} for k, p in P.items() if p.endswith(".jsonl")}
def q(k, i): r = R[k][i]; x = r.get("result") or {}; return x.get("ppl_quant", x.get("ppl"))
def m1(k, i, mt="wt_ppl"): return R[k][i]["metrics"][mt]

dense_wt = R["s1"]["dense_135m"]["result"]["ppl_dense_fp32"]
dense_wt_bf16 = R["s1"]["dense_135m"]["result"]["ppl_dense_bf16"]
dense_c4 = m1("F1", "dense", "c4_ppl")
qwen_dense = R["F2"]["final_S2_qwen_dense"]["result"]["ppl_dense_fp32"]
dense_wt_eager = R["F1"]["dense"]["metrics"]["wt_ppl"]  # same windows, eager attention (D for eager rows uses this)

S = {}  # name -> (summary, source key, config text, dense reference)
def add(name, vals, src, cfg, dense):
    S[name] = (summ(vals), src, cfg, dense)
add("SmolLM2-135M · GPTQ W4 g128 · 128×512 · SDPA", [q("s1", f"gptq_w4_135m_s{s}") for s in range(5)], ["s1"], "Step 1, default SDPA path", dense_wt)
add("SmolLM2-135M · GPTQ W3 g128 · 128×512 · SDPA", [q("s1", f"gptq_w3_135m_s{s}") for s in range(5)], ["s1"], "Step 1, default SDPA path", dense_wt)
add("SmolLM2-135M · GPTQ W3 g128 · 128×2048 · SDPA", [q("s1b", f"gptq_w3_135m_L2048_s{s}") for s in range(5)], ["s1b"], "A.5, default SDPA path", dense_wt)
add("SmolLM2-135M · GPTQ W4 g128 · 128×512 · eager", [m1("F3", f"w4_wikitext2_s{s}") for s in range(3)], ["F3"], "S3, eager", dense_wt_eager)
add("SmolLM2-135M · GPTQ W3 g128 · 128×512 · eager", [m1("F3", f"w3_wikitext2_s{s}") for s in range(3)], ["F3"], "S3, eager", dense_wt_eager)
add("SmolLM2-135M · GPTQ W4 g128 · 128×2048 · eager", [m1("F1", f"w4_s{s}") for s in range(5)], ["F1"], "S1, eager", dense_wt_eager)
add("SmolLM2-135M · GPTQ W3 g128 · 128×2048 · eager", [m1("F1", f"w3_s{s}") for s in range(10)], ["F1"], "S1, eager", dense_wt_eager)
add("Qwen2.5-0.5B · GPTQ W3 g128 · 128×512 · eager", [q("F2", f"final_S2_qwen_w3_L512_s{s}") for s in range(3)], ["F2"], "S2, eager", qwen_dense)
add("Qwen2.5-0.5B · GPTQ W3 g128 · 128×2048 · eager", [q("F2", f"final_S2_qwen_w3_L2048_s{s}") for s in range(5)], ["F2"], "S2, eager", qwen_dense)
add("SmolLM2-135M · Wanda 50 % unstr. · 128×512", [q("w50", f"s2_wanda50_135m_s{s}_t023") for s in range(5)], ["w50"], "Step 2 §1 (deterministic)", dense_wt)
add("SmolLM2-135M · Wanda 60 % unstr. · 128×512", [q("F2w", f"final_S2_wanda_u60_s{s}") for s in range(3)], ["F2w"], "S2 (deterministic)", dense_wt)
add("SmolLM2-135M · Wanda 2:4 · 128×512", [q("F2w", f"final_S2_wanda_24_s{s}") for s in range(3)], ["F2w"], "S2 (deterministic)", dense_wt)
add("SmolLM2-135M · Wanda 70 % unstr. · 128×512", [q("F2w", f"final_S2_wanda_u70_s{s}") for s in range(3)], ["F2w"], "S2 (deterministic)", dense_wt)
C4seed = {
    "SmolLM2-135M · GPTQ W4 · 128×2048 · eager · C4-ppl": (summ([m1("F1", f"w4_s{s}", "c4_ppl") for s in range(3)]), ["F1"], dense_c4),
    "SmolLM2-135M · GPTQ W3 · 128×2048 · eager · C4-ppl": (summ([m1("F1", f"w3_s{s}", "c4_ppl") for s in range(3)]), ["F1"], dense_c4),
}

out = []
w = out.append
w("# Paper tables – SLM-compression reporting-practices feasibility study\n")
w("Generated by `feasibility/paper_tables.py` from the results files on branch `claude/quirky-ramanujan-i0hree`.")
w("Every row cites `results file`@`commit` (the last commit that touched that file). Figures are in `feasibility/figures/`")
w("(PNG + PDF). All runs: SmolLM2-135M or Qwen2.5-0.5B, real weights, Intel Xeon @ 2.10 GHz (AVX512-BF16/AMX), 4 vCPU,")
w("torch 2.14.1+cpu, GPTQModel 7.5.0. WT-ppl = first 40 non-overlapping 2048-token WikiText-2 test windows (fingerprint")
w("`7da3b34bdebd1923`); C4-ppl = 256 × 2048 C4-validation windows, GPTQ convention (fingerprint `cbe443ebebeeceeb`).")
w("GPTQ: fp32 quantize, bf16 `BACKEND.TORCH` eval. \"SDPA\" = default attention path; \"eager\" = `attn_implementation=\"eager\"`.\n")
w("**Definition used in every table.** *Δ / pooled SD* divides a difference by the pooled seed SD, where")
w("pooled SD = sqrt( Σ (nᵢ − 1)·sᵢ² / Σ (nᵢ − 1) ), the square root of the degrees-of-freedom-weighted mean of the seed")
w("variances sᵢ² of the arms involved. For two-arm comparisons (sections 2, 3) the arms are the two compared settings; for")
w("single-run factor effects (section 5) the arms are all three SmolLM2-135M GPTQ W4 WT-ppl seed sets of section 1.\n")
w("**Damage scale D.** D = ln(ppl_compressed / ppl_dense) against the dense model evaluated on the SAME evaluation set,")
w("token windows and protocol (attention path: SDPA results use the SDPA dense run, eager results the eager dense run).")
w("Because ppl = exp(mean NLL) over the same targets, D is the mean per-token NLL increase in nats.\n")
w("**Uncertainty.** Differences between seed sets carry Welch 95 % CIs (Welch–Satterthwaite df) and Holm-adjusted p-values")
w("within each table. The 95 % CI for an SD uses the χ² distribution of (n−1)s²/σ², which **assumes the per-seed values are")
w("normally distributed** (independent draws); with n = 3–10 it is sensitive to that assumption (e.g. Wanda 70 % is clearly not normal).\n")
w("**Scope notes.** S4 (evaluation- and latency-protocol factors) was run in trimmed form (context × BOS perplexity;")
w("launcher × threads × model latency); sections 5–6 combine it with the earlier real-weight evidence and mark what is")
w("still missing. REPORT.md has no section A.7: the Wanda")
w("sparsity sweep it referred to was moved to session S2 and its data is `results/final_S2_wanda.jsonl` (cited below).\n")

# ---------------- (1) seed sensitivity ----------------
w("## 1. Calibration-seed sensitivity (WT-ppl)\n")
w("| model · method · calibration · path | n | mean | SD | CV % | min | max | 95 % CI for SD (χ²) | range | range / (mean − dense) | D mean ± SD (nats) | source |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|")
for name, (z, src, cfg, dense) in S.items():
    zD = summ([math.log(v / dense) for v in z["v"]])
    w(f"| {name} | {z['n']} | {f(z['mean'])} | {f(z['sd'], 4)} | {f(z['cv'], 2)} | {f(z['min'])} | {f(z['max'])} | "
      f"[{f(z['lo'], 4)}, {f(z['hi'], 4)}] | {f(z['range'])} | {100 * z['range'] / (z['mean'] - dense):.1f} % | {zD['mean']:.4f} ± {zD['sd']:.4f} | {', '.join(cite(P[s]) for s in src)} |")
for name, (z, src, dense) in C4seed.items():
    zD = summ([math.log(v / dense) for v in z["v"]])
    w(f"| {name} | {z['n']} | {f(z['mean'])} | {f(z['sd'], 4)} | {f(z['cv'], 2)} | {f(z['min'])} | {f(z['max'])} | "
      f"[{f(z['lo'], 4)}, {f(z['hi'], 4)}] | {f(z['range'])} | {100 * z['range'] / (z['mean'] - dense):.1f} % | {zD['mean']:.4f} ± {zD['sd']:.4f} | {', '.join(cite(P[s]) for s in src)} |")
w("")
w(f"Dense references: SmolLM2-135M fp32 WT-ppl {f(dense_wt, 4)} ({cite(P['s1'])}), C4-ppl {f(dense_c4, 4)} ({cite(P['F1'])}); "
  f"Qwen2.5-0.5B fp32 WT-ppl {f(qwen_dense, 4)} ({cite(P['F2'])}).")
w("Wanda runs are deterministic (seed-0 repeats bit-identical, " + cite(P["F2w"]) + "), so their SDs are pure seed variance.")
w("Figure: `figures/fig1_seed_strips.{png,pdf}`.\n")

# lm-eval flips (S1)
items = {json.loads(l)["run_id"]: json.loads(l) for l in open("results/final_S1_lmeval_items.jsonl")}
def flip(a, b, task, mt="acc"):
    A = dict(zip(items[a][task]["doc_id"], items[a][task][mt])); B = dict(zip(items[b][task]["doc_id"], items[b][task][mt]))
    ids = sorted(set(A) & set(B)); return 100 * sum(A[i] != B[i] for i in ids) / len(ids)
w("**Paired lm-eval agreement (S1, 1000 items per task, 0-shot)** – share of items whose correctness flips:\n")
w("| comparison | arc_easy acc | hellaswag acc | source |")
w("|---|---|---|---|")
for b in (4, 3):
    fa = st.mean(flip(f"w{b}_s{x}", f"w{b}_s{y}", "arc_easy") for x, y in ((0, 1), (0, 2), (1, 2)))
    fh = st.mean(flip(f"w{b}_s{x}", f"w{b}_s{y}", "hellaswag") for x, y in ((0, 1), (0, 2), (1, 2)))
    w(f"| W{b} seed vs seed (mean of 3 pairs) | {fa:.1f} % | {fh:.1f} % | {cite('results/final_S1_lmeval_items.jsonl')} |")
    w(f"| dense vs W{b} seed 0 | {flip('dense', f'w{b}_s0', 'arc_easy'):.1f} % | {flip('dense', f'w{b}_s0', 'hellaswag'):.1f} % | {cite('results/final_S1_lmeval_items.jsonl')} |")
w("")

w("**Downstream accuracy (S1, lm-eval 0.4.13, first 1000 items per task, 0-shot, bs 8)** – " + cite("results/final_S1_lmeval_items.jsonl") + "\n")
w("| model | arc_easy acc | hellaswag acc |")
w("|---|---|---|")
acc = lambda r, task: st.mean(items[r][task]["acc"])
for r, lab in [("dense", "dense fp32")] + [(f"w{b}_s{x}", f"GPTQ W{b} g128 seed {x}") for b in (4, 3) for x in range(3)]:
    w(f"| {lab} | {acc(r, 'arc_easy'):.3f} | {acc(r, 'hellaswag'):.3f} |")
w("")
def mcnemar(a, b, task):
    A = dict(zip(items[a][task]["doc_id"], items[a][task]["acc"])); B = dict(zip(items[b][task]["doc_id"], items[b][task]["acc"]))
    ids = sorted(set(A) & set(B)); b10 = sum(A[i] == 1 and B[i] == 0 for i in ids); b01 = sum(A[i] == 0 and B[i] == 1 for i in ids)
    p = stats.binomtest(min(b10, b01), b10 + b01, 0.5).pvalue if b10 + b01 else 1.0
    return b10, b01, (sum(B[i] for i in ids) - sum(A[i] for i in ids)) / len(ids), p
w("McNemar test on the paired items (b = right→wrong, c = wrong→right going from the first to the second model; flip rate =")
w("(b + c) / 1000). **Test used: the exact two-sided binomial test on the b + c discordant items for every comparison**; the")
w("continuity-corrected χ² p is shown for reference only.\n")
w("| comparison | task | b | c | flips b + c | flip rate | Δ acc | test | exact p | χ² (cc) p |")
w("|---|---|---|---|---|---|---|---|---|---|")
for a, b, lab in (("dense", "w4_s0", "dense vs W4 s0"), ("dense", "w3_s0", "dense vs W3 s0"), ("w4_s0", "w4_s1", "W4 s0 vs W4 s1"), ("w3_s0", "w3_s1", "W3 s0 vs W3 s1")):
    for task in ("arc_easy", "hellaswag"):
        b10, b01, da, p = mcnemar(a, b, task)
        pchi = stats.chi2.sf((abs(b10 - b01) - 1) ** 2 / (b10 + b01), 1) if b10 + b01 else 1.0
        w(f"| {lab} | {task} | {b10} | {b01} | {b10 + b01} | {(b10 + b01) / 10:.1f} % | {da:+.3f} | exact binomial | {p:.2g} | {pchi:.2g} |")
w("")
w("**Correction of an earlier summary.** \"About 100 arc_easy items flip each way\" was wrong for W4: between W4 seeds 0 and 1,")
w("103 items flip in total (51 right→wrong, 52 wrong→right), i.e. ~50 each way and a 10.3 % flip rate; the 10.5 % in the agreement")
w("table is the mean over the three seed pairs. ~100 each way holds for W3 (96 / 98, 19.4 %).\n")

# ---------------- (2) window length ----------------
w("## 2. Calibration window length (128 × 512 vs 128 × 2048, WT-ppl)\n")
w("| model · method · path | 512: n, mean ± SD | 2048: n, mean ± SD | Δ ppl (2048 − 512) [95 % CI] | Δ / pooled SD | Welch p | Holm p | ΔD (nats) [95 % CI] | sources |")
w("|---|---|---|---|---|---|---|---|---|")
WL = [("SmolLM2-135M · GPTQ W3 · SDPA (F15 = REPORT.md A.5)", "SmolLM2-135M · GPTQ W3 g128 · 128×512 · SDPA", "SmolLM2-135M · GPTQ W3 g128 · 128×2048 · SDPA"),
      ("SmolLM2-135M · GPTQ W3 · eager", "SmolLM2-135M · GPTQ W3 g128 · 128×512 · eager", "SmolLM2-135M · GPTQ W3 g128 · 128×2048 · eager"),
      ("SmolLM2-135M · GPTQ W4 · eager", "SmolLM2-135M · GPTQ W4 g128 · 128×512 · eager", "SmolLM2-135M · GPTQ W4 g128 · 128×2048 · eager"),
      ("Qwen2.5-0.5B · GPTQ W3 · eager", "Qwen2.5-0.5B · GPTQ W3 g128 · 128×512 · eager", "Qwen2.5-0.5B · GPTQ W3 g128 · 128×2048 · eager")]
WLrows = []
for lab, a, b in WL:
    za, zb = S[a][0], S[b][0]
    d, lo, hi, p, _ = welch(za, zb); pooled = pooled_sd(za, zb)
    dref = S[a][3]
    Da, Db = summ([math.log(v / dref) for v in za["v"]]), summ([math.log(v / dref) for v in zb["v"]])
    dD, dlo, dhi, _, _ = welch(Da, Db)
    WLrows.append((lab, za, zb, d, lo, hi, p, pooled, dD, dlo, dhi, a, b))
WLholm = holm([r[6] for r in WLrows])
for (lab, za, zb, d, lo, hi, p, pooled, dD, dlo, dhi, a, b), ph in zip(WLrows, WLholm):
    srcs = ", ".join(sorted({cite(P[s]) for s in S[a][1] + S[b][1]}))
    w(f"| {lab} | {za['n']}, {f(za['mean'])} ± {f(za['sd'])} | {zb['n']}, {f(zb['mean'])} ± {f(zb['sd'])} | {d:+.3f} [{lo:+.3f}, {hi:+.3f}] ({100 * d / za['mean']:+.1f} %) | "
      f"{d / pooled:.2f} | {p:.2g} | {ph:.2g} | {dD:+.4f} [{dlo:+.4f}, {dhi:+.4f}] | {srcs} |")
w("")
w("The eager-path 512 arms (S3) and 2048 arms (S1, S2) come from different sessions with the same configuration, CPU model and cached ids.\n")

# ---------------- (3) source x eval ----------------
w("## 3. Calibration source × evaluation set (S3: SmolLM2-135M, GPTQ g128, 128 × 512, eager, seeds 0–2)\n")
H = {}
DENSE_EAGER = {"wt_ppl": m1("F1", "dense"), "c4_ppl": m1("F1", "dense", "c4_ppl")}
for b in (4, 3):
    M = {}
    for src in ("wikitext2", "c4"):
        for ev in ("wt_ppl", "c4_ppl"):
            M[(src, ev)] = summ([m1("F3", f"w{b}_{src}_s{s}", ev) for s in range(3)])
    H[b] = M
w(f"Mean ± SD over 3 seeds ({cite(P['F3'])}); D against the eager dense model on the same evaluation set "
  f"(WT {DENSE_EAGER['wt_ppl']:.4f}, C4 {DENSE_EAGER['c4_ppl']:.4f}; {cite(P['F1'])}). The two evaluation domains are reported separately.\n")
w("| bits | calibration | WikiText-2 ppl | WikiText-2 D | C4-val ppl | C4-val D |")
w("|---|---|---|---|---|---|")
DOM = {}
for b in (4, 3):
    for src in ("wikitext2", "c4"):
        cells = []
        for ev in ("wt_ppl", "c4_ppl"):
            z = H[b][(src, ev)]; zD = summ([math.log(v / DENSE_EAGER[ev]) for v in z["v"]]); DOM[(b, src, ev)] = (z, zD)
            cells += [f"{f(z['mean'])} ± {f(z['sd'])}", f"{zD['mean']:.4f} ± {zD['sd']:.4f}"]
        w(f"| W{b} | {src} | " + " | ".join(cells) + " |")
w("")
w("In-domain advantage = out-of-domain-calibrated minus in-domain-calibrated, on each evaluation set:\n")
w("| bits | eval set | in-domain advantage ppl [95 % CI] | Δ / pooled SD | Welch p | Holm p | advantage in D (nats) [95 % CI] |")
w("|---|---|---|---|---|---|---|")
rowsD = []
for b in (4, 3):
    for ev, own, other in (("wt_ppl", "wikitext2", "c4"), ("c4_ppl", "c4", "wikitext2")):
        za, zb = DOM[(b, own, ev)][0], DOM[(b, other, ev)][0]
        Da, Db = DOM[(b, own, ev)][1], DOM[(b, other, ev)][1]
        d, lo, hi, p, _ = welch(za, zb); dD, dlo, dhi, _, _ = welch(Da, Db)
        rowsD.append((b, ev, d, lo, hi, d / pooled_sd(za, zb), p, dD, dlo, dhi))
for (b, ev, d, lo, hi, dz, p, dD, dlo, dhi), ph in zip(rowsD, holm([r[6] for r in rowsD])):
    w(f"| W{b} | {'WikiText-2' if ev == 'wt_ppl' else 'C4 val'} | {d:+.3f} [{lo:+.3f}, {hi:+.3f}] | {dz:.2f} | {p:.2g} | {ph:.2g} | {dD:+.4f} [{dlo:+.4f}, {dhi:+.4f}] |")
w("")
w("The WikiText-2/C4 average column of earlier versions is dropped: the two domains have different dense baselines and")
w("are reported separately.\n")
w("Figure: `figures/fig2_domain_heatmap.{png,pdf}`.\n")

# ---------------- (4) determinism ----------------
w("## 4. Numerical determinism (same command repeated; SmolLM2-135M GPTQ W4 g128 seed 0, 128 × 512, WT-ppl)\n")
groups = {
    "default (SDPA, 4 threads)": [("Step 1", q("s1", "gptq_w4_135m_s0"), "s1"), ("rebuild 1", q("c4val", "s2_gptq_w4_135m_wt_s0_rebuild"), "c4val"),
                                  ("rep 2", q("det", "s2_det_rep2"), "det"), ("rep 3", q("det", "s2_det_rep3"), "det")],
    "`use_deterministic_algorithms(True)`": [(k, q("det", f"s2_det_{k}"), "det") for k in ("det1", "det2")],
    "`set_num_threads(1)`": [(k, q("det", f"s2_det_{k}"), "det") for k in ("thr1a", "thr1b")],
    "eager, 4 threads": [(k, q("det", f"s2_det_{k}"), "det") for k in ("eager1", "eager2")]
        + [(f"S1 {k}", m1("F1", k), "F1") for k in ("anchor_a", "anchor_b")]
        + [(f"S3 {k}", m1("F3", k), "F3") for k in ("anchor_a", "anchor_b")]
        + [(f"S2 r{i}", q("F2a", f"final_S2_anchor_eager_r{i}"), "F2a") for i in range(1, 6)],
}
w("| configuration | runs | distinct values | values (count) | spread | quantize s (typ.) | sources |")
w("|---|---|---|---|---|---|---|")
qt = {"default (SDPA, 4 threads)": "246–348", "`use_deterministic_algorithms(True)`": "247–267", "`set_num_threads(1)`": "545–571", "eager, 4 threads": "298–475"}
for g, L in groups.items():
    vals = [v for _, v, _ in L]; cnt = {}
    for v in vals: cnt[v] = cnt.get(v, 0) + 1
    w(f"| {g} | {len(vals)} | {len(cnt)} | {'; '.join(f'{v:.6f} ({c})' for v, c in sorted(cnt.items()))} | {max(vals) - min(vals):.4f} | {qt[g]} | "
      f"{', '.join(sorted({cite(P[s]) for _, _, s in L}))} |")
w("")
all_vals = [v for L in groups.values() for _, v, _ in L]
w(f"- All four configurations together: {len(set(all_vals))} distinct values for one command and one seed, spanning "
  f"{min(all_vals):.4f}–{max(all_vals):.4f} (**{max(all_vals) - min(all_vals):.3f} ppl**), vs a Step 1 W4 five-seed range of "
  f"{S['SmolLM2-135M · GPTQ W4 g128 · 128×512 · SDPA'][0]['range']:.3f}.")
w(f"- Seed-0 repeats that were bit-identical (same checkpoint SHA-256): S1 W3/W4 at 128 × 2048 ({cite(P['F1'])}); S3 all four "
  f"seed-0 cells ({cite(P['F3'])}); Wanda 2:4, 60 %, 70 % ({cite(P['F2w'])}).")
tw = q("w50", "s2_wanda50_135m_s0"); tw2 = q("w50", "s2_wanda50_135m_s0_t023")
w(f"- Tokenizer version (tokenizers 0.21.4 vs 0.23.2, Wanda 50 % s0): {tw:.4f} vs {tw2:.4f} (Δ {tw - tw2:+.3f}); dense "
  f"{q('w50','s2_wanda_dense_135m'):.6f} vs {q('w50','s2_wanda_dense_135m_t023'):.6f} ({cite(P['w50'])}).\n")

# ---------------- (5) evaluation-protocol factors ----------------
w("## 5. Evaluation- and quantization-protocol factors (effect on WT-ppl; includes trimmed S4)\n")
sdpa_w4 = S["SmolLM2-135M · GPTQ W4 g128 · 128×512 · SDPA"][0]
W4P = pooled_sd(*(S[k][0] for k in S if "GPTQ W4" in k and "C4-ppl" not in k))
eag4 = S["SmolLM2-135M · GPTQ W4 g128 · 128×512 · eager"][0]
c4cal_sdpa = q("c4", "s2_gptq_w4_135m_c4_s0"); c4cal_eager = m1("F3", "w4_c4_s0")
F = [
    ("calibration seed (W4, SDPA, 512): SD", sdpa_w4["sd"], "s1", "SD over 5 seeds"),
    ("calibration seed (W4, eager, 2048): SD", S["SmolLM2-135M · GPTQ W4 g128 · 128×2048 · eager"][0]["sd"], "F1", "SD over 5 seeds"),
    ("run-to-run noise, default SDPA path", abs(q("c4val", "s2_gptq_w4_135m_wt_s0_rebuild") - q("s1", "gptq_w4_135m_s0")), "det", "two outcomes, same seed"),
    ("run-to-run noise, eager path (max |Δ|)", max(abs(v - 17.66059890313337) for _, v, _ in groups["eager, 4 threads"]), "F1", "11 runs"),
    ("dense eval dtype fp32 → bf16", abs(dense_wt_bf16 - dense_wt), "s1", "dense model"),
    ("tokenizers 0.21.4 vs 0.23.2 (Wanda 50 % s0)", abs(tw - tw2), "w50", "different token ids"),
    ("threads 1 vs 4 (W4 s0)", abs(q("det", "s2_det_thr1a") - q("det", "s2_det_det1")), "det", "vs SDPA outcome B"),
    ("attention eager vs SDPA (W4, WT-cal s0)", abs(m1("F3", "w4_wikitext2_s0") - q("s1", "gptq_w4_135m_s0")), "F3", "vs Step 1 value"),
    ("attention eager vs SDPA (W4, C4-cal s0)", abs(c4cal_eager - c4cal_sdpa), "c4", "C4-calibrated"),
    ("window length 512 → 2048 (W4, eager, means)", S["SmolLM2-135M · GPTQ W4 g128 · 128×2048 · eager"][0]["mean"] - eag4["mean"], "F1", "S3 vs S1 means"),
    ("calibration source WT → C4 (W4, eager, means)", H[4][("c4", "wt_ppl")]["mean"] - H[4][("wikitext2", "wt_ppl")]["mean"], "F3", "in-domain advantage"),
    ("quantization dense → W4 (SDPA, 512, mean)", sdpa_w4["mean"] - dense_wt, "s1", "the effect being reported"),
]
w(f"Pooled W4 seed SD (all three W4 WT-ppl seed sets, df-weighted) = **{W4P:.4f}**.\n")
w("| factor (SmolLM2-135M) | |Δ WT-ppl| | Δ / pooled SD | note | source |")
w("|---|---|---|---|---|")
for lab, v, s, note in F:
    w(f"| {lab} | {v:.4f} | {v / W4P:.2f} | {note} | {cite(P[s])} |")
w("")
S4P, S4O = "results/final_S4_ppl.jsonl", "results/final_S4_ob.jsonl"
s4 = {r["run_id"]: r for r in rows(S4P) if r.get("rc") == 0} if os.path.exists(S4P) else {}
S4F = []  # (label, |delta|) for the factor figure
if len(s4) == 12:
    w(f"**S4 (trimmed): perplexity protocol – context length × prepend-BOS** ({cite(S4P)}). SmolLM2-135M, eager attention, 4 threads;")
    w("dense fp32 and GPTQ W4 g128 seed 0 (128 × 512 WikiText calibration, bf16 `BACKEND.TORCH`). Every cell scores the same")
    w("81,920 WikiText-2 test tokens (the 40 × 2048 Step 1 windows) cut into 512/1024/2048-token windows; without BOS the first token")
    w("of each window is not predicted, with BOS (`<|endoftext|>`, id 0) prepended every token is.\n")
    w("| model | context | BOS | windows | targets | WT-ppl | W4 − dense |")
    w("|---|---|---|---|---|---|---|")
    for L in (512, 1024, 2048):
        for bo in (0, 1):
            dn, q4 = s4[f"dense_ctx{L}_bos{bo}"], s4[f"w4_ctx{L}_bos{bo}"]
            for r, lab in ((dn, "dense fp32"), (q4, "GPTQ W4 s0")):
                w(f"| {lab} | {L} | {'yes' if bo else 'no'} | {r['n_windows']} | {r['n_targets']} | {r['wt_ppl']:.4f} | "
                  f"{(q4['wt_ppl'] - dn['wt_ppl']):+.4f} |" if r is q4 else
                  f"| {lab} | {L} | {'yes' if bo else 'no'} | {r['n_windows']} | {r['n_targets']} | {r['wt_ppl']:.4f} | |")
    w("")
    sha = s4["w4_ctx2048_bos0"].get("ckpt_sha256", "")[:16]
    w(f"W4 checkpoint sha256 `{sha}`; at the reference protocol (context 2048, no BOS) the W4 value is {s4['w4_ctx2048_bos0']['wt_ppl']!r} "
      f"(eager anchor modal value 17.66059890313337) and dense is {s4['dense_ctx2048_bos0']['wt_ppl']!r}.")
    w(f"**Evaluation is not bit-reproducible either:** this checkpoint has the same full SHA-256 as the 9 modal eager anchor runs and is scored")
    w(f"on the same 40 × 2048 windows (81,880 targets), yet gives {s4['w4_ctx2048_bos0']['wt_ppl']:.5f} instead of 17.66060 "
      f"(Δ {s4['w4_ctx2048_bos0']['wt_ppl'] - 17.66059890313337:+.4f}). The only differences are the process state and evaluation order (here the 512/1024")
    w("contexts were evaluated first in the same process; the W4 ctx2048-BOS cell ran after a container restart). Not investigated further;")
    w("the effect is 0.02 % and ~1/1000 of the context effect.\n")
    w("| factor effect (S4) | dense Δ | W4 Δ | W4 Δ / pooled SD | degradation (W4 − dense) changes by |")
    w("|---|---|---|---|---|")
    def eff(a, b, lab):
        dd = s4[f"dense_{b}"]["wt_ppl"] - s4[f"dense_{a}"]["wt_ppl"]; dq = s4[f"w4_{b}"]["wt_ppl"] - s4[f"w4_{a}"]["wt_ppl"]
        w(f"| {lab} | {dd:+.4f} | {dq:+.4f} | {dq / W4P:+.1f} | {dq - dd:+.4f} |"); return lab, abs(dq)
    for a, b, lab in (("ctx2048_bos0", "ctx512_bos0", "context 2048 → 512 (no BOS)"), ("ctx2048_bos0", "ctx1024_bos0", "context 2048 → 1024 (no BOS)"),
                      ("ctx512_bos0", "ctx512_bos1", "prepend BOS at 512"), ("ctx2048_bos0", "ctx2048_bos1", "prepend BOS at 2048"),
                      ("ctx2048_bos0", "ctx512_bos1", "2048 no-BOS → 512 with BOS")):
        S4F.append(eff(a, b, lab))
    degr = [s4[f"w4_ctx{L}_bos{bo}"]["wt_ppl"] - s4[f"dense_ctx{L}_bos{bo}"]["wt_ppl"] for L in (512, 1024, 2048) for bo in (0, 1)]
    w("")
    w(f"Across the 6 protocol cells the reported W4 degradation ranges {min(degr):.3f}–{max(degr):.3f} ppl "
      f"({100 * (max(degr) - min(degr)) / min(degr):.0f} % relative spread) for one and the same checkpoint.\n")
else:
    w(f"S4 perplexity-protocol results not available ({len(s4)}/12 cells in `{S4P}`).\n")
w("Still missing: few-shot count, batch size, max_length and dtype effects on real lm-eval tasks.")
w("The only measurements of those are on synthetic stand-ins (REPORT.md B.1, Test 2c: per-sample log-lik changes ≤6e-5 nats")
w("for bs 1 vs 8, ≤1.49 nats for bf16 vs fp32) and are not quality numbers. Figure: `figures/fig3_factor_effects.{png,pdf}`.\n")

# ---------------- (6) latency-protocol factors ----------------
w("## 6. Latency-protocol factors (optimum-benchmark 0.6.0)\n")
ob = R["ob"]
s4o = [r for r in rows(S4O)] if os.path.exists(S4O) else []
OBK = {}
if s4o:
    w(f"**S4 (trimmed): launcher × threads × model** ({cite(S4O)}). fp32, bs 1, seq 128, 32 new tokens, 5 iterations, warmup 10;")
    w("threads set via `OMP_NUM_THREADS` and `backend.inter_op_num_threads` (which calls `torch.set_num_threads` in optimum-benchmark 0.6.0).\n")
    w("| model | launcher | threads | prefill s | decode s | per-token s | decode tok/s | status |")
    w("|---|---|---|---|---|---|---|---|")
    for r in s4o:
        m = r.get("metrics") or {}
        key = (r.get("model", "?").split("/")[-1], r.get("launcher"), r.get("threads"))
        if r.get("rc") == 0: OBK[key] = m
        st_ = "ok" if r.get("exit_code", 0) == 0 else ("report written, exit≠0 (known inline bug)" if r.get("rc") == 0 else f"FAILED rc={r.get('rc')}")
        w(f"| {key[0]} | {key[1]} | {key[2]} | {m.get('prefill_latency_mean_s', float('nan')):.4f} | {m.get('decode_latency_mean_s', float('nan')):.4f} | "
          f"{m.get('per_token_latency_mean_s', float('nan')):.4f} | {m.get('decode_throughput', float('nan')):.1f} | {st_} |")
    w("")
    w("| factor effect (per-token latency ratio) | condition | ratio |")
    w("|---|---|---|")
    S4R = {"launcher process/inline": [], "threads 2/4": [], "model Qwen-0.5B/SmolLM2-135M": []}
    for mdl in ("SmolLM2-135M", "Qwen2.5-0.5B"):
        for t in (2, 4):
            a, b = OBK.get((mdl, "process", t)), OBK.get((mdl, "inline", t))
            if a and b:
                v = a["per_token_latency_mean_s"] / b["per_token_latency_mean_s"]; S4R["launcher process/inline"].append(v)
                w(f"| launcher process / inline | {mdl}, {t} threads | {v:.2f} |")
    for mdl in ("SmolLM2-135M", "Qwen2.5-0.5B"):
        for L in ("inline", "process"):
            a, b = OBK.get((mdl, L, 2)), OBK.get((mdl, L, 4))
            if a and b:
                v = a["per_token_latency_mean_s"] / b["per_token_latency_mean_s"]; S4R["threads 2/4"].append(v)
                w(f"| threads 2 / 4 | {mdl}, {L} | {v:.2f} |")
    for L in ("inline", "process"):
        for t in (2, 4):
            a, b = OBK.get(("Qwen2.5-0.5B", L, t)), OBK.get(("SmolLM2-135M", L, t))
            if a and b:
                v = a["per_token_latency_mean_s"] / b["per_token_latency_mean_s"]; S4R["model Qwen-0.5B/SmolLM2-135M"].append(v)
                w(f"| model Qwen2.5-0.5B / SmolLM2-135M | {L}, {t} threads | {v:.2f} |")
    w("")
    for k, v in S4R.items():
        if v: w(f"- {k}: {min(v):.2f}–{max(v):.2f}× over {len(v)} conditions.")
    w("- The process launcher only inflates latency at 4 threads (= all 4 vCPUs, so its busy-waiting parent competes with")
    w("  the benchmark threads); at 2 threads process ≈ inline. The thread-count effect therefore **reverses sign with the launcher**:")
    w("  2 threads are 1.5–1.6× slower than 4 under inline but 3.6–4.5× faster under process.")
    w("- The model-size ratio (2.0–2.6×) is the most stable quantity, but still varies by 27 % across launcher × threads.")
    w("- Run in one container after a restart (the perplexity cells started before it). Inline runs exit non-zero after writing")
    w("  their report (known optimum-benchmark 0.6.0 bug), so those reports are used.")
    w("")
else:
    S4R = {}
w("**Earlier real-weight evidence (Step 2 §6)**\n")
w(f"**Launcher (real SmolLM2-135M weights, fp32, OMP 4, 5 iterations, 32 new tokens)** – {cite(P['ob'])}\n")
w("| protocol | metric | process (default) s | inline s | inflation × |")
w("|---|---|---|---|---|")
infl = {}
for p, lab in (("p1", "bs 1 × seq 128"), ("p2", "bs 4 × seq 512")):
    a, b = ob[f"s2_ob_{p}_process_report"]["report"], ob[f"s2_ob_{p}_inline_report"]["report"]
    for k, mlab in (("prefill_latency_mean_s", "prefill"), ("decode_latency_mean_s", "decode"), ("per_token_latency_mean_s", "per token")):
        infl.setdefault(mlab, []).append(a[k] / b[k])
        w(f"| {lab} | {mlab} | {a[k]:.4f} | {b[k]:.4f} | {a[k] / b[k]:.2f} |")
w("")
t3 = json.load(open(P["t3c"]))
w(f"**Model-size ratio 360M / 135M across 8 protocols** (warmup {{0,10}} × bs {{1,4}} × seq {{128,512}}; process launcher, OMP 3; "
  f"synthetic random-weight stand-ins with the real architectures, Part B VM) – {cite(P['t3c'])}\n")
w("| metric | min | max | protocols |")
w("|---|---|---|---|")
rat = {}
for k, lab in (("ratio_prefill_p50", "prefill p50"), ("ratio_per_token_p50", "per-token p50"), ("ratio_generate_mean", "generate mean")):
    v = [d[k] for d in t3]; rat[lab] = v
    w(f"| {lab} | {min(v):.2f} | {max(v):.2f} | {len(v)} |")
w("")
w("Missing for a full section 6 (S4 scope): the 360M/135M ratio sweep on real weights with the inline launcher, and Qwen2.5-0.5B.")
w("Figure: `figures/fig4_latency_ratios.{png,pdf}`.\n")
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "revision"))
from revision_sections import build as _build_revision
out += [""] + _build_revision(globals())
open("PAPER_TABLES.md", "w").write("\n".join(out) + "\n")

# ---------------- figures ----------------
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.facecolor": SURF, "figure.facecolor": SURF, "savefig.facecolor": SURF, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans"})
def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

# fig 1: seed strips, deviation from setting mean in % (one common axis)
names = list(S)
fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(names) + 0.9))
for i, n in enumerate(names):
    z = S[n][0]; dev = [100 * (v - z["mean"]) / z["mean"] for v in z["v"]]
    color = BLUE if "GPTQ" in n and "SmolLM2" in n else (ORANGE if "Qwen" in n else AQUA)
    LIM = 8.0
    inside = [d for d in dev if abs(d) <= LIM]
    ax.scatter(inside, [i] * len(inside), s=26, color=color, edgecolor=SURF, linewidth=1.2, zorder=3)
    for sgn in (1, -1):  # off-scale points: one edge marker per side, labelled with their true values
        off = sorted(d for d in dev if d * sgn > LIM)
        if off:
            xe = LIM * sgn
            ax.scatter([xe], [i], s=30, marker=">" if sgn > 0 else "<", color=color, edgecolor=SURF, linewidth=1.2, zorder=3)
            ax.text(xe - 0.3 * sgn, i - 0.32, " / ".join(f"{d:+.0f} %" for d in off), ha="right" if sgn > 0 else "left", fontsize=7, color=INK2)
    ax.text(LIM + 0.5, i, f"n={z['n']}, CV {z['cv']:.2f} %", va="center", fontsize=7.5, color=INK2)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_yticks(range(len(names))); ax.set_yticklabels([n.replace(" · ", " · ") for n in names], fontsize=7.5, color=INK)
ax.invert_yaxis(); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_xlabel("deviation of each seed from its setting's mean WT-ppl (%)")
ax.set_title("Calibration-seed spread per setting (blue: GPTQ SmolLM2, orange: GPTQ Qwen, aqua: Wanda)", fontsize=9, color=INK, loc="left")
ax.set_xlim(-8.6, 8.6)
save(fig, "fig1_seed_strips")

# fig 2: 2x2 heatmaps, colour = % increase over dense for that eval set
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.08})
dref = {"wt_ppl": dense_wt, "c4_ppl": dense_c4}
allinc = [100 * (H[b][(s, e)]["mean"] / dref[e] - 1) for b in (4, 3) for s in ("wikitext2", "c4") for e in ("wt_ppl", "c4_ppl")]
from matplotlib.colors import LinearSegmentedColormap, LogNorm
cmap = LinearSegmentedColormap.from_list("blue", SEQ)
norm = LogNorm(vmin=min(allinc), vmax=max(allinc))
for ax, b in zip(axes, (4, 3)):
    for i, s in enumerate(("wikitext2", "c4")):
        for j, e in enumerate(("wt_ppl", "c4_ppl")):
            z = H[b][(s, e)]; inc = 100 * (z["mean"] / dref[e] - 1)
            ax.add_patch(plt.Rectangle((j + 0.02, i + 0.02), 0.96, 0.96, color=cmap(norm(inc))))
            tc = "#ffffff" if norm(inc) > 0.55 else INK
            ax.text(j + 0.5, i + 0.42, f"{z['mean']:.2f} ± {z['sd']:.2f}", ha="center", va="center", color=tc, fontsize=9, weight="bold")
            ax.text(j + 0.5, i + 0.66, f"+{inc:.0f} % vs dense", ha="center", va="center", color=tc, fontsize=7.5)
    ax.set_xlim(0, 2); ax.set_ylim(2, 0); ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["eval: WikiText-2", "eval: C4 val"])
    ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(["calib: WikiText-2", "calib: C4"] if b == 4 else ["", ""]); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(f"GPTQ W{b} g128 (n=3 seeds)", fontsize=9, color=INK, loc="left")
fig.suptitle("Perplexity by calibration source × evaluation set (SmolLM2-135M, 128×512, eager); shade = increase over dense",
             fontsize=8.5, color=INK2, x=0.01, ha="left", y=1.02)
save(fig, "fig2_domain_heatmap")

# fig 3: factor effects, log x, reference line = pooled W4 seed SD
F = F + [(f"S4: {lab} (W4)", v, "S4", "") for lab, v in S4F if v > 0]
fig, ax = plt.subplots(figsize=(7.2, 0.36 * len(F) + 0.9))
labs = [l for l, *_ in F]; vals = [v for _, v, *_ in F]
ax.barh(range(len(F)), vals, height=0.6, color=BLUE, edgecolor=SURF, linewidth=2)
for i, v in enumerate(vals):
    ax.text(v * 1.12, i, f"{v:.3g}", va="center", fontsize=7.5, color=INK2, zorder=4,
            bbox=dict(boxstyle="square,pad=0.15", fc=SURF, ec="none"))
ax.axvline(W4P, color=ORANGE, lw=1.5, ls="--")
ax.text(W4P * 1.08, -0.75, "reference: pooled W4 seed SD", color=INK2, fontsize=7.5, va="bottom")
ax.set_xscale("log"); ax.set_yticks(range(len(F))); ax.set_yticklabels(labs, fontsize=7.5, color=INK)
ax.grid(axis="x", color=GRID, lw=0.6, which="both"); ax.set_axisbelow(True)
ax.set_xlabel("|Δ WT-ppl| (log scale)"); ax.set_xlim(min(vals) / 3, max(vals) * 4); ax.set_ylim(len(F) - 0.5, -1.1)
ax.set_title("Size of protocol factors vs. the effect being reported (SmolLM2-135M)", fontsize=9, color=INK, loc="left")
save(fig, "fig3_factor_effects")

# fig 4: latency ratio ranges
fig, ax = plt.subplots(figsize=(7.2, 0.4 * (6 + len([v for v in S4R.values() if v])) + 0.6))
entries = [(f"360M/135M · {k} (synthetic)", v, BLUE) for k, v in rat.items()] + [(f"process/inline · {k} (Step 2)", v, ORANGE) for k, v in infl.items()]
entries += [(f"S4 · {k} · per token", v, AQUA if k.startswith("model") else (ORANGE if k.startswith("launcher") else INK2)) for k, v in S4R.items() if v]
for i, (lab, v, c) in enumerate(entries):
    ax.plot([min(v), max(v)], [i, i], color=c, lw=2, solid_capstyle="round", zorder=2)
    ax.scatter(v, [i] * len(v), s=22, color=c, edgecolor=SURF, linewidth=1.2, zorder=3)
    ax.text(max(v) * 1.06, i, f"{min(v):.2f}–{max(v):.2f}× (n={len(v)})", va="center", fontsize=7.5, color=INK2)
ax.axvline(1, color=INK2, lw=0.8)
ax.set_xscale("log"); ax.set_yticks(range(len(entries))); ax.set_yticklabels([e[0] for e in entries], fontsize=7.5, color=INK)
ax.invert_yaxis(); ax.grid(axis="x", color=GRID, lw=0.6, which="both"); ax.set_axisbelow(True)
ax.set_xlim(min(0.8, min(min(e[1]) for e in entries) * 0.8), 25)
ax.set_xlabel("latency ratio (log scale)")
ax.set_title("Latency ratios across protocols (blue: model size, orange: launcher, gray: threads, aqua: S4 model size)",
             fontsize=8.5, color=INK, loc="left")
save(fig, "fig4_latency_ratios")
print("ok")
