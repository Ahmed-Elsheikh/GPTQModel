"""Writes feasibility/REPORT_final_S2.md from results/final_S2_*.jsonl (tables only; prose sections are kept between
'<!-- MANUAL START -->' and '<!-- MANUAL END -->' markers if present). Idempotent; rerun at any time."""
import json, math, os, re, statistics as st, subprocess
from scipy.stats import chi2, ttest_ind
H = os.path.dirname(os.path.abspath(__file__))
def rows(name):
    p = os.path.join(H, "results", f"final_S2_{name}.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
def ok(rs, pat, key):
    out = []
    for r in rs:
        if re.fullmatch(pat, r["run_id"]) and r["rc"] == 0 and (r.get("result") or {}).get(key) is not None:
            out.append(r)
    return out
def stats(p):
    n = len(p); m = st.mean(p)
    if n < 2: return dict(n=n, mean=m, sd=None, cv=None, mn=min(p), mx=max(p), lo=None, hi=None)
    sd = st.stdev(p); lo, hi = (sd * math.sqrt((n - 1) / chi2.ppf(q, n - 1)) for q in (0.975, 0.025))
    return dict(n=n, mean=m, sd=sd, cv=100 * sd / m, mn=min(p), mx=max(p), lo=lo, hi=hi)
def fmt(x, d=4): return "–" if x is None else f"{x:.{d}f}"
def srow(label, s, extra=""):
    ci = "–" if s["lo"] is None else "[%.4f, %.4f]" % (s["lo"], s["hi"])
    cv = "–" if s["cv"] is None else "%.2f %%" % s["cv"]
    return (f"| {label} | {s['n']} | {fmt(s['mean'])} | {fmt(s['sd'])} | {cv} | "
            f"{fmt(s['mn'])} | {fmt(s['mx'])} | {ci} |{extra}")
def ph(r, k):
    x = (r["result"].get("phases") or {}).get(k); return x.get("wall_s") if isinstance(x, dict) else None
HDR = "| Setting | n | Mean ppl | SD | CV | Min | Max | 95 % CI for SD (χ²) |\n|---|---|---|---|---|---|---|---|"

A = rows("anchor"); R = rows("runs"); W = rows("wanda")
L = []
L.append("# Final S2 – Qwen2.5-0.5B GPTQ seed and window-length study (with confirmed determinism)\n")
L.append("Session S2. Raw records: `results/final_S2_anchor.jsonl`, `results/final_S2_runs.jsonl`, `results/final_S2_wanda.jsonl` "
         "(one JSON line per run: model, method, bits/sparsity, group size, calibration source/count/length, seed, configuration, "
         "CPU model + flags, package versions, quantize/eval wall time, peak RSS, metrics, checkpoint SHA-256). "
         "Scripts: `chain_final_S2.sh` (resumable chain), `final_S2_run.py` (configuration wrapper around `quant_eval.py`), "
         "`final_S2_check.py`, `final_S2_report.py`. Token ids: `cache_tok/` only (WT-ppl fingerprint `7da3b34bdebd1923`).\n")
# Determinism
L.append("## Determinism (confirmed)\n")
eag = ok(A, r"final_S2_anchor_eager_r\d", "ppl_quant"); t1 = ok(A, r"final_S2_anchor_eager_thr1_r\d", "ppl_quant")
allA = [r for r in A]
L.append("Anchor: SmolLM2-135M, GPTQ W4 g128, seed 0, 128 × 512 WikiText-2 train calibration (fingerprint `07db06dabcb03c98`), "
         "fp32 quantize, bf16 TorchLinear (BACKEND.TORCH) eval, WT-ppl. Bit-identical = same perplexity (exact float repr) "
         "**and** same checkpoint SHA-256.\n")
L.append("| Run | Configuration | rc | WT-ppl (repr) | Checkpoint sha256 (16) | Quantize s | Eval s | Wall s | Peak RSS GB |\n|---|---|---|---|---|---|---|---|---|")
for r in allA:
    x = r.get("result") or {}; c = x.get("config") or {}
    L.append(f"| {r['run_id']} | {c.get('final_cfg','?')} (attn {c.get('attn_implementation','?')}, threads {c.get('torch_threads_end','?')}) | {r['rc']} | "
             f"{repr(x.get('ppl_quant'))} | `{(c.get('ckpt_sha256') or '–')[:16]}` | {ph(r,'quantize')} | {ph(r,'eval_quant')} | {r['wall_s']} | {r['peak_rss_gb']} |")
v = {}
for name, grp in (("eager", eag), ("eager_threads1", t1)):
    if grp:
        v[name] = dict(n=len(grp), ppl=sorted({repr(r['result']['ppl_quant']) for r in grp}),
                       sha=sorted({r['result']['config'].get('ckpt_sha256') for r in grp}),
                       q=st.mean(ph(r, 'quantize') for r in grp), wall=st.mean(r['wall_s'] for r in grp))
conf = None
if "eager" in v and v["eager"]["n"] >= 5 and len(v["eager"]["ppl"]) == 1 and len(v["eager"]["sha"]) == 1: conf = "eager"
elif "eager_threads1" in v and v["eager_threads1"]["n"] >= 3 and len(v["eager_threads1"]["ppl"]) == 1 and len(v["eager_threads1"]["sha"]) == 1: conf = "eager_threads1"
L.append("")
for name, x in v.items():
    L.append(f"- **{name}**: {x['n']} successful runs, distinct ppl values {x['ppl']}, distinct checkpoints {len(x['sha'])}; "
             f"mean quantize {x['q']:.0f} s, mean wall {x['wall']:.0f} s.")
SC = "17.66059890313337"
if conf:
    val = v[conf]["ppl"][0]
    desc = 'attn_implementation="eager", default 4 threads' if conf == "eager" else 'attn_implementation="eager" + set_num_threads(1)'
    L.append(f"\n**Confirmed configuration: `{conf}`** ({desc}), "
             f"BACKEND.TORCH, fp32 quantize, bf16 eval. **Anchor value: {val}.**")
    L.append(f"- Session C's eager value (REPORT_step2.md §8, other container): {SC}. "
             + ("**Equal (bit-for-bit in ppl)** → the configuration is reproducible across containers." if val == SC and conf == "eager"
                else "Not equal (or a different configuration) → the value is container- or configuration-specific."))
    L.append(f"- Quantize-time cost vs the default (SDPA, 4 threads) path: default W4 seed-0 quantize on this CPU model = 246–348 s "
             f"(Session C default runs 268 / 246 s; Step 1 W4 seeds 292–348 s, mean 314 s). Confirmed configuration: {v[conf]['q']:.0f} s "
             f"→ ×{v[conf]['q']/314:.2f} vs the Step 1 mean, ×{v[conf]['q']/257:.2f} vs Session C's default mean (257 s).")
else:
    L.append("\n**No configuration confirmed yet** (runs missing or not bit-identical).")
# S2 runs
L.append("\n## S2 results – Qwen/Qwen2.5-0.5B (configuration as confirmed above)\n")
dense = ok(R, r"final_S2_qwen_dense", "ppl_dense_fp32")
if dense:
    d = dense[-1]["result"]; L.append(f"Dense fp32 WT-ppl: **{d['ppl_dense_fp32']!r}** (bf16 dense {d.get('ppl_dense_bf16')!r}); wall {dense[-1]['wall_s']} s, peak {dense[-1]['peak_rss_gb']} GB.\n")
g2048 = ok(R, r"final_S2_qwen_w3_L2048_s\d", "ppl_quant"); g512 = ok(R, r"final_S2_qwen_w3_L512_s\d", "ppl_quant")
L.append(HDR)
S = {}
for lab, g in (("GPTQ W3 g128, 128 × 2048 (seeds 0–4)", g2048), ("GPTQ W3 g128, 128 × 512 (seeds 0–2)", g512)):
    if g:
        S[lab] = stats([r["result"]["ppl_quant"] for r in g]); L.append(srow(lab, S[lab]))
L.append("\nPer run:\n\n| Run | Seed | Calib fingerprint | WT-ppl | Quantize s | Eval s | Wall s | Peak RSS GB | ckpt sha (16) |\n|---|---|---|---|---|---|---|---|---|")
for r in g2048 + g512:
    x = r["result"]
    L.append(f"| {r['run_id']} | {x['args']['seed']} | `{x.get('calib_fingerprint')}` | {x['ppl_quant']:.6f} | {ph(r,'quantize')} | {ph(r,'eval_quant')} | {r['wall_s']} | {r['peak_rss_gb']} | `{(x.get('config') or {}).get('ckpt_sha256','')[:16]}` |")
fails = [r for r in R if r["rc"] != 0]
if fails: L.append("\nFailures: " + "; ".join(f"{r['run_id']} rc {r['rc']}: {(r.get('error_tail') or '')[-200:]!r}" for r in fails))
if len(g2048) >= 2 and len(g512) >= 2:
    a = [r["result"]["ppl_quant"] for r in g512]; b = [r["result"]["ppl_quant"] for r in g2048]
    diff = st.mean(b) - st.mean(a); p = float(ttest_ind(b, a, equal_var=False).pvalue)
    sp = math.sqrt(((len(a)-1)*st.variance(a) + (len(b)-1)*st.variance(b)) / (len(a)+len(b)-2))
    L.append(f"\n**Window-length effect (2048 − 512):** Δ = {diff:+.4f} ppl ({100*diff/st.mean(a):+.2f} %), Welch p = {p:.3g}; "
             f"Δ / seed SD = {diff/st.stdev(b):.2f} (2048-arm SD), {diff/st.stdev(a):.2f} (512-arm SD), {diff/sp:.2f} (pooled SD {sp:.4f}).")
    L.append(f"SmolLM2-135M (REPORT.md A.5, F15): +5.15 ppl at W3 (+13.6 %), = 7.1 × its 512-arm seed SD (0.723), Welch p = 7×10⁻⁵.")
# Wanda
if W:
    L.append("\n## Wanda sparsity sweep – SmolLM2-135M (128 × 512 WikiText-2 calibration from the id cache, WT-ppl)\n")
    dense_s = 14.487794398879315
    L.append(f"Dense fp32 WT-ppl 14.487794 (main and wanda venv identical, REPORT.md A.6). 50 % unstructured reference: "
             f"Session C seeds 0–4 (`results/real_step2_wanda.jsonl`, `*_t023`, same windows).\n")
    L.append("| Setting | n | Mean ppl | SD | CV | Min | Max | 95 % CI for SD (χ²) | Range | Range / (mean − dense) | Seed-0 repeat identical? |\n|---|---|---|---|---|---|---|---|---|---|---|")
    ref = [json.loads(l) for l in open(os.path.join(H, "results", "real_step2_wanda.jsonl"))]
    ref = [r for r in ref if re.fullmatch(r"s2_wanda50_135m_s\d_t023", r["run_id"]) and r["rc"] == 0]
    sets = [("50 % unstructured (Session C)", [r["result"]["ppl"] for r in ref], None)]
    for tag, lab in (("24", "2:4 (50 %)"), ("u60", "60 % unstructured"), ("u70", "70 % unstructured")):
        g = ok(W, rf"final_S2_wanda_{tag}_s\d", "ppl"); rep = ok(W, rf"final_S2_wanda_{tag}_s0_rep", "ppl")
        s0 = [r for r in g if r["run_id"].endswith("_s0")]
        same = None if not (rep and s0) else (repr(rep[0]["result"]["ppl"]) == repr(s0[0]["result"]["ppl"]))
        if g: sets.append((lab, [r["result"]["ppl"] for r in g], same))
    for lab, p, same in sets:
        s = stats(p); rng = s["mx"] - s["mn"]
        L.append(srow(lab, s, f" {rng:.4f} | {100*rng/(s['mean']-dense_s):.2f} % | {'–' if same is None else ('yes (bit-identical)' if same else 'NO')} |"))
    L.append("\nPer run:\n\n| Run | Sparsity type | Ratio | Seed | WT-ppl | Measured sparsity | Prune s | Wall s | Peak RSS GB |\n|---|---|---|---|---|---|---|---|---|")
    for r in W:
        x = r.get("result") or {}; m = r.get("meta") or {}
        L.append(f"| {r['run_id']} | {m.get('sparsity_type')} | {m.get('sparsity')} | {m.get('seed')} | {repr(x.get('ppl'))} | {x.get('sparsity')} | {(x.get('phases') or {}).get('prune')} | {r['wall_s']} | {r['peak_rss_gb']} |")
# manual block
p = os.path.join(H, "REPORT_final_S2.md")
manual = ""
if os.path.exists(p):
    t = open(p).read(); m = re.search(r"<!-- MANUAL START -->.*?<!-- MANUAL END -->", t, re.S)
    manual = m.group(0) if m else ""
L.append("\n" + (manual or "<!-- MANUAL START -->\n<!-- MANUAL END -->"))
L.append(f"\nGenerated by `final_S2_report.py` at commit `{subprocess.run(['git','rev-parse','--short','HEAD'],capture_output=True,text=True,cwd=H).stdout.strip()}` (the commit before this file's own commit).\n")
open(p, "w").write("\n".join(L) + "\n")
print("confirmed:", conf)
