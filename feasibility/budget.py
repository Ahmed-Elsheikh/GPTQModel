"""Extrapolate the CPU budget of the full design from the unit costs measured in this study.
All times in seconds, single job on the 4-vCPU VM. Prints a markdown block used in REPORT.md section 4."""
import json, os, sys

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
j = lambda p: json.load(open(os.path.join(R, p)))

# ---- measured unit costs -------------------------------------------------------------------------
q135_gptq = j("t1/gptq_135m_s0_fp32.json")["phases"]["quantize"]["wall_s"]          # fp32 GPTQ W4, 135M
q360_gptq = j("t1/gptq_360m_s0.json")["phases"]["quantize"]["wall_s"]               # fp32 GPTQ W4, 360M
ppl135 = j("t1/gptq_135m_s0_fp32_eval.json")["phases"]["eval_quant"]["wall_s"]       # 40x2048, TorchLinear bf16
ppl360 = j("t1/gptq_360m_s0_eval.json")["phases"]["eval_quant"]["wall_s"]
lm135_q_400 = j("t2/b_inmemory.json")["wall_s"]                                      # 200+200 items, GPTQ in-memory
lm135_d_400 = 4 * 60 + 1.84                                                          # 2a dense fp32 200+200 (4:01.84)
awq_path = os.path.join(R, "t1/awq_135m_s0_g64.json")
awq = json.load(open(awq_path)) if os.path.exists(awq_path) else {}
q135_awq = awq.get("phases", {}).get("quantize", {}).get("wall_s") or (float(sys.argv[1]) if len(sys.argv) > 1 else None)
AWQ_EST = "quantize" not in awq.get("phases", {})
ob_run = 24 * 60 + 35.62                                                             # 3b sweep: 16 runs
ob_unit = ob_run / 16

s360 = ppl360 / ppl135                                     # measured 360M/135M forward-cost ratio
lm135_q = lm135_q_400 * 2.5                                # 2 tasks x 500 items (linear in items: 100->122 s, 200->242 s)
lm360_q = lm135_q * s360
lm135_d = lm135_d_400 * 2.5
lm360_d = lm135_d * s360
q360_awq = q135_awq * (q360_gptq / q135_gptq) if q135_awq else None

def cell(q, ppl, lm): return q + ppl + lm
c = {
    ("135M", "GPTQ"): cell(q135_gptq, ppl135, lm135_q), ("360M", "GPTQ"): cell(q360_gptq, ppl360, lm360_q),
    ("135M", "AWQ"): cell(q135_awq, ppl135, lm135_q), ("360M", "AWQ"): cell(q360_awq, ppl360, lm360_q),
}
h = lambda x: x / 3600
out = []
P = out.append
P(f"| Unit (measured unless *est.*) | 135M | 360M |\n|---|---|---|")
P(f"| GPTQ W4/W3 quantize, N=128, L=512, fp32 | {q135_gptq:.0f} s | {q360_gptq:.0f} s |")
P(f"| AWQ W4 quantize (g64, bf16, AWQ_TORCH) | {q135_awq:.0f} s | *est.* {q360_awq:.0f} s (× GPTQ 360M/135M ratio {q360_gptq/q135_gptq:.2f}) |")
P(f"| ppl 40×2048, quantized model (TorchLinear, bf16) | {ppl135:.0f} s | {ppl360:.0f} s (ratio {s360:.2f}) |")
P(f"| lm-eval 2 tasks × 500 items, quantized (in-memory, bs 8) | {lm135_q_400:.0f} s × 2.5 = {lm135_q:.0f} s | *est.* × {s360:.2f} = {lm360_q:.0f} s |")
P(f"| lm-eval 2 tasks × 500 items, dense fp32 | {lm135_d_400:.0f} s × 2.5 = {lm135_d:.0f} s | *est.* {lm360_d:.0f} s |")
P(f"| optimum-benchmark run (5 iters, 32 new tokens) | {ob_unit/60:.1f} min (mean of 16-run sweep, both models) | |")
P("")
P("**One EXP1/EXP2 cell = quantize + ppl + 2×500 items:**")
for (m, meth), v in c.items():
    P(f"- {m} {meth}: {v:.0f} s = **{v/60:.0f} min**")
exp1 = {m: 10 * (2 * c[(m, 'GPTQ')] + c[(m, 'AWQ')]) for m in ("135M", "360M")}       # W4 + W3 GPTQ, AWQ W4, 10 seeds
exp2 = {m: 3 * 5 * (2 * c[(m, 'GPTQ')] + c[(m, 'AWQ')]) for m in ("135M", "360M")}   # x 3 sources x 5 seeds
# EXP3: 16 dense lm-eval protocol variants per model; measured multipliers vs 0-shot fp32 bs8 baseline:
# 5-shot x4.9 (9:56/2:02), bs1 x1.8, bf16 x2.0, max_len change ~x1 (with 5-shot x4.3). Assume 8 zero-shot variants
# averaging x1.5 and 8 five-shot variants averaging x4.6.
exp3 = {m: (8 * 1.5 + 8 * 4.6) * d for m, d in (("135M", lm135_d), ("360M", lm360_d))}
exp4 = 3 * 24 * ob_unit * 1.3   # 3 models (Qwen-0.5B ~1.3x slower than the 135M/360M mean)
P("")
P(f"**EXP1** = 2 models × {{GPTQ W4, GPTQ W3, AWQ W4}} × 10 seeds: 135M 10×(2×{c[('135M','GPTQ')]/60:.0f}+{c[('135M','AWQ')]/60:.0f}) min = "
  f"{h(exp1['135M']):.1f} h; 360M 10×(2×{c[('360M','GPTQ')]/60:.0f}+{c[('360M','AWQ')]/60:.0f}) min = {h(exp1['360M']):.1f} h → **{h(sum(exp1.values())):.0f} h**")
P(f"**EXP2** = same × 3 sources × 5 seeds: 135M {h(exp2['135M']):.1f} h; 360M {h(exp2['360M']):.1f} h → **{h(sum(exp2.values())):.0f} h** "
  f"(≈ {h(sum(exp2.values()))*2/3:.0f} h if one source reuses EXP1 cells)")
P(f"**EXP3** = 2 models × 16 dense lm-eval variants (8 zero-shot at ~1.5× the 0-shot baseline, 8 five-shot at ~4.6×): "
  f"135M {h(exp3['135M']):.1f} h; 360M {h(exp3['360M']):.1f} h → **{h(sum(exp3.values())):.0f} h**")
P(f"**EXP4** = 3 models × 24 protocols × {ob_unit/60:.1f} min × 1.3 → **{h(exp4):.1f} h** (with optimum-benchmark's default "
  f"≥10 iterations/≥10 s/100 tokens, ~3–5× more)")
tot = sum(exp1.values()) + sum(exp2.values()) + sum(exp3.values()) + exp4
P(f"\n**Total ≈ {h(tot):.0f} h ≈ {h(tot)/24:.1f} days of continuous single-VM compute.**")
# reduced design
lm135_q2, lm360_q2 = lm135_q_400, lm135_q_400 * s360            # 200 items per task
cr = lambda m, meth: (q135_gptq if m == "135M" else q360_gptq) * (1 if meth == "GPTQ" else (q135_awq / q135_gptq)) \
    + (ppl135 if m == "135M" else ppl360) + (lm135_q2 if m == "135M" else lm360_q2)
r1 = sum(10 * cr(m, "GPTQ") + 3 * cr(m, "GPTQ") + 3 * cr(m, "AWQ") for m in ("135M", "360M"))   # W4x10, W3x3, AWQx3
r2 = sum(3 * 3 * cr(m, "GPTQ") for m in ("135M", "360M"))                                      # GPTQ W4 x 3 src x 3 seeds
r3 = (10 * 1.5 + 2 * 4.6) * lm135_d + (6 * 1.5) * lm360_d                                       # 0-shot factorial + 2 few-shot; 360M 6 variants
r4 = exp4
rt = r1 + r2 + r3 + r4
P(f"\n**Reduced design** (200 items/task in seed sweeps; EXP1 = GPTQ W4 ×10 seeds + W3 ×3 + AWQ ×3; EXP2 = GPTQ W4 × 3 sources × 3 seeds; "
  f"EXP3 = 135M 10 zero-shot + 2 few-shot variants, 360M 6 variants; EXP4 unchanged): EXP1 {h(r1):.0f} h, EXP2 {h(r2):.0f} h, "
  f"EXP3 {h(r3):.0f} h, EXP4 {h(r4):.1f} h → **{h(rt):.0f} h** (≈ {h(rt)/4:.0f} h wall on 4 parallel VMs).")
print("\n".join(out))
