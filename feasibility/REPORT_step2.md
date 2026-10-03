# Step 2 (parallel worker): Wanda seed sweep, C4 calibration, AWQ, Qwen2.5-0.5B, real lm-eval, optimum-benchmark launchers

Date: 2026-10-03, 08:10–09:25 UTC. Same 4-vCPU CPU-only VM type as `REPORT.md`, **real weights and real data** (HF Hub now
reachable). Runs were one at a time in this container. The Step 1 session ran in a different container, so the two sessions
did not compete for CPU. Every run went through `runlog.py` (wall time and peak RSS of the child tree).
Raw records: `results/real_step2_{wanda,c4,awq,qwen,lmeval,ob}.jsonl`. Logs: `logs/real_s2_*.log`.
Scripts: `chain_step2a2.sh` (Wanda), `chain_step2b.sh` (tasks 2–6), `wanda_real_driver.py`, `step2_summarize.py`.
Common settings: SmolLM2-135M unless noted. Calibration is 128 × 512 windows and perplexity uses the **Step 1 protocol**
(first 40 non-overlapping 2048-token windows of the WikiText-2 test split, eval fingerprint `7da3b34bdebd1923`).
Dense fp32 ppl = **14.487794** (identical to Step 1 `dense_135m`).

## 1. Wanda 50 % unstructured, seeds 0–4

locuslab/wanda @8e8fc87 + `patches/wanda_cpu.patch` (fp32, seqlen 512, CPU), wanda venv (transformers 4.47.1), 4 threads.
The driver makes two swaps, and wanda's code is otherwise unchanged:
- **Calibration.** Wanda hard-codes C4. The driver feeds it `calib.draw_windows("wikitext2", n=128, L=512, seed)` instead.
  These are byte-identical to the Step 1 GPTQ windows for the same seed (fingerprints match:
  s0 `07db06da…`, s1 `ecd2326d…`, s2 `21eb6029…`, s3 `eaa59f11…`, s4 `67f586b3…`).
- **Perplexity.** `main.eval_ppl` is replaced by `quant_eval.perplexity` on the Step 1 windows.

| seed | ppl | sparsity | prune s | run wall s | peak RSS GB |
|---|---|---|---|---|---|
| 0 | 31.5001 | 0.5000 | 90.2 | 169.5 | 2.19 |
| 1 | 31.4202 | 0.5000 | 90.1 | 171.3 | 2.19 |
| 2 | 31.6719 | 0.5000 | 91.9 | 165.2 | 2.20 |
| 3 | 31.3436 | 0.5000 | 93.5 | 176.4 | 2.19 |
| 4 | 31.5168 | 0.5000 | 88.2 | 167.9 | 2.19 |

**Mean 31.4905, SD 0.1227, min 31.3436, max 31.6719, CV 0.39 %.**
- Dense → Wanda increase: **+17.00 ppl** (×2.17).
- **Seed range 0.328 = 1.9 % of the increase.**
- For comparison, Step 1 GPTQ W4 g128 seeds 0–4 (same calibration windows): mean 17.743, SD 0.021, CV 0.12 %,
  range 0.054, which is **1.7 %** of its +3.26 increase. Wanda's absolute seed spread is about 6× GPTQ W4's, but relative to
  the compression damage it is about the same size.

**Tokenizer-version pitfall (found while doing this).** The wanda venv ships tokenizers 0.21.4 and the main venv
ships 0.23.2. On the joined WikiText-2 text they produce **different ids from token 66 557 onward**: 15 171 of the first
82 000 test ids differ, and the stream lengths are 304 978 vs 304 986.
- So with the venv's own tokenizer, eval windows 33–40 and all calibration windows differ from Step 1.
- The first attempt (`s2_wanda_dense_135m`, `s2_wanda50_135m_s0`, kept in the jsonl) gave dense 14.487248 and
  Wanda s0 31.3864.
- Δ = 0.114 for Wanda s0, which is about one seed-SD. The tokenizer version alone moves the result by about as much as
  the calibration seed does.
- The reported sweep (`*_t023`) instead loads token streams pre-tokenized by the main venv
  (`WIKITEXT_STREAM_DIR`, `wanda_real_driver.py`).
- The tokenizers version is another hidden protocol variable whenever tools need separate venvs.

## 2. GPTQ W4 g128 seed 0 with C4 calibration

`allenai/c4`, `en/c4-train.00000-of-01024.json.gz`, GPTQ-paper-style window sampling (`calib.py`), fingerprint
`9d3328272dd6188b`. Quantized in fp32, reloaded in bf16 with `BACKEND.TORCH`, ppl on the WikiText-2 windows.

| calibration | ppl | Δ vs dense | quantize s | run wall s | peak RSS GB |
|---|---|---|---|---|---|
| WikiText-2 s0 (Step 1, `gptq_w4_135m_s0`) | 17.7448 | +3.257 | 348.4 | 478.8 | 2.91 |
| **C4 s0** (`s2_gptq_w4_135m_c4_s0`) | **18.9251** | **+4.437** | 256.8 | 363.5 | 2.94 |

- C4 calibration costs **+1.18 ppl** on WikiText-2 relative to WikiText-2 calibration. That is 36 % more degradation,
  and **≈ 22× the Step 1 W4 seed range** (0.054).
- The calibration source therefore dominates the seed for in-domain WikiText perplexity. Part of this is the
  calibration set matching the eval domain, which is the usual GPTQ-paper caveat.
- Quantize time varies run to run: 257 s here vs 292–348 s for the five Step 1 W4 runs (same code, other container).

## 3. AWQ W4 on 135M, `quantize(backend=BACKEND.AWQ_TORCH)`

| config | status | quantize s | eval s | run wall s | peak RSS GB | ppl |
|---|---|---|---|---|---|---|
| g128 (as planned) | **FAIL after 22.6 s**: `awq_processor.py:83 ValueError: Expected in_features (576) to be divisible by group_size (128)` | – | – | 22.6 | 2.52 | – |
| g64, auto dtype (bf16) | **PASS** | **953.9** | 48.1 | 1024.1 | 3.58 | **17.4524** (Δ +2.96) |

- Real-weight AWQ g64 took 954 s, **7.4× faster than the 7036 s measured on random weights** in `REPORT.md`, and
  about 2.7–3.7× the fp32 GPTQ W4 quantize time (257–348 s).
- The `REPORT.md` AWQ budget (≈ 13× GPTQ) is therefore far too pessimistic: a 135M AWQ cell is about 17 min, not 2 h.
- AWQ g64 beats GPTQ g128 here (17.45 vs 17.74), but the group sizes differ, so this is **not** a like-for-like
  method comparison.

## 4. Qwen/Qwen2.5-0.5B GPTQ W4 g128 seed 0 (WikiText-2 calibration)

| phase | wall s | peak RSS GB |
|---|---|---|
| data (tokenize + windows) | 14.8 | 2.26 |
| dense fp32 ppl (40×2048) | 219.6 | 5.98 |
| quantize (fp32) | **635.5** | – |
| save + reload (bf16 TorchLinear) | 10.7 | – |
| quantized ppl | 131.1 | 7.08 |
| **total run** | **1020.3** | **7.08** |

**Dense 12.2285 → W4 13.3959 (+1.17, +9.5 %).** Hidden size 896 = 7×128, so g128 is valid. Quantize takes about 1.8–2.5× the
135M time (vs 2.3× on synthetic weights, 1233 s), and peak RSS reaches 7.1 GB during the quantized ppl eval on 15.7 GB.

## 5. lm-eval 0.4.13, real arc_easy + hellaswag, 200 items each, 0-shot, bs 8, seed 0

Dense = HFLM fp32. GPTQ = in-memory `GPTQModel.load(BACKEND.TORCH, bf16)` → `HFLM(pretrained=qm.model)`, using the
**C4-calibrated seed-0 checkpoint from §2**, because Step 1 checkpoints live in the other container.

| model | arc_easy acc | acc_norm | hellaswag acc | acc_norm | run wall s | peak RSS GB |
|---|---|---|---|---|---|---|
| dense fp32 | 0.570 | 0.555 | 0.425 | 0.470 | 72.6 | 2.15 |
| GPTQ W4 (C4 s0) | 0.555 | 0.570 | 0.430 | 0.440 | 85.2 | 1.64 |

- Stderr is ≈ 0.035 per cell (n = 200), and every dense–GPTQ difference (≤ 0.03) is inside one stderr. At 200 items these
  tasks **cannot resolve** a W4 effect that costs +4.4 ppl. That argues for ≥ 500 items, or for paired per-item analysis,
  when comparing seeds.
- Real tasks are much cheaper than the synthetic stand-ins (73 s vs 242 s dense), because real items are shorter.
- The quantized model is only 1.2× slower than dense here, vs 2.3× in `REPORT.md`.

## 6. optimum-benchmark 0.6.0: default (process) vs inline launcher

venv `bench` (transformers 4.x). SmolLM2-135M fp32, OMP_NUM_THREADS=4, warmup 10, 5 iterations, 32 new tokens.
p1 = bs 1 × seq 128 and p2 = bs 4 × seq 512 (two of the Test 3b grid points).

| protocol | launcher | prefill s | decode s | per-token s | decode tok/s | run wall s | rc |
|---|---|---|---|---|---|---|---|
| p1 | process (default) | 0.938 | 7.070 | 0.2302 | 4.38 | 99.7 | 0 |
| p1 | inline | 0.104 | 0.767 | 0.0265 | 40.44 | 16.3 | 1* |
| p2 | process (default) | 6.876 | 11.749 | 0.5184 | 10.55 | 255.5 | 0 |
| p2 | inline | 1.349 | 1.416 | 0.0732 | 87.60 | 45.6 | 1* |

\* The inline launcher writes `benchmark_report.json`, then crashes while serializing the config:
`PerTokenLatencySessionTrackerLogitsProcessor is not JSON serializable`. This is the same bug as `REPORT.md` §6.6.

- With real weights, the process launcher inflates latency **8.7× (p1) and 7.1× (p2) per token**, and prefill by
  9.0× and 5.1×. The parent busy-wait competes with the 4 OMP threads.
- The size of the artefact depends on the protocol, so a ratio computed with the default launcher is not even a constant
  rescaling of the true one.
- EXP4 must use `launcher=inline` (patched or tolerating the exit code) or OMP ≤ 3.
- `runlog.py`'s peak-RSS value for these runs (1.30 GB for all four) is not reliable, because RUSAGE_CHILDREN misses the
  launcher's grandchild. Use optimum-benchmark's own memory fields instead.

## Summary (10 lines)
1. Wanda 50 % (135M, 5 seeds, Step 1 windows): ppl 31.49 ± 0.12 (CV 0.39 %, range 31.34–31.67) vs dense 14.49; seed range = 1.9 % of the +17.0 increase.
2. GPTQ W4 seed spread is relatively similar (range = 1.7 % of +3.26) but 6× smaller in absolute ppl.
3. The wanda venv's older tokenizers (0.21.4 vs 0.23.2) changes WikiText-2 ids after token 66 557 and moves Wanda ppl by about 1 SD. Tokenizer version is a hidden protocol variable.
4. C4 vs WikiText-2 calibration (GPTQ W4 s0): 18.93 vs 17.74 WikiText ppl, a +1.18 gap ≈ 22× the seed range; the source dominates the seed.
5. AWQ via `BACKEND.AWQ_TORCH`: g128 fails at once on K=576; g64 works in 954 s quantize (17 min/run), ppl 17.45.
6. Real-weight AWQ is 7.4× faster than the synthetic-weight estimate; the `REPORT.md` AWQ budget is far too pessimistic.
7. Qwen2.5-0.5B GPTQ W4 g128: quantize 636 s, whole run 17 min, peak 7.1 GB, ppl 12.23 → 13.40 (+9.5 %).
8. lm-eval real arc_easy/hellaswag (200 items, 0-shot): dense 0.570/0.425 vs GPTQ-W4 0.555/0.430 acc. All Δ < 1 stderr (≈ 0.035), so 200 items cannot resolve W4 effects.
9. lm-eval on real tasks: 73 s dense, 85 s GPTQ (in-memory, BACKEND.TORCH bf16), 3–6× cheaper than the synthetic stand-ins (242 s / 547 s).
10. optimum-benchmark: the default process launcher inflates per-token latency 8.7× (p1) and 7.1× (p2) vs inline. Inline still crashes after writing its report, so EXP4 needs `launcher=inline` or OMP ≤ 3.
