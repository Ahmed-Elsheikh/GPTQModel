# S1 final: calibration-seed sensitivity, SmolLM2-135M, GPTQ W3/W4 g128, 128 × 2048 WikiText-2 calibration

Runs: 2026-10-03 15:16 UTC (anchors) and 2026-10-04 09:46 – 2026-10-05 09:31 UTC (main). Branch
`claude/quirky-ramanujan-i0hree`; all results are in commit `512ee7e` ("feasibility S1: chain complete").
Records:
- `results/final_S1_runs.jsonl`: one JSON line per run with model, method, bits, group size, calibration
  source/count/length/fingerprint, seed, configuration flags, CPU model + flags, package versions, per-phase wall time,
  peak RSS, all metrics and checkpoint SHA-256;
- `results/final_S1_lmeval_items.jsonl`: per-item `acc`/`acc_norm` for every lm-eval run.

Scripts: `final_S1_run.py`, `final_S1_chain.sh` (resumable), `final_S1_report.py` (generates every table below).

## Protocol

- **Configuration:** `attn_implementation="eager"` at quantization, reload and dense evaluation; 4 threads.
  - Quantized in fp32 with quantize backend AUTO, as in `quant_eval.py`. Quantized models are evaluated in bf16 with
    `BACKEND.TORCH`; the dense model in fp32.
  - This was the user's decision of 2026-10-04: proceed with eager, and **replace bit-identity with measured noise**.
- **Anchor gate:** each anchor run (W4 g128 seed 0, 128 × 512 WikiText, WT-ppl) must be within 0.05 ppl of 17.660599.
  anchor_a 17.632549 (Δ −0.028) and anchor_b 17.660599 both **pass**. Neither was rerun.
- **Calibration:** 128 × 2048 windows from the cached WikiText-2 train ids (the Step 1 main-venv stream), starts drawn
  with `random.Random(seed)`. Seeds share calibration fingerprints across W3 and W4.
- **Evaluation:**
  - WT-ppl: first 40 × 2048 test windows (fingerprint `7da3b34bdebd1923`).
  - C4-ppl: cached 256 × 2048 validation windows, GPTQ convention (fingerprint `cbe443ebebeeceeb`).
  - lm-eval 0.4.13: arc_easy and hellaswag, `limit=1000` (the same first 1000 items for every model), 0-shot,
    batch size 8, `log_samples=True`.
  - WT-ppl on every run. C4-ppl and lm-eval on dense and seeds 0–2.
- **Run order:** dense → W3 s0 (+ repeat) → W3 s1–4 → W4 s0 (+ repeat) → W4 s1–4 → W3 s5–9.
- **Hardware:** Intel Xeon @ 2.10 GHz (AVX512-BF16, AMX), 4 vCPU; torch 2.14.1+cpu, GPTQModel 7.5.0, transformers 5.18.0.

## Results

### Eager run-to-run noise

**A. Anchor command** (W4 g128 seed 0, 128 × 512 WikiText, WT-ppl): every eager run in this project

| session | run | WT-ppl | ckpt sha (16) | Δ vs modal value |
|---|---|---|---|---|
| S1 | anchor_a | 17.632548589483903 | `e2b8938da584dd6d` | -0.028050 |
| S1 | anchor_b | 17.66059890313337 | `9b3c5e64c541388e` | +0.000000 |
| S3 | anchor_a | 17.664705682453345 | `bfb81cf2743058d0` | +0.004107 |
| S3 | anchor_b | 17.66059890313337 | `9b3c5e64c541388e` | +0.000000 |
| S2 | final_S2_anchor_eager_r1 | 17.66059890313337 | `–` | +0.000000 |
| S2 | final_S2_anchor_eager_r2 | 17.66059890313337 | `–` | +0.000000 |
| S2 | final_S2_anchor_eager_r3 | 17.66059890313337 | `–` | +0.000000 |
| S2 | final_S2_anchor_eager_r4 | 17.66059890313337 | `–` | +0.000000 |
| S2 | final_S2_anchor_eager_r5 | 17.66059890313337 | `–` | +0.000000 |
| step2 §8 | s2_det_eager1 | 17.66059890313337 | `9b3c5e64c541388e` | +0.000000 |
| step2 §8 | s2_det_eager2 | 17.66059890313337 | `9b3c5e64c541388e` | +0.000000 |

- 11 runs, 3 distinct values, 9/11 at the modal value 17.66059890313337.
- Noise SD (all 11 runs) **0.0087**, range **0.0322**, max |Δ| 0.0281.

**B. S1 128 × 2048 seed-0 repeats** (WT-ppl + checkpoint sha)

| cell | run | WT-ppl | ckpt sha (16) | quantize s |
|---|---|---|---|---|
| W3 s0 | w3_s0 | 43.958737527474575 | `dba423fcf9fc708b` | 2894.4 |
| W3 s0 | w3_s0_rep | 43.958737527474575 | `dba423fcf9fc708b` | 2881.37 |
| W3 s0 | → | Δ = 0.000000 | identical | |
| W4 s0 | w4_s0 | 18.649760279942242 | `2d8a7582002857b7` | 3175.34 |
| W4 s0 | w4_s0_rep | 18.649760279942242 | `2d8a7582002857b7` | 3236.24 |
| W4 s0 | → | Δ = 0.000000 | identical | |

### Dense fp32 (eager)

WT-ppl **14.4878**, C4-ppl **18.7520**, arc_easy acc 0.628 (acc_norm 0.587), hellaswag acc 0.366 (acc_norm 0.441); wall 2096.5 s, peak 2.126 GB.

### Seed statistics (128 × 2048 WikiText calibration)

| setting | metric | seeds | n | mean | SD | CV % | min | max | range | 95 % CI for SD (χ²) | SD / anchor-noise SD | range / anchor-noise range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W3 | wt_ppl | 0–9 | 10 | 42.8505 | 1.2561 | 2.93 | 40.2990 | 44.3582 | 4.0592 | [0.8640, 2.2931] | 144.9 | 126.2 |
| W3 | wt_ppl | 0–4 | 5 | 43.4739 | 0.8785 | 2.02 | 42.4988 | 44.3582 | 1.8594 | [0.5263, 2.5245] | 101.3 | 57.8 |
| W4 | wt_ppl | 0–4 | 5 | 18.8627 | 0.1309 | 0.69 | 18.6498 | 19.0063 | 0.3566 | [0.0784, 0.3760] | 15.1 | 11.1 |
| W3 | c4_ppl | 0–2 | 3 | 72.1388 | 0.6104 | 0.85 | 71.7801 | 72.8436 | 1.0635 | [0.3178, 3.8364] | 70.4 | 33.1 |
| W3 | arc_easy_acc | 0–2 | 3 | 0.4087 | 0.0081 | 1.99 | 0.4030 | 0.4180 | 0.0150 | [0.0042, 0.0512] | – | – |
| W3 | arc_easy_acc_norm | 0–2 | 3 | 0.3913 | 0.0081 | 2.07 | 0.3820 | 0.3960 | 0.0140 | [0.0042, 0.0508] | – | – |
| W3 | hellaswag_acc | 0–2 | 3 | 0.3193 | 0.0025 | 0.79 | 0.3170 | 0.3220 | 0.0050 | [0.0013, 0.0158] | – | – |
| W3 | hellaswag_acc_norm | 0–2 | 3 | 0.3877 | 0.0038 | 0.98 | 0.3850 | 0.3920 | 0.0070 | [0.0020, 0.0238] | – | – |
| W4 | c4_ppl | 0–2 | 3 | 25.0699 | 0.1343 | 0.54 | 24.9258 | 25.1917 | 0.2658 | [0.0699, 0.8440] | 15.5 | 8.3 |
| W4 | arc_easy_acc | 0–2 | 3 | 0.5773 | 0.0067 | 1.15 | 0.5730 | 0.5850 | 0.0120 | [0.0035, 0.0418] | – | – |
| W4 | arc_easy_acc_norm | 0–2 | 3 | 0.5410 | 0.0078 | 1.44 | 0.5320 | 0.5460 | 0.0140 | [0.0041, 0.0491] | – | – |
| W4 | hellaswag_acc | 0–2 | 3 | 0.3553 | 0.0021 | 0.59 | 0.3530 | 0.3570 | 0.0040 | [0.0011, 0.0131] | – | – |
| W4 | hellaswag_acc_norm | 0–2 | 3 | 0.4183 | 0.0015 | 0.37 | 0.4170 | 0.4200 | 0.0030 | [0.0008, 0.0096] | – | – |

### Seed range relative to compression gaps (WT-ppl)

- dense → W4 (seeds 0–4 mean): **4.3749**; W4 → W3 (seeds 0–4 means): **24.6112**
- W4 s0–4: range 0.3566 = 8.2 % of dense→W4, 1.4 % of W4→W3; SD 0.1309
- W3 s0–4: range 1.8594 = 42.5 % of dense→W4, 7.6 % of W4→W3; SD 0.8785
- W3 s0–9: range 4.0592 = 92.8 % of dense→W4, 16.5 % of W4→W3; SD 1.2561

### Paired per-item agreement between seeds (lm-eval, 1000 items per task)

| setting | task | metric | seed pair | items | flips | flip % | Δ accuracy |
|---|---|---|---|---|---|---|---|
| W3 | arc_easy | acc | 0–1 | 1000 | 194 | 19.4 | +0.002 |
| W3 | arc_easy | acc | 0–2 | 1000 | 227 | 22.7 | +0.015 |
| W3 | arc_easy | acc | 1–2 | 1000 | 195 | 19.5 | +0.013 |
| W3 | arc_easy | acc_norm | 0–1 | 1000 | 170 | 17.0 | +0.000 |
| W3 | arc_easy | acc_norm | 0–2 | 1000 | 190 | 19.0 | -0.014 |
| W3 | arc_easy | acc_norm | 1–2 | 1000 | 168 | 16.8 | -0.014 |
| W3 | hellaswag | acc | 0–1 | 1000 | 71 | 7.1 | -0.005 |
| W3 | hellaswag | acc | 0–2 | 1000 | 77 | 7.7 | -0.003 |
| W3 | hellaswag | acc | 1–2 | 1000 | 76 | 7.6 | +0.002 |
| W3 | hellaswag | acc_norm | 0–1 | 1000 | 155 | 15.5 | -0.007 |
| W3 | hellaswag | acc_norm | 0–2 | 1000 | 162 | 16.2 | -0.006 |
| W3 | hellaswag | acc_norm | 1–2 | 1000 | 169 | 16.9 | +0.001 |
| W4 | arc_easy | acc | 0–1 | 1000 | 103 | 10.3 | +0.001 |
| W4 | arc_easy | acc | 0–2 | 1000 | 100 | 10.0 | +0.012 |
| W4 | arc_easy | acc | 1–2 | 1000 | 111 | 11.1 | +0.011 |
| W4 | arc_easy | acc_norm | 0–1 | 1000 | 84 | 8.4 | +0.014 |
| W4 | arc_easy | acc_norm | 0–2 | 1000 | 95 | 9.5 | +0.013 |
| W4 | arc_easy | acc_norm | 1–2 | 1000 | 79 | 7.9 | -0.001 |
| W4 | hellaswag | acc | 0–1 | 1000 | 44 | 4.4 | -0.004 |
| W4 | hellaswag | acc | 0–2 | 1000 | 49 | 4.9 | -0.001 |
| W4 | hellaswag | acc | 1–2 | 1000 | 43 | 4.3 | +0.003 |
| W4 | hellaswag | acc_norm | 0–1 | 1000 | 74 | 7.4 | -0.002 |
| W4 | hellaswag | acc_norm | 0–2 | 1000 | 83 | 8.3 | -0.003 |
| W4 | hellaswag | acc_norm | 1–2 | 1000 | 79 | 7.9 | -0.001 |
| dense vs W3 s0 | arc_easy | acc | – | 1000 | 327 | 32.7 | -0.225 |
| dense vs W3 s0 | hellaswag | acc | – | 1000 | 104 | 10.4 | -0.044 |
| dense vs W4 s0 | arc_easy | acc | – | 1000 | 143 | 14.3 | -0.055 |
| dense vs W4 s0 | hellaswag | acc | – | 1000 | 61 | 6.1 | -0.009 |

- W3 arc_easy acc: mean flip rate between seeds 20.5 %
- W3 arc_easy acc_norm: mean flip rate between seeds 17.6 %
- W3 hellaswag acc: mean flip rate between seeds 7.5 %
- W3 hellaswag acc_norm: mean flip rate between seeds 16.2 %
- W4 arc_easy acc: mean flip rate between seeds 10.5 %
- W4 arc_easy acc_norm: mean flip rate between seeds 8.6 %
- W4 hellaswag acc: mean flip rate between seeds 4.5 %
- W4 hellaswag acc_norm: mean flip rate between seeds 7.9 %

### Per run

| run | bits | seed | calib fp | WT-ppl | C4-ppl | arc_easy acc | hellaswag acc | ckpt sha (16) | quantize s | WT s | C4 s | lm-eval s | wall s | peak RSS GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| anchor_a | 4 | 0 | `07db06dabcb03c98` | 17.632548589483903 | – | – | – | `e2b8938da584dd6d` | 332.31 | 253.93 | – | – | 606.6 | 2.726 |
| anchor_b | 4 | 0 | `07db06dabcb03c98` | 17.66059890313337 | – | – | – | `9b3c5e64c541388e` | 334.13 | 262.67 | – | – | 608.5 | 2.724 |
| dense | – | – | `–` | 14.487794658067827 | 18.7520 | 0.628 | 0.366 | `–` | – | 246.75 | 1580.54 | 263.72 | 2096.5 | 2.126 |
| w3_s0 | 3 | 0 | `899864c1ee89b0d4` | 43.958737527474575 | 72.8436 | 0.403 | 0.322 | `dba423fcf9fc708b` | 2894.4 | 164.28 | 928.43 | 353.92 | 4357.0 | 6.042 |
| w3_s0_rep | 3 | 0 | `899864c1ee89b0d4` | 43.958737527474575 | – | – | – | `dba423fcf9fc708b` | 2881.37 | 181.45 | – | – | 3073.3 | 6.077 |
| w3_s1 | 3 | 1 | `0edfe174d4876132` | 42.49880037416078 | 71.7926 | 0.405 | 0.317 | `78875cd61aba3078` | 3025.64 | 203.91 | 1137.84 | 396.02 | 4776.0 | 6.07 |
| w3_s2 | 3 | 2 | `201f12d590e05a38` | 43.998596790084314 | 71.7801 | 0.418 | 0.319 | `2a76c40fa6ed601b` | 3054.48 | 221.02 | 1442.45 | 456.57 | 5186.1 | 6.148 |
| w3_s3 | 3 | 3 | `a5d4c759da11173a` | 42.55520657688512 | – | – | – | `2471ae9fbca12b90` | 3146.69 | 224.05 | – | – | 3385.0 | 6.155 |
| w3_s4 | 3 | 4 | `4051a8dfd01518ca` | 44.358171403125965 | – | – | – | `10280f537d8692ab` | 3285.56 | 194.81 | – | – | 3491.7 | 6.06 |
| w4_s0 | 4 | 0 | `899864c1ee89b0d4` | 18.649760279942242 | 24.9258 | 0.573 | 0.357 | `2d8a7582002857b7` | 3175.34 | 219.86 | 1430.04 | 440.92 | 5277.9 | 6.037 |
| w4_s0_rep | 4 | 0 | `899864c1ee89b0d4` | 18.649760279942242 | – | – | – | `2d8a7582002857b7` | 3236.24 | 221.07 | – | – | 3468.3 | 6.039 |
| w4_s1 | 4 | 1 | `0edfe174d4876132` | 18.862303506391942 | 25.1917 | 0.574 | 0.353 | `5f1daaa2c9d1bbb4` | 2987.72 | 202.86 | 1212.87 | 434.55 | 4849.0 | 6.146 |
| w4_s2 | 4 | 2 | `201f12d590e05a38` | 18.90418856123344 | 25.0922 | 0.585 | 0.356 | `20eea1b97d574911` | 3158.35 | 199.0 | 1149.49 | 428.13 | 4946.6 | 6.329 |
| w4_s3 | 4 | 3 | `a5d4c759da11173a` | 19.006338035392794 | – | – | – | `506bd73af8fea4ce` | 4460.47 | 635.38 | – | – | 5151.0 | 6.132 |
| w4_s4 | 4 | 4 | `4051a8dfd01518ca` | 18.89084176707636 | – | – | – | `0084e170448802b3` | 4614.63 | 639.35 | – | – | 5267.5 | 6.181 |
| w3_s5 | 3 | 5 | `63bf075f092f2336` | 42.462082168639135 | – | – | – | `5f1975278c434b81` | 4462.44 | 537.02 | – | – | 5013.2 | 6.106 |
| w3_s6 | 3 | 6 | `b575efd08b4dc268` | 43.733411515069115 | – | – | – | `eaa1aad5bfc62768` | 4573.03 | 595.38 | – | – | 5182.9 | 6.096 |
| w3_s7 | 3 | 7 | `57602e82f1a512a5` | 40.29896134406989 | – | – | – | `6c79e36a2bfeff49` | 4892.02 | 706.46 | – | – | 5613.4 | 5.86 |
| w3_s8 | 3 | 8 | `0a3f1f496fd3eb0e` | 41.556847966096285 | – | – | – | `714dd8d52192b683` | 4179.02 | 637.75 | – | – | 4832.4 | 6.239 |
| w3_s9 | 3 | 9 | `38eb590b9ab110b8` | 43.084316668766164 | – | – | – | `c4b2cf846454dc61` | 4783.03 | 663.11 | – | – | 5517.1 | 6.224 |


## Interpretation

**Eager noise vs seed effects.**
- The anchor command has now been run 11 times with eager attention, across S1, S3, S2 and REPORT_step2 §8.
  - 9 of 11 runs give exactly 17.66059890313337. The two others are 17.632549 (S1) and 17.664706 (S3).
  - Noise SD is 0.0087 ppl and the range 0.032 ppl.
  - S2's 5/5 bit-identical runs are consistent with this: deviations are rare.
- **The 128 × 2048 seed-0 repeats were bit-identical** (same checkpoint SHA-256) at both W3 and W4. So no larger
  run-to-run noise was observed at W3. With one repeat per bit-width, this only bounds the noise loosely.
- **Every seed SD clearly exceeds the noise:**

  | setting | seed SD (ppl) | multiple of the noise SD | lower 95 % CI bound vs noise range (0.032) |
  |---|---|---|---|
  | W4 WT-ppl | 0.131 | 15× | 0.078, still above |
  | W3 WT-ppl (n = 10) | 1.256 | 145× | – |
  | W4 C4-ppl | – | 15× | – |
  | W3 C4-ppl | – | 70× | – |

  **All of these spreads can be read as calibration-seed effects.** The eager-path noise accounts for at most
  about 7 % of the W4 SD and less than 1 % of the W3 SD.
- The noise was measured at 128 × 512 for the anchors. At 128 × 2048 only the two seed-0 repeats exist, so
  the comparison assumes the noise does not grow with calibration length; the identical repeats support that.

**Size of the seed effect.**
- **W4** (n = 5): WT-ppl 18.863 ± 0.131 (CV 0.69 %), range 0.357.
  - The range is 8.2 % of the dense→W4 degradation (+4.37) and 1.4 % of the W4→W3 gap.
  - The SD CI is [0.078, 0.376]. That is larger than the Step 1 W4 128 × 512 SDPA estimate (0.021, CI up to 0.060),
    and the CIs do not overlap.
  - Longer calibration windows and/or eager attention therefore appear to increase W4 seed sensitivity. The design
    here cannot separate the two.
- **W3** (n = 10): WT-ppl 42.851 ± 1.256 (CV 2.93 %), range 4.06 (40.30–44.36).
  - The range is **93 % of the whole dense→W4 degradation** and 16.5 % of the W4→W3 gap.
  - Seeds 0–4 alone underestimate it: SD 0.879, range 1.86. Seeds 7 and 8 produced the two best values.
  - **At W3, five seeds are not enough to bound the spread.** The n = 10 SD CI is [0.86, 2.29].
- **The W3 mean (42.85) agrees with Session B's SDPA 128 × 2048 W3 arm** (43.11 ± 1.14, REPORT.md A.5). The
  "2048 windows make W3 worse than 512" finding reproduces on the eager path.
- **C4-ppl seed SDs:** W4 0.134, W3 0.610. Their CVs (0.54 %, 0.85 %) are lower than on WikiText-2.

**lm-eval: aggregate accuracy hides large per-item churn.**
- Across seeds, accuracy moves by at most 0.015 (arc_easy) and 0.005 (hellaswag), with SDs of 0.002–0.008.
- Paired per item, the share of items whose correctness flips between two seeds is much larger:

  | setting | arc_easy acc | hellaswag acc | hellaswag acc_norm |
  |---|---|---|---|
  | W4 | 10.5 % | 4.5 % | 7.9 % |
  | W3 | 20.5 % | 7.5 % | 16.2 % |

- **For W4 arc_easy, the seed-to-seed flip rate (10.5 %) is about 70 % of the dense-vs-W4 flip rate (14.3 %).** A single
  calibration seed therefore changes most of the items that quantization itself changes, while leaving the aggregate
  accuracy almost unchanged.
- Accuracy compression effects are real at both bit-widths:
  - dense → W4: arc_easy 0.628 → 0.577; hellaswag 0.366 → 0.355.
  - dense → W3: arc_easy 0.628 → 0.409; hellaswag 0.366 → 0.319.

**Operational notes.**
- **Container restarts:** two restarts killed in-flight runs, `w4_s3` at about 23:00 and `w3_s9` at about 07:55 UTC.
  The resumable chain re-ran each from scratch. Lost partial runs leave no record.
- **Timing drift:** runs after the 23:06 restart were slower on the same CPU model.
  - Quantize took 4180–4890 s, vs 2880–3290 s before.
  - WT eval took 537–706 s, vs 164–224 s before.
  - Wall times are therefore not comparable across that boundary. Quality metrics are unaffected (deterministic inputs).
- **Cost:** a 128 × 2048 GPTQ run with eager attention needs ≈ 48–80 min to quantize, and peak RSS is ≈ 6.0–6.3 GB.
  With C4-ppl + lm-eval a run takes 73–88 min.

## History of this report

The first version of this file (commit `95a7307`) recorded the anchor gate failing under the original bit-identity
rule. That rule was replaced by the ±0.05 ppl tolerance on 2026-10-04, and the same two anchor runs pass it.

## 10-line summary

1. All S1 runs finished (rc 0, commit `512ee7e`): dense, W3 seeds 0–9, W4 seeds 0–4, and seed-0 repeats at W3 and W4. Eager attention, 128 × 2048 WikiText calibration.
2. Eager noise: 11 anchor runs give 3 distinct values, 9 of them identical; SD 0.0087, range 0.032. The 2048-token seed-0 repeats were bit-identical at W3 and W4.
3. Dense fp32: WT-ppl 14.488, C4-ppl 18.752, arc_easy 0.628, hellaswag 0.366.
4. W4 (n = 5): WT-ppl 18.863 ± 0.131 (CV 0.69 %, 95 % CI for SD [0.078, 0.376]); range 0.357 = 8.2 % of dense→W4, 1.4 % of W4→W3.
5. W3 (n = 10): WT-ppl 42.851 ± 1.256 (CV 2.93 %, CI [0.86, 2.29]); range 4.06 = 93 % of dense→W4, 16.5 % of W4→W3.
6. Every seed SD is 15–145× the noise SD, so all reported spreads are calibration-seed effects, not run-to-run noise.
7. At W3, seeds 0–4 underestimated the spread (SD 0.88 vs 1.26 with 10 seeds). W3 needs ≥ 10 seeds.
8. The W4 seed SD at 128 × 2048 with eager attention is 6× Step 1's 128 × 512 SDPA estimate, and the CIs don't overlap. The two factors are confounded here.
9. lm-eval: aggregate accuracy barely moves across seeds (≤ 0.015), but 4.5–10.5 % (W4) and 7.5–20.5 % (W3) of items flip correctness. For W4 arc_easy that is ~70 % of the dense-vs-W4 flip rate.
10. Caveats: two container restarts (2 runs redone), timing drift of up to 3× after a restart, and ~80 min per full run. C4-ppl dominates evaluation time.
