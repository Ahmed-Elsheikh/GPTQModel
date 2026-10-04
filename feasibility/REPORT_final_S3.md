# S3 final: calibration source × evaluation set, GPTQ W4/W3 g128, SmolLM2-135M

Date: 2026-10-03 16:14 – 2026-10-04 06:33 UTC (≈ 4.5 h lost to a container reclaim, see below).
Branch `claude/quirky-ramanujan-i0hree`, results commit `aee81f0` (all 18 runs).
Records: `results/final_S3_runs.jsonl` (one JSON line per run, with model, method, bits, group size, calibration
source/count/length/fingerprint, seed, configuration flags, CPU model + flags, package versions, per-phase wall time,
peak RSS, metrics, checkpoint SHA-256) and `results/final_S3_cache.jsonl` (calibration caches).
Scripts: `final_S3_cache.py`, `final_S3_run.py`, `final_S3_chain.sh` (resumable), `final_S3_report.py` (tables below).

## Protocol

- **Model and method:** SmolLM2-135M, GPTQModel 7.5.0, GPTQ g128, W4 and W3. Quantized in fp32 (quantize backend AUTO,
  as in `quant_eval.py`). Evaluated in bf16 with `BACKEND.TORCH`.
- **Configuration: `attn_implementation="eager"`, 4 threads.** The user chose this (option 3) knowing it is **not
  fully deterministic** (REPORT_final_S1.md). For that reason every run records its checkpoint SHA-256, and the seed-0
  command was repeated so the noise is measured alongside the results.
- **Calibration:** 128 × 512 windows, seeds 0–2, from two sources. Both were tokenized once and cached as `.npy`
  with fingerprints:
  - WikiText-2 train: windows drawn with `random.Random(seed)` from the cached Step 1 token stream. Fingerprints
    `07db06da`/`ecd2326d`/`21eb6029`, the same as Step 1.
  - C4 `en/c4-train.00000-of-01024.json.gz`, GPTQ-style document sampling. Fingerprints `9d332827` (as in
    REPORT_step2 §2), `2fd7455e`, `ff4dee1e`.
- **Evaluation sets.** Every run is evaluated on both:
  - **WT-ppl:** 40 × 2048 WikiText-2 test windows (fingerprint `7da3b34bdebd1923`).
  - **C4-ppl:** 256 × 2048 C4-validation windows, GPTQ convention (fingerprint `cbe443ebebeeceeb`).
- **Hardware and versions:** Intel Xeon @ 2.10 GHz (AVX512-BF16, AMX), 4 vCPU; torch 2.14.1+cpu, transformers 5.18.0.

## Anchor check (recorded, not a gate, per option 3)

The check was W4 g128, seed 0, WikiText-2 128 × 512, WT-ppl, expected value 17.66059890313337.
- **anchor_a: 17.664705682453345** (checkpoint `bfb81cf2`), a **new value**.
- **anchor_b: 17.66059890313337** (checkpoint `9b3c5e64`), equal to the expected value.

The pair is **not bit-identical**. This is the second failed eager anchor pair (S1: 17.632549 vs 17.660599). Across
S1 and S3, 3 of 6 eager runs of this command differ from the modal value, but the deviations are small
(≤ 0.028 ppl).

## Results

| bits | calibration | eval | n | mean | SD | CV % | min | max | 95 % CI for SD (χ²) |
|---|---|---|---|---|---|---|---|---|---|
| W4 | wikitext2 | WikiText-2 | 3 | 17.6545 | 0.1052 | 0.60 | 17.5463 | 17.7564 | [0.0548, 0.6610] |
| W4 | wikitext2 | C4 | 3 | 23.8546 | 0.1463 | 0.61 | 23.6915 | 23.9741 | [0.0762, 0.9193] |
| W4 | c4 | WikiText-2 | 3 | 19.2997 | 0.0712 | 0.37 | 19.2256 | 19.3676 | [0.0371, 0.4476] |
| W4 | c4 | C4 | 3 | 23.5463 | 0.0956 | 0.41 | 23.4752 | 23.6549 | [0.0498, 0.6005] |
| W3 | wikitext2 | WikiText-2 | 3 | 38.5807 | 0.7155 | 1.85 | 37.9142 | 39.3368 | [0.3725, 4.4968] |
| W3 | wikitext2 | C4 | 3 | 64.1891 | 1.2422 | 1.94 | 62.7726 | 65.0928 | [0.6468, 7.8072] |
| W3 | c4 | WikiText-2 | 3 | 50.0415 | 0.3371 | 0.67 | 49.7127 | 50.3864 | [0.1755, 2.1187] |
| W3 | c4 | C4 | 3 | 54.8571 | 1.7704 | 3.23 | 53.1511 | 56.6856 | [0.9218, 11.1265] |

**W4 2×2 (mean ± SD over seeds 0–2)**

| calibration ↓ / eval → | WikiText-2 | C4 val | average of both |
|---|---|---|---|
| wikitext2 | 17.6545 ± 0.1052 | 23.8546 ± 0.1463 | 20.7545 |
| c4 | 19.2997 ± 0.0712 | 23.5463 ± 0.0956 | 21.4230 |

- In-domain advantage on WikiText-2 (C4-cal − WT-cal): **+1.6453** = 18.3× pooled seed SD (0.0898)
- In-domain advantage on C4 (WT-cal − C4-cal): **+0.3083** = 2.5× pooled seed SD (0.1235)
- Average over both eval sets, C4-cal − WT-cal: **+0.6685**

**W3 2×2 (mean ± SD over seeds 0–2)**

| calibration ↓ / eval → | WikiText-2 | C4 val | average of both |
|---|---|---|---|
| wikitext2 | 38.5807 ± 0.7155 | 64.1891 ± 1.2422 | 51.3849 |
| c4 | 50.0415 ± 0.3371 | 54.8571 ± 1.7704 | 52.4493 |

- In-domain advantage on WikiText-2 (C4-cal − WT-cal): **+11.4608** = 20.5× pooled seed SD (0.5593)
- In-domain advantage on C4 (WT-cal − C4-cal): **+9.3320** = 6.1× pooled seed SD (1.5293)
- Average over both eval sets, C4-cal − WT-cal: **+1.0644**

**Per run**

| run | bits | calib | seed | calib fp | WT-ppl | C4-ppl | ckpt sha (16) | quantize s | WT eval s | C4 eval s | wall s | peak RSS GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| anchor_a | 4 | wikitext2 | 0 | `07db06dabcb03c98` | 17.664705682453345 | – | `bfb81cf2743058d0` | 296.38 | 232.71 | – | 540.8 | 2.609 |
| anchor_b | 4 | wikitext2 | 0 | `07db06dabcb03c98` | 17.66059890313337 | – | `9b3c5e64c541388e` | 289.55 | 228.66 | – | 528.3 | 2.667 |
| w4_wikitext2_s0 | 4 | wikitext2 | 0 | `07db06dabcb03c98` | 17.66059890313337 | 23.898264774430007 | `9b3c5e64c541388e` | 284.42 | 225.18 | 1455.37 | 1975.2 | 2.668 |
| w4_c4_s0 | 4 | c4 | 0 | `9d3328272dd6188b` | 19.3060900842731 | 23.508785934265607 | `b0163b96e040416e` | 309.33 | 235.88 | 1455.98 | 2011.7 | 2.66 |
| w3_wikitext2_s0 | 3 | wikitext2 | 0 | `07db06dabcb03c98` | 37.91418911559394 | 62.77255376584804 | `0b8c32e9251cd959` | 318.69 | 256.38 | 1572.37 | 2157.0 | 2.644 |
| w3_c4_s0 | 3 | c4 | 0 | `9d3328272dd6188b` | 50.38638134270746 | 54.73470048297413 | `98d6b032dac7a5ee` | 311.58 | 230.6 | 1439.31 | 1991.4 | 2.59 |
| w4_wikitext2_s1 | 4 | wikitext2 | 1 | `ecd2326d62d1d923` | 17.756436895747743 | 23.97405706303006 | `e12c6208ee591a89` | 298.82 | 226.13 | 1441.67 | 1977.1 | 2.691 |
| w4_c4_s1 | 4 | c4 | 1 | `2fd7455eeb8d1a5d` | 19.22556446664012 | 23.475198259298192 | `4f86126f45ca7ead` | 296.91 | 243.41 | 1495.96 | 2046.0 | 2.63 |
| w3_wikitext2_s1 | 3 | wikitext2 | 1 | `ecd2326d62d1d923` | 38.4911494876271 | 65.0927759335774 | `5b11d3e75c6d204a` | 329.73 | 245.21 | 1497.1 | 2124.1 | 2.687 |
| w3_c4_s1 | 3 | c4 | 1 | `2fd7455eeb8d1a5d` | 50.025415310513736 | 56.68555700231611 | `c4129096b4d70ce2` | 314.46 | 236.35 | 1507.47 | 2068.9 | 2.602 |
| w4_wikitext2_s2 | 4 | wikitext2 | 2 | `21eb6029dad8e6e8` | 17.546341723484865 | 23.69146065865116 | `ed1870d8170677df` | 313.47 | 254.8 | 1512.96 | 2091.4 | 2.688 |
| w4_c4_s2 | 4 | c4 | 2 | `ff4dee1e1b01f396` | 19.367587410793146 | 23.654923052306973 | `6f7874e5ae250b75` | 324.84 | 244.15 | 1510.66 | 2090.2 | 2.712 |
| w3_wikitext2_s2 | 3 | wikitext2 | 2 | `21eb6029dad8e6e8` | 39.336766193551234 | 64.70203300332781 | `bfc63c90d68df2eb` | 314.85 | 236.34 | 1484.28 | 2045.6 | 2.645 |
| w3_c4_s2 | 3 | c4 | 2 | `ff4dee1e1b01f396` | 49.712715823987736 | 53.15111194091182 | `2fb7831ba9df9762` | 314.41 | 230.75 | 1493.95 | 2049.2 | 2.625 |
| w4_wikitext2_s0_rep | 4 | wikitext2 | 0 | `07db06dabcb03c98` | 17.66059890313337 | 23.898264774430007 | `9b3c5e64c541388e` | 311.2 | 248.6 | 1517.4 | 2087.2 | 2.652 |
| w4_c4_s0_rep | 4 | c4 | 0 | `9d3328272dd6188b` | 19.3060900842731 | 23.508785934265607 | `b0163b96e040416e` | 319.85 | 249.49 | 1516.8 | 2096.2 | 2.673 |
| w3_wikitext2_s0_rep | 3 | wikitext2 | 0 | `07db06dabcb03c98` | 37.91418911559394 | 62.77255376584804 | `0b8c32e9251cd959` | 313.06 | 230.81 | 1465.42 | 2019.1 | 2.67 |
| w3_c4_s0_rep | 3 | c4 | 0 | `9d3328272dd6188b` | 50.38638134270746 | 54.73470048297413 | `98d6b032dac7a5ee` | 315.85 | 239.14 | 1532.77 | 2097.7 | 2.608 |

**Eager-path noise (same command repeated)**

| cell | run | WT-ppl | C4-ppl | ckpt sha (16) |
|---|---|---|---|---|
| W4 WikiText s0 (anchor + main) | anchor_a | 17.664705682453345 | – | `bfb81cf2743058d0` |
| W4 WikiText s0 (anchor + main) | anchor_b | 17.66059890313337 | – | `9b3c5e64c541388e` |
| W4 WikiText s0 (anchor + main) | w4_wikitext2_s0 | 17.66059890313337 | 23.898264774430007 | `9b3c5e64c541388e` |
| W4 wikitext2 s0 | w4_wikitext2_s0 | 17.66059890313337 | 23.898264774430007 | `9b3c5e64c541388e` |
| W4 wikitext2 s0 | w4_wikitext2_s0_rep | 17.66059890313337 | 23.898264774430007 | `9b3c5e64c541388e` |
| W4 c4 s0 | w4_c4_s0 | 19.3060900842731 | 23.508785934265607 | `b0163b96e040416e` |
| W4 c4 s0 | w4_c4_s0_rep | 19.3060900842731 | 23.508785934265607 | `b0163b96e040416e` |
| W3 wikitext2 s0 | w3_wikitext2_s0 | 37.91418911559394 | 62.77255376584804 | `0b8c32e9251cd959` |
| W3 wikitext2 s0 | w3_wikitext2_s0_rep | 37.91418911559394 | 62.77255376584804 | `0b8c32e9251cd959` |
| W3 c4 s0 | w3_c4_s0 | 50.38638134270746 | 54.73470048297413 | `98d6b032dac7a5ee` |
| W3 c4 s0 | w3_c4_s0_rep | 50.38638134270746 | 54.73470048297413 | `98d6b032dac7a5ee` |

- W4 WikiText s0 (anchor + main): 3 runs, 2 distinct checkpoints, WT-ppl spread 0.0041
- W4 wikitext2 s0: 2 runs, 1 distinct checkpoints, WT-ppl spread 0.0000
- W4 c4 s0: 2 runs, 1 distinct checkpoints, WT-ppl spread 0.0000
- W3 wikitext2 s0: 2 runs, 1 distinct checkpoints, WT-ppl spread 0.0000
- W3 c4 s0: 2 runs, 1 distinct checkpoints, WT-ppl spread 0.0000


## Interpretation

**In-domain advantage.** Each calibration source wins on its own domain at both bit-widths, with the same crossover
as the single-seed table in REPORT_step2 §7.

| bit-width | evaluation set | in-domain advantage | gap / pooled seed SD |
|---|---|---|---|
| W4 | WikiText-2 | +1.645 ppl | 18× |
| W4 | C4 | +0.308 ppl | 2.5× |
| W3 | WikiText-2 | +11.46 ppl | 20× |
| W3 | C4 | +9.33 ppl | 6× |

- **Averaged over both evaluation sets, WikiText calibration is better:** by 0.67 ppl at W4 and 1.06 ppl at W3.
  Its WikiText-2 advantage is larger than C4 calibration's C4 advantage.
- **With n = 3 the SD CIs are wide** (for example W4 WT-cal/WT-eval SD 0.105, CI [0.055, 0.661]). The
  WikiText-2-eval gaps exceed the upper CI bound of the seed SD by far. The W4 C4-eval gap (0.31) is only 2.5× the
  pooled SD, so it is the least certain conclusion.
- **W3 amplifies everything.** Seed SDs are 5–18× larger than at W4, and the out-of-domain penalty is severe:
  WikiText-calibrated W3 reaches C4-ppl 64.2 vs 54.9 with C4 calibration.

**Run-to-run noise vs seed effect (eager path).**
- **Seed-0 repeats were bit-identical** (same checkpoint SHA and same perplexities) in all 4 cells: W4/W3 ×
  WikiText/C4 calibration.
- **The only non-reproduction** was anchor_a, at WT-ppl Δ = 0.004.
- **The observed eager noise (≤ 0.028 ppl across S1 and S3) is far below every seed SD measured here** (W4 0.07–0.15,
  W3 0.34–1.77) and every calibration-source gap. It therefore does not change any conclusion above. It does make
  exact-value reproduction checks unreliable.

**Implementation effect (flag).** For the same seed-0 C4 calibration data, eager attention gives a different result
from the default SDPA path, measured once in REPORT_step2 §7:

| C4-calibrated W4 s0 | WT-ppl | C4-ppl |
|---|---|---|
| SDPA (REPORT_step2 §7) | 18.925 | 23.343 |
| eager (here) | 19.306 | 23.509 |

- The WT-ppl Δ of 0.38 is about 5× the W4 C4-calibration seed SD (0.071).
- For WikiText calibration the difference is smaller: 17.811 SDPA (17.745 on its other outcome) vs 17.661 eager.
- **The attention implementation is a first-order protocol variable for C4-calibrated GPTQ** and must be reported.
  S3's numbers are internally consistent, but should not be mixed with SDPA-path numbers.
- **Seed SD on the eager path is larger than on the SDPA path:** W4 WikiText-cal WT-ppl SD 0.105 here (n = 3) vs 0.021
  in Step 1 (SDPA, n = 5). The CIs barely overlap ([0.055, 0.66] vs [0.012, 0.060]). This was not investigated further.

**Operational notes.**
- The container was reclaimed at about 20:04 UTC while `w3_wikitext2_s1` was in its C4 evaluation. That run was lost
  without a record because the process died with the container. It was re-run from scratch after the resumable chain
  restarted at 00:46 UTC.
- No run failed (all 18 lines have rc 0).
- Cost per run: about 33–35 min. Quantize takes about 5 min; C4-ppl eval in bf16 with eager attention takes about
  24–26 min and dominates.

## 10-line summary

1. All 18 runs finished: 12 main + 2 anchors + 4 seed-0 repeats, eager attention, rc 0. Results at commit `aee81f0`.
2. W4 (mean over seeds 0–2), WikiText-cal vs C4-cal: WT-ppl 17.65 vs 19.30, C4-ppl 23.85 vs 23.55.
3. W3: WT-ppl 38.58 vs 50.04, C4-ppl 64.19 vs 54.86.
4. In-domain advantage on WikiText-2: +1.65 (W4, 18× seed SD) and +11.46 (W3, 20×).
5. In-domain advantage on C4: +0.31 (W4, 2.5×, weakest) and +9.33 (W3, 6×).
6. Averaged over both evaluation sets, WikiText calibration is better by 0.67 (W4) and 1.06 ppl (W3).
7. Seed CVs: W4 0.4–0.6 %, W3 0.7–3.2 %. With n = 3 the SD CIs span about 12×.
8. Eager noise: the anchor pair mismatched again (17.6647 vs 17.6606), but all 4 seed-0 repeats were bit-identical. Noise ≤ 0.028 ppl, far below the seed and source effects.
9. Flag: eager vs SDPA changes the C4-calibrated W4 seed-0 WT-ppl by 0.38 (19.31 vs 18.93). The attention implementation is a first-order protocol variable.
10. A container reclaim cost about 4.5 h. One in-flight run was lost and redone via the resumable chain; the per-run cost is about 34 min, dominated by C4-ppl.
