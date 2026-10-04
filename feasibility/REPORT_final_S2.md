# Final S2 – Qwen2.5-0.5B GPTQ seed and window-length study (with confirmed determinism)

Session S2. Raw records: `results/final_S2_anchor.jsonl`, `results/final_S2_runs.jsonl`, `results/final_S2_wanda.jsonl` (one JSON line per run: model, method, bits/sparsity, group size, calibration source/count/length, seed, configuration, CPU model + flags, package versions, quantize/eval wall time, peak RSS, metrics, checkpoint SHA-256). Scripts: `chain_final_S2.sh` (resumable chain), `final_S2_run.py` (configuration wrapper around `quant_eval.py`), `final_S2_check.py`, `final_S2_report.py`. Token ids: `cache_tok/` only (WT-ppl fingerprint `7da3b34bdebd1923`).

## Determinism (confirmed)

Anchor: SmolLM2-135M, GPTQ W4 g128, seed 0, 128 × 512 WikiText-2 train calibration (fingerprint `07db06dabcb03c98`), fp32 quantize, bf16 TorchLinear (BACKEND.TORCH) eval, WT-ppl. Bit-identical = same perplexity (exact float repr) **and** same checkpoint SHA-256.

| Run | Configuration | rc | WT-ppl (repr) | Checkpoint sha256 (16) | Quantize s | Eval s | Wall s | Peak RSS GB |
|---|---|---|---|---|---|---|---|---|
| final_S2_anchor_eager_r1 | eager (attn eager, threads 4) | 0 | 17.66059890313337 | `9b3c5e64c541388e` | 424.06 | 309.03 | 758.1 | 2.623 |
| final_S2_anchor_eager_r2 | eager (attn eager, threads 4) | 0 | 17.66059890313337 | `9b3c5e64c541388e` | 465.68 | 322.06 | 805.9 | 2.615 |
| final_S2_anchor_eager_r3 | eager (attn eager, threads 4) | 0 | 17.66059890313337 | `9b3c5e64c541388e` | 449.57 | 323.29 | 790.7 | 2.636 |
| final_S2_anchor_eager_r4 | eager (attn eager, threads 4) | 0 | 17.66059890313337 | `9b3c5e64c541388e` | 474.96 | 299.97 | 793.8 | 2.64 |
| final_S2_anchor_eager_r5 | eager (attn eager, threads 4) | 0 | 17.66059890313337 | `9b3c5e64c541388e` | 435.72 | 326.29 | 779.3 | 2.65 |

- **eager**: 5 successful runs, distinct ppl values ['17.66059890313337'], distinct checkpoints 1; mean quantize 450 s, mean wall 786 s.

**Confirmed configuration: `eager`** (attn_implementation="eager", default 4 threads), BACKEND.TORCH, fp32 quantize, bf16 eval. **Anchor value: 17.66059890313337.**
- Session C's eager value (REPORT_step2.md §8, other container): 17.66059890313337. **Equal (bit-for-bit in ppl)** → the configuration is reproducible across containers.
- Quantize-time cost vs the default (SDPA, 4 threads) path: default W4 seed-0 quantize on this CPU model = 246–348 s (Session C default runs 268 / 246 s; Step 1 W4 seeds 292–348 s, mean 314 s). Confirmed configuration: 450 s → ×1.43 vs the Step 1 mean, ×1.75 vs Session C's default mean (257 s).

## S2 results – Qwen/Qwen2.5-0.5B (configuration as confirmed above)

Dense fp32 WT-ppl: **12.228462033347538** (bf16 dense 12.24444547001298); wall 1008.8 s, peak 5.566 GB.

| Setting | n | Mean ppl | SD | CV | Min | Max | 95 % CI for SD (χ²) |
|---|---|---|---|---|---|---|---|
| GPTQ W3 g128, 128 × 2048 (seeds 0–4) | 5 | 20.2814 | 0.3039 | 1.50 % | 19.7584 | 20.5147 | [0.1821, 0.8733] |
| GPTQ W3 g128, 128 × 512 (seeds 0–2) | 3 | 20.2378 | 0.1647 | 0.81 % | 20.0839 | 20.4116 | [0.0858, 1.0354] |

Per run:

| Run | Seed | Calib fingerprint | WT-ppl | Quantize s | Eval s | Wall s | Peak RSS GB | ckpt sha (16) |
|---|---|---|---|---|---|---|---|---|
| final_S2_qwen_w3_L2048_s0 | 0 | `e1873f9616c4e0f6` | 19.758440 | 5359.78 | 351.23 | 5737.4 | 9.824 | `2373ee003261c1a4` |
| final_S2_qwen_w3_L2048_s1 | 1 | `08024dfe51f10dab` | 20.439198 | 5402.27 | 386.73 | 5815.9 | 9.718 | `c6d2ca0a32caddc7` |
| final_S2_qwen_w3_L2048_s2 | 2 | `0032af368a16abd8` | 20.410100 | 4832.71 | 245.11 | 5104.4 | 10.186 | `197a16d1c84a3f16` |
| final_S2_qwen_w3_L2048_s3 | 3 | `1178f0416d637016` | 20.284615 | 5246.2 | 332.56 | 5607.4 | 9.756 | `4b42d16e1aac6c37` |
| final_S2_qwen_w3_L2048_s4 | 4 | `9fd67da21b6f85fb` | 20.514736 | 4551.34 | 279.89 | 4856.6 | 9.945 | `b52e786da1bce627` |
| final_S2_qwen_w3_L512_s0 | 0 | `516aef8fb7a53ab7` | 20.218043 | 838.2 | 363.56 | 1223.7 | 5.666 | `24b1726a118d722c` |
| final_S2_qwen_w3_L512_s1 | 1 | `75efb4b90b5db078` | 20.083870 | 820.45 | 362.48 | 1203.0 | 5.66 | `cff7f873e37ca5a1` |
| final_S2_qwen_w3_L512_s2 | 2 | `a2050b9016ef85e3` | 20.411582 | 815.58 | 365.09 | 1200.7 | 5.679 | `696c8aed529a1c95` |

**Window-length effect (2048 − 512):** Δ = +0.0436 ppl (+0.22 %), Welch p = 0.802; Δ / seed SD = 0.14 (2048-arm SD), 0.26 (512-arm SD), 0.16 (pooled SD 0.2657).
SmolLM2-135M (REPORT.md A.5, F15): +5.15 ppl at W3 (+13.6 %), = 7.1 × its 512-arm seed SD (0.723), Welch p = 7×10⁻⁵.

## Wanda sparsity sweep – SmolLM2-135M (128 × 512 WikiText-2 calibration from the id cache, WT-ppl)

Dense fp32 WT-ppl 14.487794 (main and wanda venv identical, REPORT.md A.6). 50 % unstructured reference: Session C seeds 0–4 (`results/real_step2_wanda.jsonl`, `*_t023`, same windows).

| Setting | n | Mean ppl | SD | CV | Min | Max | 95 % CI for SD (χ²) | Range | Range / (mean − dense) | Seed-0 repeat identical? |
|---|---|---|---|---|---|---|---|---|---|---|
| 50 % unstructured (Session C) | 5 | 31.4905 | 0.1227 | 0.39 % | 31.3436 | 31.6719 | [0.0735, 0.3525] | 0.3283 | 1.93 % | – |
| 2:4 (50 %) | 3 | 109.0997 | 1.5514 | 1.42 % | 107.4696 | 110.5581 | [0.8077, 9.7500] | 3.0885 | 3.26 % | yes (bit-identical) |
| 60 % unstructured | 3 | 90.1179 | 0.3201 | 0.36 % | 89.9083 | 90.4864 | [0.1667, 2.0120] | 0.5780 | 0.76 % | yes (bit-identical) |
| 70 % unstructured | 3 | 1135.7981 | 329.2633 | 28.99 % | 917.2790 | 1514.5041 | [171.4337, 2069.3326] | 597.2251 | 53.26 % | yes (bit-identical) |

Per run:

| Run | Sparsity type | Ratio | Seed | WT-ppl | Measured sparsity | Prune s | Wall s | Peak RSS GB |
|---|---|---|---|---|---|---|---|---|
| final_S2_wanda_24_s0 | 2:4 | 0.5 | 0 | 107.46964878388229 | 0.5 | 107.4 | 207.2 | 2.058 |
| final_S2_wanda_24_s1 | 2:4 | 0.5 | 1 | 110.55814819212158 | 0.5 | 108.2 | 206.0 | 2.045 |
| final_S2_wanda_24_s2 | 2:4 | 0.5 | 2 | 109.27132919488085 | 0.5 | 109.8 | 201.3 | 2.075 |
| final_S2_wanda_24_s0_rep | 2:4 | 0.5 | 0 | 107.46964878388229 | 0.5 | 105.7 | 204.3 | 2.054 |
| final_S2_wanda_u60_s0 | unstructured | 0.6 | 0 | 90.48637841803539 | 0.59912109375 | 109.6 | 207.6 | 2.06 |
| final_S2_wanda_u60_s1 | unstructured | 0.6 | 1 | 89.90834195536661 | 0.59912109375 | 108.7 | 204.0 | 2.07 |
| final_S2_wanda_u60_s2 | unstructured | 0.6 | 2 | 89.95888093516365 | 0.59912109375 | 105.6 | 201.1 | 2.052 |
| final_S2_wanda_u60_s0_rep | unstructured | 0.6 | 0 | 90.48637841803539 | 0.59912109375 | 105.3 | 199.1 | 2.05 |
| final_S2_wanda_u70_s0 | unstructured | 0.7 | 0 | 975.6111962553042 | 0.69970703125 | 107.6 | 204.0 | 2.063 |
| final_S2_wanda_u70_s1 | unstructured | 0.7 | 1 | 1514.5040569267721 | 0.69970703125 | 105.4 | 200.8 | 2.061 |
| final_S2_wanda_u70_s2 | unstructured | 0.7 | 2 | 917.2789694390561 | 0.69970703125 | 105.5 | 199.4 | 2.048 |
| final_S2_wanda_u70_s0_rep | unstructured | 0.7 | 0 | 975.6111962553042 | 0.69970703125 | 103.0 | 197.9 | 2.058 |

<!-- MANUAL START -->
## Interpretation

**Determinism.** `attn_implementation="eager"` (default 4 threads) gave 5/5 bit-identical anchor runs (ppl
17.66059890313337, checkpoint sha256 `9b3c5e64c541388e…`). This is the same value and checkpoint prefix as Session C's
eager runs in a different container, so the configuration is reproducible across containers on this CPU model.
The 1-thread fallback was not needed. Cost: anchor quantize about 450 s, ×1.4–1.75 the default path. On Qwen
2048-token calibration the cost is much higher: about 90 min per run (quantize 4550–5400 s) and **peak RSS up to 10.2 GB**,
above the ~7 GB planned. Eager attention materialises the full 2048 × 2048 attention matrix.

**Qwen2.5-0.5B, GPTQ W3 g128, deterministic configuration.**
- Seed variability: 2048-token windows SD 0.304 (CV 1.50 %, 95 % CI [0.18, 0.87]); 512-token windows SD 0.165
  (CV 0.81 %, CI [0.09, 1.04]). Every run used distinct calibration windows and produced a distinct checkpoint, so
  with the deterministic configuration this spread is pure calibration-seed variance.
- **Window-length effect: none detectable.** 2048 − 512 = +0.044 ppl (+0.22 %), Welch p = 0.80, i.e. 0.16 pooled seed
  SDs. Qwen's effect is **indistinguishable from seed noise**. SmolLM2-135M's +5.15 ppl (+13.6 %, 7.1 × its seed SD,
  F15) is >100× larger in absolute terms.
- This agrees with REPORT.md A.6. Qwen has the same position-0 "massive activation" structure, but its
  sink-dominated `down_proj` modules carry only ~1 % of its W3 quantization error (vs ~39 % on SmolLM2), so diluting
  the sink token in the Hessian costs little.
- The single-seed default-path Qwen contrast in A.6 (20.186 → 20.658, +0.47) is superseded. On the deterministic
  path the seed-0 values are 20.218 (512) and 19.758 (2048), the opposite sign. A one-seed contrast on this model is
  dominated by seed (and, on the default path, run-to-run) noise.
- Dense fp32: 12.228462033 with eager vs 12.228462216 with SDPA (A.6): the attention kernel changes the dense value in
  the 8th digit.

**Wanda sweep (SmolLM2-135M, dense 14.488).**
- **Wanda is deterministic.** All three seed-0 repeats were bit-identical to their originals. The 50 % runs were
  already bit-identical across two containers (A.6 vs Session C). The SDs below are therefore pure calibration-seed
  variance.
- Seed sensitivity, as range / (pruned − dense increase):
  - 50 % unstructured 1.9 % (CV 0.39 %);
  - 60 % unstructured 0.8 % (CV 0.36 %);
  - 2:4 3.3 % (CV 1.4 %);
  - **70 % unstructured 53 % (CV 29 %; seeds give 917, 976 and 1515)**.
- So the seed effect is small while the model still works, and it becomes dominant once pruning breaks the model
  (70 %).
- 2:4 (50 %) is far worse than unstructured 50 % (109.1 vs 31.5) and even worse than 60 % unstructured (90.1).
  Wanda's own sparsity check reports 0.5991 and 0.6997 for the 60 % and 70 % runs (slightly under nominal, the same
  for every seed) and exactly 0.5 for 2:4.
- n = 3 per new setting: the SD intervals are wide (e.g. 2:4 SD CI [0.81, 9.75]).

**Process notes.**
- Everything ran in one resumable chain (`chain_final_S2.sh`), one job at a time, on `Intel(R) Xeon(R) Processor @
  2.10GHz` (4 vCPU, AVX512-BF16/AMX).
- One early commit (`71a831f`) included a partial GPTQModel log of the then-running first anchor run, because
  GPTQModel reopens its log per write. `commit_done.sh` was fixed to skip logs touched since the oldest running job
  started, and the log was later committed complete.
- The determinism caveat paragraphs were added to REPORT.md A.3, A.5 and A.6 (commit `3a08ebf`).
<!-- MANUAL END -->

Generated by `final_S2_report.py` at commit `34ae20e` (the commit before this file's own commit).

