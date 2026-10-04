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
| 2:4 (50 %) | 2 | 109.0139 | 2.1839 | 2.00 % | 107.4696 | 110.5581 | [0.9743, 69.6886] | 3.0885 | 3.27 % | – |

Per run:

| Run | Sparsity type | Ratio | Seed | WT-ppl | Measured sparsity | Prune s | Wall s | Peak RSS GB |
|---|---|---|---|---|---|---|---|---|
| final_S2_wanda_24_s0 | 2:4 | 0.5 | 0 | 107.46964878388229 | 0.5 | 107.4 | 207.2 | 2.058 |
| final_S2_wanda_24_s1 | 2:4 | 0.5 | 1 | 110.55814819212158 | 0.5 | 108.2 | 206.0 | 2.045 |

<!-- MANUAL START -->
<!-- MANUAL END -->

Generated by `final_S2_report.py` at commit `db8a2b4` (the commit before this file's own commit).

