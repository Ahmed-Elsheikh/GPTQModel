# S1 final runs: STOPPED at the anchor gate

Date: 2026-10-03, 15:16–15:36 UTC. Branch `claude/quirky-ramanujan-i0hree`. Anchor results are in commit `97f8bae`
(`results/final_S1_runs.jsonl`, `logs/final_S1_anchor_{a,b}.log`).
Driver: `final_S1_run.py`. Resumable chain: `final_S1_chain.sh`.

> **The configuration chosen for S1, `attn_implementation="eager"` with the default thread count, is NOT deterministic.**
> The two anchor runs gave different checkpoints and different perplexities. Per the instructions, S1 stopped before any
> main run (dense, W3, W4), with no fallback to another configuration. No S1 quality numbers exist.

## Anchor check

Setup:
- SmolLM2-135M, GPTQ W4 g128, seed 0, 128 × 512 WikiText-2 train windows from the cached ids.
- Calibration fingerprint `07db06dabcb03c98`; WT-ppl on the cached 40 × 2048 test windows (fingerprint `7da3b34bdebd1923`).
- Quantized in fp32 with quantize backend AUTO (as in `quant_eval.py`), reloaded with `BACKEND.TORCH` in bf16.
- `attn_implementation="eager"` at quantization and at evaluation (verified from the model config), `torch.set_num_threads(4)`.
- Hardware: Intel Xeon @ 2.10 GHz (AVX512-BF16, AMX), torch 2.14.1+cpu, GPTQModel 7.5.0, transformers 5.18.0.
  The full CPU flags and versions are in each JSON line.

| run | WT-ppl | checkpoint sha256 (16) | quantize s | peak RSS GB | equals anchor 17.66059890313337? |
|---|---|---|---|---|---|
| anchor_a | **17.632548589483903** | `e2b8938da584dd6d` | 332.3 | 2.73 | **no** (new, fifth distinct value) |
| anchor_b | 17.66059890313337 | `9b3c5e64c541388e` | 334.1 | 2.72 | yes (same checkpoint as REPORT_step2 §8 eager 1/2) |

**Gate result: FAIL.** The two runs are not bit-identical: they produce different checkpoints, and Δ ppl = 0.028.

Where the runs diverge:
- GPTQModel's per-module losses for anchor_b equal those of REPORT_step2 §8 `eager1` in all 210 modules.
- anchor_a matches anchor_b for all of layer 0's attention modules (q/k/v/o_proj). It first differs at **layer 0
  `mlp.gate_proj`** (loss 2.25799e-5 vs 2.25791e-5), and 206 of 210 losses differ in total.
- With eager attention, the nondeterminism therefore moves from `o_proj` (default SDPA path) to the MLP. This fits
  run-to-run variation in GPTQModel's concurrent processing (`gate_proj`/`up_proj` are quantized in parallel and share
  the 4 intra-op threads), not an attention-kernel effect alone.
- The earlier 2-of-2 agreement for eager (REPORT_step2 §8) was a coincidence of a small sample.

## Distinct WT-ppl values now seen for this one command (seed 0, 128×512, same data)

| configuration | values (runs) |
|---|---|
| default (4 threads, SDPA) | 17.811016 (2), 17.744843 (2: Step 1 and rep 3) |
| eager, 4 threads | 17.660599 (3: eager 1, eager 2, anchor_b), **17.632549 (1: anchor_a)** |
| `use_deterministic_algorithms(True)` | 17.744843 (2) |
| 1 thread | 17.790199 (2) |

The implementation-only spread is now 17.6325–17.8110 = 0.178 ppl, which is 3.3× the Step 1 five-seed range (0.054).

## Not run

- Dense, GPTQ W3 seeds 0–9, GPTQ W4 seeds 0–4 at 128 × 2048, C4-ppl and lm-eval: none of these started.
  The chain resumes and skips the completed anchors if restarted with a new gate.
- The cached C4-validation ids already exist (`c4val_256x2048_s0.npy`, fingerprint `cbe443ebebeeceeb`); no rebuild was needed.

## S2 cross-check

`feasibility/REPORT_final_S2.md` did not exist on the branch at 15:40 UTC. S2 has pushed its chain and scripts, but not
yet its report, so its 5-repeat eager confirmation could not be compared.

## What would unblock S1

Single-thread quantization (`set_num_threads(1)`) is the only configuration whose observed repeats (2/2) do not
contradict determinism. It costs about 2.2× the quantize time, and two runs are again weak evidence.
Two candidates would need ≥ 5 bit-identical repeats before being adopted:
- `GPTQConfig(auto_forward_data_parallel=False)` combined with eager attention;
- single-threaded GPTQ module processing.

This is your call; I did not fall back to it.
