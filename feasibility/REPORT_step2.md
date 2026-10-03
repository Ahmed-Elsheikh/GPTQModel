# Step 2 (parallel worker): Wanda seed sweep, C4 calibration, AWQ, Qwen2.5-0.5B, real lm-eval, optimum-benchmark launchers

Date: 2026-10-03, 08:10–09:25 UTC (§7: 10:52–11:30; §8: 11:43–13:20). 4 vCPU, CPU-only, **real weights and real data** (HF Hub now
reachable). Runs were one at a time in this container. The Step 1 session ran in a different container, so the two sessions
did not compete for CPU. Every run went through `runlog.py` (wall time and peak RSS of the child tree).
Raw records: `results/real_step2_{wanda,c4,awq,qwen,lmeval,ob}.jsonl`. Logs: `logs/real_s2_*.log`.
Scripts: `chain_step2a2.sh` (Wanda), `chain_step2b.sh` (tasks 2–6), `wanda_real_driver.py`, `step2_summarize.py`.
CPU: Intel Xeon @ 2.10 GHz with AVX512-BF16 and AMX. That is the **same model as Session B's Step 1 VM**, and all 115
`lscpu` flags match Session B's per-run records (`results/real_step1b.jsonl`). It differs from the Part B study VM
(`env.json`: 2.80 GHz, no BF16/AMX).
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
- Quantize time: 257 s here vs 292–348 s for the five Step 1 W4 runs (same code, same CPU model, other container).
- **Caveat:** the exact same seed-0 command does not always give the same checkpoint (§8, Determinism). Its default
  outcomes are 17.7448 (Step 1's value) or 17.8110. The C4 gap is therefore +1.11 to +1.18, still ≈ 21× the seed
  range. §7 has the 2×2.

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

## 7. Add-on: calibration source × evaluation set (GPTQ W4 g128 seed 0, SmolLM2-135M)

**C4-validation set.** GPTQ convention (`IST-DASLab/gptq` `datautils.get_c4`):
- source `en/c4-validation.00000-of-00008.json.gz`, `random.seed(0)`;
- 256 times: draw random docs until one has ≥ 2048 tokens, then take a random 2048-token window inside it;
- ids cached as `256×2048 int32` (fingerprint `cbe443ebebeeceeb`, main-venv tokenizers 0.23.2; SmolLM2 adds no BOS).

Finding the 256 windows took 8 288 doc draws (most C4 docs are < 2048 tokens). They come from 234 distinct docs, so a few
docs supply two windows, which may overlap. Script: `step2_c4val.py`; chain: `chain_step2c.sh`;
records: `results/real_step2_c4val.jsonl`.

**WikiText-2 set.** The Step 1 40 × 2048 test windows, as everywhere else in this report.

Both checkpoints were built **in this container**, and quantized models are evaluated in bf16 with `BACKEND.TORCH`, as above.
The WikiText-calibrated checkpoint was rebuilt with the exact Step 1 command (`s2_gptq_w4_135m_wt_s0_rebuild`: quantize
272 s, run 379 s, peak 2.90 GB).

| calibration ↓ / eval → | WikiText-2 test (40×2048) | C4 val (256×2048) |
|---|---|---|
| dense fp32 (reference) | 14.4878 | 18.7520 (bf16: 18.7676) |
| **WikiText-2 train, s0** | **17.8110** (+3.32) | 23.9849 (+5.23) |
| **C4 train, s0** | 18.9251 (+4.44) | **23.3427** (+4.59) |

- **Clean crossover: each calibration source wins on its own domain.**
  - On WikiText-2, WikiText-2 calibration is better by 1.114 ppl, i.e. 25 % less degradation.
  - On C4, C4 calibration is better by 0.642 ppl, i.e. 12 % less degradation.
  - Both gaps are 12–21× the Step 1 W4 seed range (0.054), so for 135M the calibration-source effect dwarfs the seed effect.
  - In-domain calibration matters more for WikiText, which is narrower than C4.
- **Which result you report depends on the eval set.** A paper that evaluates only on WikiText-2 would call
  WikiText calibration clearly better. Averaged over both sets the two are within 0.24 ppl:
  20.90 for WikiText calibration vs 21.13 for C4 calibration.
- **Cost of C4-val ppl.** 256 windows = 6.4× the WikiText set.
  - Quantized (bf16): 279–288 s.
  - Dense: fp32 523 s, bf16 204 s. On this AMX CPU bf16 is 2.6× *faster*, the opposite of `REPORT.md` §6.3.
  - Building the cache: 80 s.

**Reproducibility note (revised; see §8).** The first rebuild of the WikiText s0 checkpoint gave 17.8110, not
Step 1's 17.7448. An earlier version of this section blamed a CPU difference between containers. **That was wrong.**
- Session B ran on the same CPU model with identical flags.
- Repeating the command in this container reproduces Step 1's value exactly in 2 of 4 runs.
- The cause is **run-to-run nondeterminism of the default GPTQModel CPU path**, not hardware.
- The WikiText-calibrated cells above use the 17.8110 checkpoint. With the other default outcome, the in-domain
  WikiText gap would be +1.18 instead of +1.11. Its C4-val value was not measured.

## 8. Determinism

**Setup.** The exact Step 1 command: SmolLM2-135M, GPTQ W4 g128, WikiText-2 seed 0, fp32 quantize, bf16 TorchLinear eval
on the cached Step 1 ids (calibration fingerprint `07db06da…`, eval fingerprint `7da3b34b…`).
- Each run goes through `step2_det.py`, which applies one setting and then runs `quant_eval.py` unchanged.
- Runner: `chain_step2d.sh`. Comparison: `step2_detcmp.py`. Records: `results/real_step2_det.jsonl`.
- Columns: the checkpoint is compared by SHA-256 over its safetensors. GPTQ losses are compared per (layer, module),
  because GPTQModel quantizes `gate_proj` and `up_proj` concurrently and logs them in completion order.
- All runs were in this container, one at a time.

| run | setting | threads | attention | ppl (40×2048 WikiText-2) | checkpoint sha | L0 `o_proj` loss | first module differing from Step 1 | quantize s |
|---|---|---|---|---|---|---|---|---|
| Step 1 (Session B) | default | 4 | sdpa | 17.744842947166127 | – | 5.59e-8 | – | 348 |
| rebuild 1 | default | 4 | sdpa | 17.811016451979064 | `72b61b9e` | 5.60e-8 | L0 `o_proj` | 272 |
| rep 2 | default | 4 | sdpa | 17.811016451979064 | `72b61b9e` | (log lost)† | (identical ckpt to rebuild 1) | 268 |
| rep 3 | default | 4 | sdpa | **17.744842947166127** | `3c9ab62e` | 5.59e-8 | **none** (all 210 losses = Step 1) | 246 |
| det 1 | `use_deterministic_algorithms(True)` | 4 | sdpa | 17.744842947166127 | `3c9ab62e` | 5.59e-8 | none | 247 |
| det 2 | `use_deterministic_algorithms(True)` | 4 | sdpa | 17.744842947166127 | `3c9ab62e` | 5.59e-8 | none | 267 |
| thr1 a | `set_num_threads(1)` | 1 | sdpa | 17.790198509693287 | `5952c417` | 5.61e-8 | L0 `k_proj` (first module) | 571 |
| thr1 b | `set_num_threads(1)` | 1 | sdpa | 17.790198509693287 | `5952c417` | 5.61e-8 | L0 `k_proj` | 545 |
| eager 1 | `attn_implementation="eager"` | 4 | eager | 17.66059890313337 | `9b3c5e64` | 5.60e-8 | L0 `o_proj` | 315 |
| eager 2 | `attn_implementation="eager"` | 4 | eager | 17.66059890313337 | `9b3c5e64` | 5.60e-8 | L0 `o_proj` | 298 |

† My mid-run `git pull --autostash` replaced the live log file, so the process kept writing to an unlinked inode.
The checkpoint is byte-identical to rebuild 1's, so its losses are rebuild 1's.

**1. Are 3 rebuilds in this container bit-identical? No.**
- Default runs in this container: rebuild 1, rep 2 and rep 3, plus Session B's Step 1.
- They land on **two discrete outcomes**, each bit-identical when it recurs:
  - A = 17.811016 (2 of 3 here, checkpoint `72b61b9e`);
  - B = 17.744843 (1 of 3 here, checkpoint `3c9ab62e`, and Step 1's exact value with all 210 losses identical).
- The A/B gap is 0.066, more than Step 1's whole five-seed range (0.054).
- **The original Step 1 result is reproducible in this container. It is simply one of at least two outcomes the
  default path can produce.**

**2. Which setting makes repeats identical?** Each setting was run twice, and all three gave a bit-identical pair.
Two runs per setting is weak evidence, especially for a two-outcome process:

| setting | result | interpretation |
|---|---|---|
| `use_deterministic_algorithms(True)` | 2/2 outcome B | **Inconclusive.** B is also a default outcome, so this does not show the flag changed anything. No op raised a "no deterministic implementation" error. |
| `set_num_threads(1)` | 2/2 = 17.790199, a third value | Consistent with removing a thread-count-dependent reduction order. It changes results from the very first module (L0 `k_proj` loss 2.9682e-6 vs 2.9675e-6), so single-thread GEMM/Hessian accumulation differs numerically from 4-thread. Costs 2.2× quantize time. |
| `attn_implementation="eager"` | 2/2 = 17.660599, a fourth value | Consistent with the nondeterminism being in the SDPA attention path: the first divergence is L0 `o_proj`, the first module whose Hessian input passes through attention. Costs about 1.15× quantize time and 1.6× run time. |

- **Where it starts.** Under the default settings, q/k/v_proj losses (computed from the pre-attention layernorm
  output) are always identical. The first difference is always L0 `o_proj`, whose input is the attention output
  (5.59e-8 vs 5.60e-8). Everything downstream then diverges.
- **Likely mechanism (not proven).** GPTQModel quantizes modules and replays layer forwards concurrently
  (`auto_forward_data_parallel=True`, parallel `gate_proj`/`up_proj`). Concurrent work would give the SDPA kernel a
  varying share of the 4 intra-op threads, and so a varying fp32 reduction order.
- **Recommendation.** Use `attn_implementation="eager"` or 1 thread, and confirm with ≥ 5 repeats.
  `auto_forward_data_parallel=False` is the next candidate to test.
- **Spread caused by implementation alone.** Over the same seed and data, the four distinct values span
  17.6606–17.8110 = **0.150 ppl**. That is **2.8× the five-seed range** and 7× the seed SD.
  - The Step 1 W4 seed SD (0.021) therefore mixes calibration-seed variance with this run-to-run noise, and does
    not cleanly measure seed sensitivity.
  - "Which attention kernel and how many threads" belong in the reported protocol.
  - EXP1 seed sweeps should fix them, and should verify determinism with repeats before attributing variance to
    seeds.

**3. Cross-container check (requested only if the repeats were identical; given here because the CPU question
needed an answer).**

| item | this container | Session B (Step 1) |
|---|---|---|
| CPU model | Intel(R) Xeon(R) Processor @ 2.10GHz | same (`REPORT.md` Part A; `real_step1b.jsonl` `cpu_model`) |
| `lscpu` flags | 115 flags incl. avx512_bf16, avx512_fp16, amx_bf16/int8/tile | **identical set** (diff empty) |
| torch | 2.14.1+cpu, MKL 2024.2, oneDNN (MKL-DNN) v3.12.0, OpenMP 4.5, "CPU capability usage: AVX512" | 2.14.1+cpu (build config not recorded) |
| gptqmodel / transformers | 7.5.0 / 5.18.0 | 7.5.0 / 5.18.0 |

There is no evidence of a hardware or library difference. Step 1's exact result recurs here.

**Status of earlier follow-ups.**
- **Tokenizer-version effect on dense perplexity: measured (§1).**
  - tokenizers 0.21.4 (wanda venv, own tokenization) gives dense fp32 ppl **14.487248**; 0.23.2 (Step 1 windows) gives **14.487794**.
  - Δ = −0.00055 (−0.004 %). Only eval windows 33–40 differ.
  - The effect is negligible for dense, but 0.114 for Wanda s0, where the calibration windows also change.
  - No dedicated dense rerun beyond these two runs was done.
- **Wanda 2:4 / 60 % / 70 % sweep: not started.** It is not in this session's queue and no runs exist. Estimated cost
  on this VM: about 3 min per run (prune ~90 s + eval ~70 s). Seeds 0–4 × {2:4, 60 %, 70 %} = 15 runs ≈ 45 min,
  ready to launch with `chain_step2a2.sh`-style flags (`--sparsity_type 2:4 --sparsity_ratio 0.5`;
  `--sparsity_ratio 0.6/0.7`).

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

**Add-on (§7):**
- 2×2 calibration × eval, GPTQ W4 s0, built in one container:
  - WikiText-calibrated: 17.81 on WikiText-2, 23.98 on C4-val.
  - C4-calibrated: 18.93 on WikiText-2, 23.34 on C4-val.
  - Each source wins in-domain, by 1.11 and 0.64 ppl (12–21× the seed range).
- Earlier CPU explanation retracted (same CPU, identical flags).

**Add-on (§8, Determinism):**
- The exact Step 1 seed-0 command is **not deterministic** on the default path (4 threads, SDPA). It gives two
  outcomes, 17.8110 or 17.7448 (Step 1's exact value), diverging at layer-0 `o_proj`.
- With eager attention (17.6606) or 1 thread (17.7902), 2/2 repeats were bit-identical.
- `use_deterministic_algorithms(True)` was inconclusive.
- Implementation-only spread of 0.150 ppl ≈ 2.8× the five-seed range.
