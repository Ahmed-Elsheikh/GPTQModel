# Feasibility study: SLM-compression reporting-practices project on a CPU-only cloud VM

Date: 2026-10-02 (13:54–~19:00 UTC). VM: Intel Xeon @ 2.80 GHz, 4 vCPU (1 thread/core, AVX-512 + VNNI,
**no AVX512-BF16 / AMX**), 15.7 GiB RAM, ~30 GB writable disk, no GPU. Python 3.11.15.
All raw logs are in `feasibility/logs/`, machine-readable results in `feasibility/results/`, notes in `NOTES.md`.

> **Two blockers shape everything below.**
> 1. **`huggingface.co` (and its CDNs) is blocked** by the environment's egress policy. No real model weights and no
>    real datasets (WikiText-2, C4, ARC, HellaSwag) could be downloaded. Every model/quality experiment was
>    therefore run on **randomly initialised stand-ins with the exact SmolLM2-135M / SmolLM2-360M / Qwen2.5-0.5B
>    architectures** (param counts 134.5M / 361.8M / 494.0M) and **synthetic token data with the same shapes**.
>    Wall-time and memory numbers are representative (compute depends on shapes, not weight values);
>    perplexities and accuracies are **meaningless as quality numbers** (≈ vocab size / chance) and are only used to
>    check determinism, plumbing, and whether a factor changes the output at all.
> 2. **Git push to `Ahmed-Elsheikh/GPTQModel` is refused (HTTP 403, Claude GitHub App has no access).** All work is
>    committed locally on branch `feasibility`; it is *not* on GitHub yet (see §7).
>
> Hosts to add to the allowlist: `huggingface.co`, `cdn-lfs.huggingface.co`, `cas-bridge.xethub.hf.co` (HF Xet
> storage), `download.pytorch.org` (CPU-only torch wheel), `arxiv.org` (PDFs for Test 5). Optional:
> `codeload.github.com` (tarballs; `git clone` already works), `openrouter.ai` (MetaScreener's only LLM provider).

## 1. Summary

| Test | Status | Key numbers (4 vCPU, synthetic stand-ins) | Blockers / caveats |
|---|---|---|---|
| 0 env | PASS | torch 2.14.1+cu130 from PyPI (CPU index blocked); install 3m55s; venv 5.4 GB | `download.pytorch.org` blocked → 4 GB of unused CUDA libs |
| 1a install GPTQModel | PASS | `pip install gptqmodel==7.5.0`: 1m36s, pure-Python sdist, **no CUDA compile**; JIT-compiles one CPU C++ op (`pack_block_cpu`) with g++/ninja at first quantize | none |
| 1c calib.py | PASS (synthetic source only) | 128×512 dicts → GPTQModel kept 128/128, identical multiset, "Total tokens 65536, padded 0" | WikiText-2/C4 loaders written but untestable (HF blocked) |
| 1d GPTQ W4 g128, 135M | PASS-with-workaround | quantize **976–993 s** (auto dtype=bf16; 1172 s when contended), **528 s with dtype=float32**; ppl eval 40×2048: dense fp32 176 s, quantized 373–430 s; peak RSS 2.6 GB | `BACKEND.AUTO` and `TORCH_FUSED` inference kernels **crash on SmolLM2 (K=576 not divisible by 128)**; must load with `backend=BACKEND.TORCH` |
| 1e determinism / seed | PASS | seed 0 twice → **bit-identical ppl** (58944.632020…); seed 1 → 58811.203 (Δ=−0.23 %) | effect size on real models unknown |
| 1f 360M / Qwen-0.5B | PASS | GPTQ W4 fp32 quantize: 360M **1110 s** (2.1× 135M), Qwen2.5-0.5B **1233 s**; ppl eval (bf16 reload): 360M 819 s, Qwen 1207 s; peak RSS 2.4 / 5.7 GB | fp32-quantized ckpt cannot be *evaluated* in fp32 (TorchLinear is fp16/bf16 only) |
| 1g AWQ W4 135M | PASS-with-workaround (g64, not g128) | default: `ValueError: AWQ: CUDA is not available`; with `quantize(backend=BACKEND.AWQ_TORCH)`: g128 → `in_features (576) not divisible by group_size (128)`; fp32 → AwqTorchLinear fp16/bf16 only; **g64 + bf16 + AWQ_TORCH: quantize 7036 s (1 h 57 min)**, ppl eval 367 s, peak 3.2 GB | AWQ ≈ 13× GPTQ-fp32 cost; g128 impossible on SmolLM2 |
| 2a lm-eval dense | PASS (synthetic tasks) | 200 arc-like + 200 hellaswag-like items, 0-shot, bs 8, fp32: **4m02s**, 2.4 GB | real tasks need HF datasets |
| 2b lm-eval GPTQ | PASS via in-memory model | `gptqmodel=True` → K=576 crash; `backend=torch` in model_args collides with HFLM's own `backend` arg; **in-memory `HFLM(pretrained=qm.model)`: 9m07s** (2.3× dense) | CLI path unusable for SmolLM2 |
| 2c EXP3 factors | PASS | all 4 factors settable by flags/model_args; per-sample log-lik changes: bs 1 vs 8 ≤6e-5 nats; bf16 vs fp32 ≤1.49 nats; 5-shot changes predictions; max_length 512 truncates 5-shot hellaswag (Δ up to 25 nats) but not arc | bf16 is 2× *slower* than fp32 on this CPU |
| 2d wikitext | PASS-with-setting | real task: HF blocked; synthetic stand-in at default max_length (8192) & bs 8: **OOM-killed** (13.2 GB); re-run with `max_length=2048`: **16:58**, 7.5 GB | |
| 3a optimum-benchmark | PASS-with-workaround | 0.6.0 **fails on transformers 5.x** (`SpecialTokensMixin` import); works in separate venv (transformers 4.57.6). Default protocol run: 10m48s | default process launcher busy-waits → **6× latency inflation** on 4 vCPU |
| 3b/c sweep + ratio | PASS | 16 runs, 24m36s; 360M/135M ratio: prefill **2.09–2.84**, per-token decode **1.65–2.36**, generate **1.66–2.43** across 8 protocols | iterations cut to 5, 32 new tokens, OMP=3 |
| 4 Wanda | PASS-with-patch (old transformers) | 3-line patch; **only works with transformers ≤ 4.47.1** (4.48.3/4.57.6: `position_embeddings` None; 5.18: no `hf_device_map`); 32 samples: 92 s, 1.7 GB | needs its own venv; HF data loaders replaced in a driver |
| 5 MetaScreener | PARTIAL (b SKIPPED) | installs (1m20s), server starts on CPU; OpenRouter-only; Claude via OpenAI-compat endpoint = 5-line patch (untested); form = Excel template, evidence = "exact quote" checked fuzzily (>0.80 token overlap) | no `SLMREP_ANTHROPIC_API_KEY` in env; `arxiv.org` blocked |

## 2. Versions, commits, patches

| Component | Version / commit | Where | How |
|---|---|---|---|
| Python | 3.11.15 | system | |
| torch | 2.14.1+cu130 (PyPI; CPU wheel index blocked) | all venvs | pip |
| transformers | 5.18.0 (main), 4.57.6 (venv-ob), 4.47.1 (venv-w447) | | pip |
| GPTQModel | 7.5.0 (sdist, pure Python) | `/home/user/venv` | `pip install gptqmodel==7.5.0` |
| lm-evaluation-harness | lm_eval 0.4.13 | `/home/user/venv` | pip |
| optimum-benchmark | 0.6.0 | `/home/user/venv-ob` (works), `/home/user/venv` (broken on tf 5) | pip |
| locuslab/wanda | `8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6` (2023-11-03) | `third_party/wanda` | `git clone` (not on PyPI; PyPI "wanda" is an unrelated package) |
| ChaokunHong/MetaScreener | `532ee3cc10a7e22e54747d7c3413509c31b9a99b` (2026-06-11), pkg version 2.0.0a5 (PyPI has only 2.0.0a4) | `third_party/MetaScreener`, `/home/user/venv-ms` | `git clone` + `pip install ./` |
| accelerate / datasets / numpy | 1.15.0 / 5.0.1 / 2.2.6 (main; GPTQModel pins numpy==2.2.6) | | |

Full `pip freeze` per venv: `freeze_venv.txt`, `freeze_venv-ob.txt`, `freeze_venv-ms.txt` (venv-w447 = torch 2.14.1,
transformers 4.47.1, accelerate 1.15.0, datasets 5.0.1).

Patches (`feasibility/patches/`):

| Patch | Lines | Why |
|---|---|---|
| `wanda_cpu.patch` (main.py) | +3 −3 | `float16→float32`, `seqlen=min(512, max_pos)`, `cuda:0→cpu` — exactly the planned minimal patch |
| `metascreener_anthropic_compat.patch` (adapter + factory) | +3 −2 (+1 blank) | env-overridable base URL (`METASCREENER_LLM_BASE_URL`) and fallback key `SLMREP_ANTHROPIC_API_KEY`; **drafted, not applied, not run** |

No library internals were modified. Work-arounds that live *outside* libraries: `feasibility/wanda_synth_driver.py`
(monkeypatches wanda's `get_loaders` to synthetic data because HF is blocked), `lmeval_inmemory.py` (Test 2b),
explicit `backend=` arguments to GPTQModel, `hydra.job.env_set.OMP_NUM_THREADS=3` for optimum-benchmark, and a
fixed `tokenizer_config.json` in the synthetic 135M copy used under transformers 4.x (our synthetic tokenizer was
saved by transformers 5 with class `TokenizersBackend`, unknown to 4.x; real SmolLM2 ships a GPT2 tokenizer).

## 3. Measured times and peak memory

Wall clock from `/usr/bin/time -v` (process tree) or in-script `time.time()`; peak RSS = max resident set size.
`torch.set_num_threads(4)` unless noted.

### Installs
| Step | Wall | Peak RSS |
|---|---|---|
| torch 2.14.1 (PyPI, incl. CUDA libs) | 3:55 | 0.6 GB |
| gptqmodel 7.5.0 + deps | 1:36 | 0.2 GB |
| lm-eval 0.4.13 + optimum-benchmark 0.6.0 | 0:40 | 0.1 GB |
| venv-ob (torch from cache, transformers<5, optimum-benchmark) | 2:40 | – |
| venv-w447 (torch from cache, transformers 4.47.1) | 1:56 | – |
| MetaScreener (venv-ms) | 1:20 | 0.25 GB |
| synthetic models (135M+360M+0.5B) | 0:50 | 2.9 GB |

### Test 1 – GPTQModel (N=128, L=512, W4 g128, ppl on 40×2048 tokens)
| Run | dtype (quant) | quantize | ppl eval | total wall | peak RSS | ppl |
|---|---|---|---|---|---|---|
| 135M dense | fp32 | – | 176 s | – | 2.4 GB | 58774.80 (random weights) |
| 135M seed0 run a | auto→bf16 | 1172 s (contended by pip installs) | 430 s (separate eval, TorchLinear) | – | 2.6 GB | 58944.632020 |
| 135M seed0 run b | auto→bf16 | 993 s | 373 s | 23:06 | 2.6 GB | 58944.632020 (identical) |
| 135M seed1 | auto→bf16 | 976 s | 375 s | 22:49 | 2.7 GB | 58811.202808 |
| 135M seed0 | **fp32** | **528 s** | 349 s (reloaded bf16) | 9:01 + 6:06 | 2.4 GB | 58845.097 (≠ bf16-quantized 58944.632) |
| 360M seed0 | fp32 | 1110 s | 819 s (reloaded bf16) | 18:44 + 13:58 | 2.7 GB | 56037.99 |
| Qwen2.5-0.5B seed0 | fp32 | 1233 s | 1207 s (bf16) | 41:06 | 6.0 GB | 176540.2 (vocab 151936) |
| AWQ W4 **g64** 135M seed0 | bf16 (auto), AWQ_TORCH | **7036 s** (≈ 3.9 min/layer) | 367 s (AwqTorchLinear) | 2:03:37 | 3.2 GB | 59617.93 |

Per-layer pace: 135M GPTQ ≈ 33 s/layer (bf16) vs 17.5 s/layer (fp32); 360M ≈ 35 s/layer (fp32); Qwen-0.5B ≈ 52 s/layer (fp32);
135M AWQ g64 ≈ 235 s/layer (bf16). Failed AWQ attempts: AUTO backend (CUDA error, 11 s), AWQ_TORCH g128 (in_features error), AWQ_TORCH g64 fp32 (pack error after 2:29).
Reload of a quantized checkpoint: 3–6 s.

### Test 2 – lm-eval 0.4.13 (synthetic arc/hellaswag stand-ins, device cpu, seed 0)
| Run | Settings | Wall | Peak RSS |
|---|---|---|---|
| 2a dense 135M | 0-shot, limit 200+200, bs 8, fp32 | 4:02 | 2.4 GB |
| 2b GPTQ 135M (in-memory, TorchLinear bf16) | same | 9:07 | 1.8 GB |
| 2c baseline | 0-shot, limit 100+100, bs 8, fp32, max_len 2048 | 2:02 | 2.4 GB |
| 2c num_fewshot 5 | | 9:56 | 4.9 GB |
| 2c batch_size 1 | | 3:36 | 2.0 GB |
| 2c dtype bfloat16 | | 4:01 | 1.9 GB |
| 2c max_length 512 (with 5-shot) | | 8:44 | 4.1 GB |
| 2d wikitext-like rolling ll, 62 docs ≈ 280k tok, bs 8, default max_len 8192 | | OOM-kill at 3:47 | 13.2 GB |
| 2d same, max_length 2048 | | **16:58** | 7.5 GB |

### Test 3 – optimum-benchmark 0.6.0 (venv-ob)
| Run | Wall | Peak RSS |
|---|---|---|
| 3a default protocol (process launcher, OMP 4, warmup 10, ≥10 iters/≥10 s, 100 new tokens), 135M bs1 seq128 | 10:48 | 1.9 GB |
| diagnostic, 20 new tok / 3 iters: process launcher OMP 4 → per-token **0.344 s**; inline launcher → **0.053 s**; process OMP 3 → **0.062 s**; plain `perf_counter` loop → 0.056–0.063 s | 0:17–1:45 each | |
| 3b sweep 2 models × warmup{0,10} × bs{1,4} × seq{128,512} (process, OMP 3, 5 iters, 32 new tok) | 24:36 (16 runs) | 3.2 GB |

### Test 4 – Wanda (unstructured 50 %, 32 samples × 512, eval 40×512)
| transformers | Result | Wall | Peak RSS |
|---|---|---|---|
| 5.18.0 | `AttributeError: 'LlamaForCausalLM' object has no attribute 'hf_device_map'` | 0:12 | 1.7 GB |
| 4.57.6 | `TypeError: cannot unpack non-iterable NoneType` (`position_embeddings`) | 0:09 | 1.9 GB |
| 4.48.3 | same TypeError | – | – |
| **4.47.1** | **works**, sparsity 0.5000 exact, ppl computed | **1:38** (92 s in driver) | 1.8 GB |

## 4. Extrapolated CPU budget for the full design

Generated by `feasibility/budget.py` from the measured unit costs (single job on the 4-vCPU VM; 360M lm-eval time
is extrapolated with the measured 360M/135M forward-cost ratio from the ppl evals). Arc+hellaswag time scales
linearly in items (limit 100 → 122 s, limit 200 → 242 s). GPTQ W3 is assumed to cost the same as W4 (not measured).

| Unit (measured unless *est.*) | 135M | 360M |
|---|---|---|
| GPTQ W4/W3 quantize, N=128, L=512, fp32 | 528 s | 1110 s |
| AWQ W4 quantize (g64, bf16, AWQ_TORCH) | 7036 s | *est.* 14798 s (× GPTQ 360M/135M ratio 2.10) |
| ppl 40×2048, quantized model (TorchLinear, bf16) | 349 s | 819 s (ratio 2.35) |
| lm-eval 2 tasks × 500 items, quantized (in-memory, bs 8) | 535 s × 2.5 = 1337 s | *est.* × 2.35 = 3141 s |
| lm-eval 2 tasks × 500 items, dense fp32 | 242 s × 2.5 = 605 s | *est.* 1421 s |
| optimum-benchmark run (5 iters, 32 new tokens) | 1.5 min (mean of 16-run sweep, both models) | |

**One EXP1/EXP2 cell = quantize + ppl + 2×500 items:**
- 135M GPTQ: 2213 s = **37 min**
- 360M GPTQ: 5070 s = **85 min**
- 135M AWQ: 8722 s = **145 min**
- 360M AWQ: 18759 s = **313 min**

**EXP1** = 2 models × {GPTQ W4, GPTQ W3, AWQ W4} × 10 seeds: 135M 10×(2×37+145) min = 36.5 h; 360M 10×(2×85+313) min = 80.3 h → **117 h**
**EXP2** = same × 3 sources × 5 seeds: 135M 54.8 h; 360M 120.4 h → **175 h** (≈ 117 h if one source reuses EXP1 cells)
**EXP3** = 2 models × 16 dense lm-eval variants (8 zero-shot at ~1.5× the 0-shot baseline, 8 five-shot at ~4.6×): 135M 8.2 h; 360M 19.3 h → **27 h**
**EXP4** = 3 models × 24 protocols × 1.5 min × 1.3 → **2.4 h** (with optimum-benchmark's default ≥10 iterations/≥10 s/100 tokens, ~3–5× more)

**Total ≈ 322 h ≈ 13.4 days of continuous single-VM compute.**

**Reduced design** (200 items/task in seed sweeps; EXP1 = GPTQ W4 ×10 seeds + W3 ×3 + AWQ ×3; EXP2 = GPTQ W4 × 3 sources × 3 seeds; EXP3 = 135M 10 zero-shot + 2 few-shot variants, 360M 6 variants; EXP4 unchanged): EXP1 37 h, EXP2 11 h, EXP3 8 h, EXP4 2.4 h → **59 h** (≈ 15 h wall on 4 parallel VMs).

**Verdict: the full design does not fit in an interactive session** — not even close (sessions are reclaimed when
idle and background jobs do not survive). Individual cells (37–85 min for GPTQ, 2–5 h for AWQ) do fit, so the work
must be organised as a restartable job queue that writes each cell's JSON as soon as it finishes, ideally over
several parallel sessions/VMs.

**What I would cut / change, in order of payoff:**
1. **AWQ is the cost driver** (≈ 13× fp32 GPTQ, 7× bf16 GPTQ on this CPU, and only works with g64 on SmolLM2 — see §6). Keep AWQ at 3 seeds
   (or drop it from the CPU arm) and keep GPTQ W4 as the 10-seed workhorse.
2. Quantize GPTQ in **float32**, not GPTQModel's auto-selected bf16: 1.9× faster on this CPU (no AVX512-BF16).
3. **200 items per task** in the EXP1/EXP2 seed sweeps (lm-eval is ~60 % of a GPTQ cell); 500 only for anchor cells.
4. GPTQ W3 at 3 seeds; EXP2 at 3 seeds per source, GPTQ W4 only.
5. EXP3: run the 0-shot factorial on 135M plus 2 few-shot variants; on 360M only the factors that move 135M.
6. EXP4 is cheap (≈ 2–3 h) — keep it whole, but fix the launcher/thread protocol first (§6.6).

## 5. Recommended setup script

`feasibility/setup.sh` (bash, idempotent, always `exit 0`, prints WARN lines on failure). It creates `venv`
(torch + gptqmodel 7.5.0 + lm-eval 0.4.13), `venv-ob` (transformers 4.57.6 + optimum-benchmark 0.6.0),
clones wanda at the pinned commit and applies `wanda_cpu.patch`, optionally MetaScreener (`WITH_METASCREENER=1`), and
regenerates the synthetic stand-ins. **Timing caveat:** with torch from PyPI the first venv alone takes ~5.5 min on a
cold cache (torch 3:55 + gptqmodel 1:36); the two-venv script will exceed the 5-minute target unless
`download.pytorch.org` is allowlisted (CPU wheel ≈ 0.2 GB) or the session snapshot keeps the pip cache. Wanda's
transformers-4.47.1 venv is *not* in the script (create it on demand: `pip install torch==2.14.1 transformers==4.47.1
accelerate datasets`, ~2 min from cache). Verified here only on the already-provisioned VM (warm path: exit 0 in 41 s, both import checks OK, wanda patch applied); a cold run could not be tested because the 30 GB allowance was nearly used up (see §6.11).

## 6. Surprises and contradictions to the design assumptions

1. **No Hugging Face access** — the design assumes HF Hub models/datasets; nothing quality-related can be measured
   until the hosts are allowlisted.
2. **GPTQModel's CPU inference kernels reject SmolLM2's hidden size.** 576 (135M) and 960 (360M) are not multiples of
   128; `BACKEND.AUTO` selects `TorchAtenLinear` (int4pack) which raises `expect K to be divisible by qGroupSize`;
   `TORCH_FUSED` too. Only `BACKEND.TORCH` (dequantise-then-matmul, fp16/bf16 only) works — and it is 2.1–2.4× slower
   than the dense fp32 model. Quantization itself is fine (partial last group). This is relevant to the reporting
   study itself: "which kernel evaluated the quantized model" is a hidden protocol variable.
3. **GPTQModel auto-selects bf16 on CPU**, which is *slower* on CPUs without AVX512-BF16/AMX (quantize 1.9×, lm-eval
   2×). The dtype used during GPTQ is another unreported protocol variable.
4. **AWQ on CPU is not automatic**: `quantize()` with `BACKEND.AUTO` picks CUDA-only `AWQ_GEMM` and raises
   `AWQ: CUDA is not available`; `BACKEND.AWQ_TORCH` must be passed explicitly. Then AWQ **cannot use group size 128
   on SmolLM2 at all** (`awq_processor.py: Expected in_features (576) to be divisible by group_size (128)`), and its
   CPU packing kernel `AwqTorchLinear` rejects float32 (`only supports [torch.float16, torch.bfloat16]`). The only AWQ
   configuration that runs is **g64 + bf16 + AWQ_TORCH**, and it is ~7× slower than GPTQ in the same dtype
   (AWQ's scale search runs ~20 grid points × 4 scaling groups of layer forwards per layer). So "GPTQ W4 g128 vs
   AWQ W4 g128" as planned is not possible for SmolLM2 with GPTQModel; either use g64 for both methods or accept
   mismatched group sizes. (Qwen2.5-0.5B, hidden 896 = 7×128, would allow g128.)
5. **Determinism is good**: two GPTQ runs with the same seed gave bit-identical perplexity; seed 0 vs 1 differ.
   Timing is noisy though: the same run took 993 s alone vs 1172 s while a pip install ran (+18 %).
6. **optimum-benchmark 0.6.0 is incompatible with transformers 5.x** (needs < 5) and its **process launcher
   busy-waits in the parent**, stealing a core: on 4 vCPU with 4 OMP threads, latencies inflate **~6×**
   (decode 0.34 s/token vs 0.053 s). This is itself an EXP4-style finding: the measured "speedup" depends on
   launcher × thread count. The `inline` launcher measures correctly but crashes at the end when serialising the
   config (`PerTokenLatencySessionTrackerLogitsProcessor is not JSON serializable`) — the report files are written
   before the crash.
7. **lm-eval's `gptqmodel=True` path cannot select a GPTQModel kernel**: `backend=` in `model_args` is captured by
   HFLM's own `backend` parameter. Use the in-memory model object.
8. **lm-eval rolling-loglikelihood (wikitext) OOMs at default settings** for SmolLM2 (max_length defaults to
   8192 → 8 × 8192 × 49152 fp32 logits ≈ 13 GB). `max_length` must be set; it is yet another protocol variable that
   changes perplexity.
9. **Wanda only runs with transformers ≤ 4.47.1** (released Dec 2024) — the "current transformers" assumption fails;
   a separate venv is required. The patch stayed minimal (3 lines). Wanda also hard-codes C4/WikiText downloads
   from HF and uses 128 pre-allocated calibration slots.
10. **MetaScreener is OpenRouter-only** (hard-coded `https://openrouter.ai/api/v1`, 17 open-weight models in
    `configs/models.yaml`); retrieval talks to PubMed/EuropePMC/OpenAlex/Semantic Scholar/Scopus/Unpaywall; OCR uses
    litellm. Extraction forms are **Excel templates**, not YAML (YAML is only used for domain plugins such as
    `amr_v1`). It runs **two models (A/B) plus an optional arbiter** per field. Evidence is requested as "exact quote"
    but validated **fuzzily** (token-overlap > 0.80), so quotes are not guaranteed verbatim.
11. Disk: the PyPI torch wheel brings ~4 GB of CUDA libraries per venv; four venvs ≈ 20 GB of the 30 GB allowance.
    By the end free space was 4.2 GB (three torch venvs ≈ 6 GB each, pip cache 3.2 GB, MetaScreener clone 1.1 GB with
    1 GB of `experiments/`); `pip cache purge` recovered 3.2 GB. Plan for one torch install shared across venvs, or a
    CPU-only wheel.

## 7. State of the branch

The task asked for branch `feasibility`; the session's designated branch is `claude/feasibility-study-tut4ve`.
All commits are on local branch `feasibility`. `git push` to `https://github.com/Ahmed-Elsheikh/GPTQModel` returns
**403: "Claude doesn't have GitHub access to Ahmed-Elsheikh/GPTQModel"** for both branch names, so nothing is on
GitHub. Fix: install the Claude GitHub App on the repository / reconnect GitHub at
https://claude.ai/connect-github, then push `feasibility` (the working tree is in this container only and will be
lost when it is reclaimed).

## Appendix: Test 5 answers (MetaScreener @532ee3c)

- **Providers / models / hosts:** all screening & extraction LLM calls go through `OpenRouterAdapter`
  (`httpx`, `POST https://openrouter.ai/api/v1/chat/completions`, key `OPENROUTER_API_KEY`). Models
  (`configs/models.yaml`): deepseek-v3.2, qwen3-235b-a22b, kimi-k2.5, kimi-k2-0905, llama-4-maverick, glm-5-turbo,
  mimo-v2-pro, minimax-m2.7, hermes-4-405b, llama-3.1-nemotron-70b, cogito-v2.1-671b, jamba-large-1.7, gemma-3-27b,
  gemma-4-26b-a4b, glm-5.1, mistral-small-2603, phi-4. Other hosts: eutils.ncbi.nlm.nih.gov, www.ebi.ac.uk
  (EuropePMC), api.openalex.org, api.semanticscholar.org, api.elsevier.com, api.unpaywall.org, doi.org; litellm
  (OCR/VLM + token counting).
- **Switch to Claude by configuration?** No. There is no provider switch; the base URL is a module constant and
  the key env-var name is read in 5 places. A 5-line patch (`metascreener_anthropic_compat.patch`) makes the base URL
  and key configurable; pointing it at Anthropic's OpenAI-compatible endpoint (`https://api.anthropic.com/v1/`) plus a
  `models.yaml` entry with a Claude model id should work: a probe with a dummy key returned
  `{"error":{"code":"authentication_error","message":"Invalid Anthropic API Key"}}`, i.e. the host is reachable and
  accepts the request format. Untested beyond that: `SLMREP_ANTHROPIC_API_KEY` is not set in this environment.
  The web-UI paths (`api/routes/settings.py` key validation against openrouter.ai, `screening_helpers.py`,
  `quality.py`) would need more edits (~20–30 lines) if the UI is used rather than the Python API.
  Alternatively, OpenRouter itself serves `anthropic/claude-*` models — pure configuration — but needs an
  OpenRouter key and `openrouter.ai` on the allowlist.
- **YAML form?** No — the extraction schema is compiled from an **.xlsx template** (`module2_extraction/compiler`).
- **Evidence quotes?** Yes: prompts require `"evidence": {"field": "<exact quote from text>"}`;
  `validation/source_coherence.py` accepts evidence with token-overlap ratio > 0.80, so non-verbatim quotes pass.
- **5b (2 arXiv papers) — SKIPPED:** no Anthropic key, and `arxiv.org` is blocked. Cost/time per paper not measured.
  What it would take: allowlist `arxiv.org`, provide the key, apply the 5-line patch, write a 5-column .xlsx template,
  drive `/api/v2/extraction/sessions` (create → template → pdfs → run → results).
