# Running notes (raw log of findings; REPORT.md is the curated version)

## Network (2026-10-02 13:55 UTC)
- BLOCKED (egress proxy answers 403 to CONNECT): download.pytorch.org, huggingface.co, cdn-lfs.huggingface.co,
  cas-bridge.xethub.hf.co, arxiv.org, export.arxiv.org, codeload.github.com (403), openrouter.ai, api.openai.com.
- OK: pypi.org / files.pythonhosted.org, github.com via git (clone / ls-remote), api.anthropic.com (no-proxy).
- See logs/network_probe.log.
- Consequence: CPU-only torch wheel not installable; installed PyPI torch 2.14.1+cu130 (bundles ~4 GB of
  unused CUDA libs, works on CPU). NO model weights or datasets can be downloaded from the HF Hub.

## Test 0
- torch install from PyPI: 3m55s wall, 0.6 GB peak RSS, venv = 5.4 GB.

## HF Hub blocked -> synthetic stand-ins (14:00)
- AutoConfig.from_pretrained("HuggingFaceTB/SmolLM2-135M") -> OSError; load_dataset("wikitext") -> ProxyError 403.
- make_synth_models.py: random-init models with configs recalled from memory; param counts 134.5M / 361.8M / 494.0M
  match the published sizes (135M / 360M / 0.49B), so the shapes (and hence compute cost) should be right.

## Test 1 (GPTQModel 7.5.0)
- pip install gptqmodel==7.5.0: sdist only, but pure-python build (no CUDA compile, 0 nvcc lines); 1m36s.
  Pulled transformers 5.18.0, torchao 0.18, datasets 5.0.1, numpy 2.2.6 (downgrade from 2.4.6).
- At first quantize it JIT-compiles a CPU C++ op `pack_block_cpu` with ninja/g++ (cached in ~/.cache/gptqmodel).
- Loader auto-chose dtype bfloat16 ("Auto dtype (CPU + Torch Fused)"); CPU has avx512 but NO avx512_bf16/AMX.
- Calibration: 128x512 pre-tokenised dicts -> prepare_dataset kept 128/128, identical multiset; log "Total tokens: 65536,
  padded 0". GPTQModel warns "Calibration dataset size should be more than 256".
- 135M seed0 run a: dense fp32 ppl eval (40x2048) 176.5 s; quantize 1171.7 s (~35-40 s/layer x 30); save 0.8 s;
  peak RSS 2.45 GB.
- Reload with BACKEND.AUTO picked TorchAtenLinear -> RuntimeError `_weight_int4pack_mm_cpu: expect K to be divisible by
  qGroupSize, got K:576, qGroupSize:128` (SmolLM2-135M hidden=576 is not a multiple of 128). Applies to the REAL model.

## Test 2 (lm-eval 0.4.13)
- Real tasks fail: HF Hub blocked (OSError config / ProxyError). Synthetic stand-in tasks via --include_path.
- gptqmodel=True path: AUTO kernel -> K=576 int4pack error; `backend=torch` in model_args collides with HFLM's own
  `backend` (causal/seq2seq) arg -> AssertionError. Fallback = in-memory model (lmeval_inmemory.py).
- wikitext-style loglikelihood_rolling at default max_length (= model max 8192) & bs 8 -> OOM-killed (rc 137, 13.9 GB).
## Test 3 (optimum-benchmark 0.6.0)
- Fails on transformers 5.x: `cannot import name 'SpecialTokensMixin'`. Works in separate venv with transformers 4.57.6.
- Default process launcher busy-waits in the parent (`while alive and not poll(): pass`, launcher.py:56) ->
  on 4 vCPU with OMP_NUM_THREADS=4 latencies inflate ~6x (per-token 0.34 s vs 0.053 s inline / 0.062 s OMP=3).
## Test 4 (wanda 8e8fc87)
- tf 5.18: AttributeError hf_device_map (device_map="auto" on CPU-only). tf 4.57.6: position_embeddings None.
  tf 4.48.3: same. tf 4.47.1: WORKS (3-line patch), 92 s, 1.7 GB.
## Test 5 (MetaScreener 2.0.0a5 @532ee3c)
- Server starts on CPU; 77 API paths; extraction v2/v3 session API: template (.xlsx) + PDFs -> run -> results/evidence.
