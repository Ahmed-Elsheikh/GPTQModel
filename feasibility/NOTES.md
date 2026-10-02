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
