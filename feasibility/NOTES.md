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
