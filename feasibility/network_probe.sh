#!/usr/bin/env bash
# Probe hosts needed by the design; prints HTTP code (000 = CONNECT refused by egress proxy)
for h in https://pypi.org/simple/ https://files.pythonhosted.org https://download.pytorch.org/whl/cpu/ \
  https://huggingface.co/api/models/HuggingFaceTB/SmolLM2-135M https://cdn-lfs.huggingface.co https://cas-bridge.xethub.hf.co \
  https://github.com https://codeload.github.com/locuslab/wanda/tar.gz/refs/heads/main https://raw.githubusercontent.com \
  https://arxiv.org/abs/2306.11695v3 https://export.arxiv.org https://api.anthropic.com/v1/messages \
  https://api.openai.com https://openrouter.ai https://api.together.xyz https://s3.amazonaws.com https://data.together.xyz; do
  code=$(curl -s -o /dev/null -w "%{http_code}" -m 20 "$h"); echo "$code $h"; done
