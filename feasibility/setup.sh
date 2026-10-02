#!/usr/bin/env bash
# Recommended SessionStart setup for the SLM-compression reproducibility study (CPU-only cloud VM).
# Installs exactly what worked in the feasibility study (2026-10-02). Idempotent; exits 0 even if an optional
# step fails (failures are printed). Wall time: ~4-5 min on a cold pip cache when torch comes from PyPI
# (the PyPI wheel drags ~4 GB of unused CUDA libs); ~1-2 min if download.pytorch.org is allowlisted.
set -u
ROOT=${ROOT:-/home/user}
REPO=$(cd "$(dirname "$0")/.." && pwd)
log() { echo "[setup $(date +%H:%M:%S)] $*"; }

command -v /usr/bin/time >/dev/null || { apt-get install -y -qq time >/dev/null 2>&1 && log "installed GNU time"; }

# 1) main venv: torch + GPTQModel + lm-eval (transformers 5.x)
if [ ! -x "$ROOT/venv/bin/python" ]; then
  python3 -m venv "$ROOT/venv" && "$ROOT/venv/bin/pip" install -q --upgrade pip
fi
if ! "$ROOT/venv/bin/python" -c "import torch, gptqmodel, lm_eval" 2>/dev/null; then
  log "installing torch (CPU index first, PyPI fallback)"
  "$ROOT/venv/bin/pip" install -q torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu 2>/dev/null \
    || "$ROOT/venv/bin/pip" install -q torch==2.14.1
  log "installing gptqmodel/lm-eval"
  "$ROOT/venv/bin/pip" install -q gptqmodel==7.5.0 lm-eval==0.4.13 psutil || log "WARN: main venv install failed"
fi

# 2) venv-ob: optimum-benchmark 0.6.0 + wanda need transformers<5 (0.6.0 imports SpecialTokensMixin, removed in 5.x)
if [ ! -x "$ROOT/venv-ob/bin/python" ]; then
  python3 -m venv "$ROOT/venv-ob" && "$ROOT/venv-ob/bin/pip" install -q --upgrade pip
fi
if ! "$ROOT/venv-ob/bin/python" -c "import optimum_benchmark, transformers" 2>/dev/null; then
  log "installing venv-ob"
  "$ROOT/venv-ob/bin/pip" install -q torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu 2>/dev/null \
    || "$ROOT/venv-ob/bin/pip" install -q torch==2.14.1
  "$ROOT/venv-ob/bin/pip" install -q "transformers==4.57.6" optimum-benchmark==0.6.0 datasets psutil || log "WARN: venv-ob install failed"
fi

# 3) third-party repos not on PyPI (git works through the proxy; codeload.github.com tarballs are blocked)
mkdir -p "$REPO/third_party"
clone() { [ -d "$REPO/third_party/$2/.git" ] || git clone -q "https://github.com/$1" "$REPO/third_party/$2"; git -C "$REPO/third_party/$2" checkout -q "$3"; }
clone locuslab/wanda wanda 8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6 || log "WARN: wanda clone failed"
( cd "$REPO/third_party/wanda" && git apply --check "$REPO/feasibility/patches/wanda_cpu.patch" 2>/dev/null && git apply "$REPO/feasibility/patches/wanda_cpu.patch" ) || true

# 4) MetaScreener (optional; separate venv, 1.1 GB clone because of experiments/)
if [ "${WITH_METASCREENER:-0}" = 1 ]; then
  clone ChaokunHong/MetaScreener MetaScreener 532ee3cc10a7e22e54747d7c3413509c31b9a99b || log "WARN: MetaScreener clone failed"
  [ -x "$ROOT/venv-ms/bin/python" ] || python3 -m venv "$ROOT/venv-ms"
  "$ROOT/venv-ms/bin/pip" install -q "$REPO/third_party/MetaScreener" || log "WARN: metascreener install failed"
fi

# 5) offline stand-ins (only needed while huggingface.co is blocked)
[ -f "$REPO/feasibility/models/synth-SmolLM2-135M/config.json" ] || "$ROOT/venv/bin/python" "$REPO/feasibility/make_synth_models.py" >/dev/null 2>&1 || log "WARN: synthetic models failed"
[ -f "$REPO/feasibility/lm_eval_tasks/arc_test.jsonl" ] || python3 "$REPO/feasibility/lm_eval_tasks/make_synth_tasks.py"

"$ROOT/venv/bin/python" -c "import torch,transformers,gptqmodel,lm_eval;print('venv OK', torch.__version__, transformers.__version__, gptqmodel.__version__)" 2>/dev/null | tail -1 || log "WARN: venv import check failed"
"$ROOT/venv-ob/bin/python" -c "import transformers,optimum_benchmark;print('venv-ob OK', transformers.__version__)" 2>/dev/null || log "WARN: venv-ob import check failed"
log "done"
exit 0
