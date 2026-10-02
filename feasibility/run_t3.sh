#!/usr/bin/env bash
# Test 3: optimum-benchmark on CPU. a = single run; b = Hydra multirun sweep.
set -u
cd "$(dirname "$0")/.."
source ${VENV:-/home/user/venv}/bin/activate
export OB_MODEL_ROOT=$PWD/feasibility/models OMP_NUM_THREADS=4
CFG="--config-dir $PWD/feasibility/ob_configs --config-name smollm_cpu"
case $1 in
  a) export OB_OUT=$PWD/feasibility/results/t3a
     /usr/bin/time -v -o feasibility/results/t3a.time optimum-benchmark $CFG > feasibility/logs/t3a.log 2>&1; echo rc=$? ;;
  b) export OB_OUT=$PWD/feasibility/results/t3b
     /usr/bin/time -v -o feasibility/results/t3b.time optimum-benchmark $CFG -m \
       "backend.model=$OB_MODEL_ROOT/synth-SmolLM2-135M,$OB_MODEL_ROOT/synth-SmolLM2-360M" \
       "scenario.warmup_runs=0,10" "scenario.input_shapes.batch_size=1,4" "scenario.input_shapes.sequence_length=128,512" \
       hydra.job.env_set.OMP_NUM_THREADS=3 scenario.iterations=5 scenario.duration=0 \
       +scenario.generate_kwargs.max_new_tokens=32 +scenario.generate_kwargs.min_new_tokens=32 \
       > feasibility/logs/t3b.log 2>&1; echo rc=$? ;;
esac
