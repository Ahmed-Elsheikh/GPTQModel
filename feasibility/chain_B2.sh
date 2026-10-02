#!/usr/bin/env bash
# Rest of Test 1 (+ Test 2b in-memory) run strictly sequentially.
cd "$(dirname "$0")/.."
source /home/user/venv/bin/activate; export OMP_NUM_THREADS=4
M=feasibility/models; R=feasibility/results/t1; L=feasibility/logs/chain_B2.log
q() { local name=$1; shift; echo "$(date -Is) start $name" >> $L
  /usr/bin/time -v -o $R/$name.time python feasibility/quant_eval.py --backend torch --out $R/$name.json "$@" > feasibility/logs/t1_$name.log 2>&1
  echo "$(date -Is) end $name rc=$?" >> $L; }
q gptq_135m_s0_b --model $M/synth-SmolLM2-135M --seed 0                         # 1e determinism repeat (auto dtype = bf16)
q gptq_135m_s1   --model $M/synth-SmolLM2-135M --seed 1                         # 1e seed effect
echo "$(date -Is) start t2b_inmemory" >> $L
/usr/bin/time -v -o feasibility/results/t2/b_inmemory.time python feasibility/lmeval_inmemory.py $M/q-gptq-135m-s0 feasibility/results/t2/b_inmemory.json > feasibility/logs/t2_b_inmemory.log 2>&1
echo "$(date -Is) end t2b_inmemory rc=$?" >> $L
q gptq_135m_s0_fp32 --model $M/synth-SmolLM2-135M --seed 0 --dtype float32      # dtype speed check
DT=$(python - <<'PY'
import json
try:
    a=json.load(open("feasibility/results/t1/gptq_135m_s0_a.json"))["phases"]["quantize"]["wall_s"]
    b=json.load(open("feasibility/results/t1/gptq_135m_s0_fp32.json"))["phases"]["quantize"]["wall_s"]
    print("float32" if b < a else "auto")
except Exception: print("auto")
PY
)
echo "$(date -Is) chose dtype=$DT for 360M/AWQ/Qwen" >> $L
q gptq_360m_s0   --model $M/synth-SmolLM2-360M --seed 0 --dtype $DT             # 1f
q awq_135m_s0    --model $M/synth-SmolLM2-135M --seed 0 --method awq --dtype $DT  # 1g
q gptq_qwen05_s0 --model $M/synth-Qwen2.5-0.5B --seed 0 --dtype $DT             # 1f optional
echo "$(date -Is) chain_B2 done" >> $L
