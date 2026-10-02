#!/usr/bin/env bash
# Sequential chain for the rest of Test 1. Starts after chain A finishes.
cd "$(dirname "$0")/.."
until grep -q "t4 (tf5)" feasibility/logs/chain_A.log 2>/dev/null; do sleep 20; done
source /home/user/venv/bin/activate
M=feasibility/models; R=feasibility/results/t1
q() { local name=$1; shift; echo "$(date -Is) start $name" >> feasibility/logs/chain_B.log
  /usr/bin/time -v -o $R/$name.time python feasibility/quant_eval.py --backend torch --out $R/$name.json "$@" > feasibility/logs/t1_$name.log 2>&1
  echo "$(date -Is) end $name rc=$?" >> feasibility/logs/chain_B.log; }
q gptq_135m_s0_b --model $M/synth-SmolLM2-135M --seed 0                         # 1e determinism repeat (auto dtype = bf16)
q gptq_135m_s1   --model $M/synth-SmolLM2-135M --seed 1                         # 1e seed effect
q gptq_135m_s0_fp32 --model $M/synth-SmolLM2-135M --seed 0 --dtype float32      # dtype speed check
q gptq_360m_s0   --model $M/synth-SmolLM2-360M --seed 0 --eval_dense            # 1f
q awq_135m_s0    --model $M/synth-SmolLM2-135M --seed 0 --method awq            # 1g
q gptq_qwen05_s0 --model $M/synth-Qwen2.5-0.5B --seed 0                         # 1f optional
echo "$(date -Is) chain_B done" >> feasibility/logs/chain_B.log
