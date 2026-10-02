#!/usr/bin/env bash
cd "$(dirname "$0")/.."
until grep -q "chain_B2 done" feasibility/logs/chain_B2.log 2>/dev/null; do sleep 30; done
feasibility/run_t2.sh d2 >> feasibility/logs/chain_D.log 2>&1
echo "$(date -Is) chain_D done" >> feasibility/logs/chain_D.log
# post-hoc evals of checkpoints whose in-run eval crashed (fp32 reload not supported by TorchLinear)
source /home/user/venv/bin/activate
for n in gptq_135m_s0_fp32:synth-SmolLM2-135M gptq_360m_s0:synth-SmolLM2-360M; do
  name=${n%%:*}; model=${n##*:}; d=feasibility/results/t1/_tmp_q_$name
  [ -d $d ] || continue
  /usr/bin/time -v -o feasibility/results/t1/${name}_eval.time python feasibility/quant_eval.py --model feasibility/models/$model --eval_only $d --backend torch --dtype float32 --out feasibility/results/t1/${name}_eval.json > feasibility/logs/t1_${name}_eval.log 2>&1
  echo "$(date -Is) eval $name rc=$?" >> feasibility/logs/chain_D.log
done
echo "$(date -Is) chain_D evals done" >> feasibility/logs/chain_D.log
# AWQ on CPU: quantize() with BACKEND.AUTO picks AWQ_GEMM (CUDA-only) -> pass awq_torch explicitly
/usr/bin/time -v -o feasibility/results/t1/awq_135m_s0_awqtorch.time python feasibility/quant_eval.py --model feasibility/models/synth-SmolLM2-135M --seed 0 --method awq --dtype float32 --quant_backend awq_torch --backend awq_torch --out feasibility/results/t1/awq_135m_s0_awqtorch.json > feasibility/logs/t1_awq_135m_s0_awqtorch.log 2>&1
echo "$(date -Is) awq_torch rc=$?" >> feasibility/logs/chain_D.log
echo "$(date -Is) chain_D all done" >> feasibility/logs/chain_D.log
