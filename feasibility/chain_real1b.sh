#!/bin/bash
# Extra check A: Step 1's GPTQ W3 g128 runs with calibration windows of 2048 tokens (was 512); everything else unchanged.
cd "$(dirname "$0")"
PY=/opt/venvs/main/bin/python
M=HuggingFaceTB/SmolLM2-135M
J=results/real_step1b.jsonl
COMMON="--model $M --source wikitext2 --n 128 --L 2048 --eval_n 40 --eval_L 2048 --threads 4"
echo "chain1b start $(date -u)"
for s in 0 1 2 3 4; do
  id=gptq_w3_135m_L2048_s$s
  [ -f results/real/$id.json ] && grep -q ppl_quant results/real/$id.json && { echo "skip $id"; continue; }
  $PY runlog.py $J $id --json results/real/$id.json -- $PY quant_eval.py $COMMON --bits 3 --group_size 128 --seed $s \
     --dtype float32 --backend torch --save_dir ckpt_real/$id --out results/real/$id.json
done
echo "chain1b end $(date -u)"
