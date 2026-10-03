#!/bin/bash
# Step 1 of the real-data rerun: dense ppl, GPTQ W4 g128 seeds 0-4, GPTQ W3 g128 seeds 0-4 (SmolLM2-135M, WikiText-2).
cd "$(dirname "$0")"
PY=/opt/venvs/main/bin/python
M=HuggingFaceTB/SmolLM2-135M
J=results/real_step1.jsonl
COMMON="--model $M --source wikitext2 --n 128 --L 512 --eval_n 40 --eval_L 2048 --threads 4"
echo "chain start $(date -u)"
$PY runlog.py $J dense_135m --json results/real/dense_135m.json -- $PY quant_eval.py $COMMON --dense_only --out results/real/dense_135m.json
for b in 4 3; do for s in 0 1 2 3 4; do
  id=gptq_w${b}_135m_s$s
  [ -f results/real/$id.json ] && grep -q ppl_quant results/real/$id.json && { echo "skip $id"; continue; }
  $PY runlog.py $J $id --json results/real/$id.json -- $PY quant_eval.py $COMMON --bits $b --group_size 128 --seed $s \
     --dtype float32 --backend torch --save_dir ckpt_real/$id --out results/real/$id.json
done; done
echo "chain end $(date -u)"
