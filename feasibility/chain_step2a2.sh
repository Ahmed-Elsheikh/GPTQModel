#!/bin/bash
# Step 2 task 1: Wanda 50% unstructured, SmolLM2-135M, 128x512 WikiText-2 train calibration, seeds 0-4,
# ppl on the Step 1 protocol (40x2048 test windows). wanda venv (transformers 4.47.1), fp32, 4 threads.
# v2: token streams from the main venv (tokenizers 0.23.2) so windows are byte-identical to Step 1 (see REPORT_step2.md).
source "$(dirname "$0")/step2_common.sh"
export WIKITEXT_STREAM_DIR=$OUT
PY=/opt/venvs/wanda/bin/python; M=HuggingFaceTB/SmolLM2-135M; J=results/real_step2_wanda.jsonl
echo "chain2a2 start $(date -u)"
id=s2_wanda_dense_135m_t023
$PY runlog.py $J $id --json $OUT/$id.json -- $PY wanda_real_driver.py $OUT/$id.json --model $M --sparsity_ratio 0 \
   --sparsity_type unstructured --prune_method wanda --nsamples 128 --seed 0 --save $OUT/$id; after_run "$id"
for s in 0 1 2 3 4; do id=s2_wanda50_135m_s${s}_t023
  $PY runlog.py $J $id --json $OUT/$id.json -- $PY wanda_real_driver.py $OUT/$id.json --model $M --sparsity_ratio 0.5 \
     --sparsity_type unstructured --prune_method wanda --nsamples 128 --seed $s --save $OUT/$id; after_run "$id"
done
commit_push "wanda chain done"
echo "chain2a2 end $(date -u)"
