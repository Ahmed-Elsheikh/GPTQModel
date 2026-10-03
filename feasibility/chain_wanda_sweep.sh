#!/bin/bash
# A.7 item 1: Wanda sparsity sweep on SmolLM2-135M. 2:4 (50 %), 60 % and 70 % unstructured; seeds 0-4 plus a
# repeat of seed 0 (determinism check). 128 x 512 WikiText-2 train windows from the token-id cache, the Step 1 eval
# (40 x 2048 test windows). wanda venv, fp32, 4 threads. 50 % unstructured seeds 0-4 come from Step 2
# (results/real_step2_wanda.jsonl, *_t023 records, same windows).
cd "$(dirname "$0")"
PY=/opt/venvs/main/bin/python; WPY=/opt/venvs/wanda/bin/python; M=HuggingFaceTB/SmolLM2-135M
J=results/real_wanda_sweep.jsonl; R=$PWD/results/real
run() {  # id sparsity_type ratio seed
  [ -f $R/$1.json ] && grep -q '"ppl"' $R/$1.json && { echo "skip $1"; return; }
  $PY runlog.py $J $1 --json $R/$1.json -- env WANDA_SEQLEN=512 $WPY wanda_diag_driver.py $R/$1.json --model $M \
     --sparsity_ratio $3 --sparsity_type $2 --prune_method wanda --nsamples 128 --seed $4 --save $PWD/ckpt_real/$1
}
echo "sweep start $(date -u)"
for set in "24 2:4 0.5" "u60 unstructured 0.6" "u70 unstructured 0.7"; do
  read tag st ratio <<< "$set"
  for s in 0 1 2 3 4; do run sweep_wanda_${tag}_135m_s$s $st $ratio $s; done
  run sweep_wanda_${tag}_135m_s0_rep $st $ratio 0
done
echo "sweep end $(date -u)"
