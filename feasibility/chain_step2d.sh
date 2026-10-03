#!/bin/bash
# Determinism probe: exact Step 1 GPTQ W4 g128 seed-0 command (WikiText-2), repeated; then settings if repeats differ.
source "$(dirname "$0")/step2_common.sh"
PY=/opt/venvs/main/bin/python; M=HuggingFaceTB/SmolLM2-135M; J=results/real_step2_det.jsonl; CK=$OUT/ckpt
run() {  # run id setting threads
  local id=$1; DET_SETTING=$2 $PY runlog.py $J $id --json $OUT/$id.json -- env DET_SETTING=$2 $PY step2_det.py --model $M \
    --source wikitext2 --n 128 --L 512 --eval_n 40 --eval_L 2048 --threads $3 --bits 4 --group_size 128 --seed 0 \
    --dtype float32 --backend torch --save_dir $CK/$id --out $OUT/$id.json
  after_run $id
}
echo "chain2d start $(date -u)"
for r in "$@"; do set -- $(echo $r | tr ':' ' '); run s2_det_$1 $2 $3; done
commit_push "determinism chain done"
echo "chain2d end $(date -u)"
