#!/bin/bash
# Step 2 add-on: 2x2 calibration source (WikiText-2 / C4) x eval set (WikiText-2 40x2048 / C4-val 256x2048), GPTQ W4 g128 s0.
source "$(dirname "$0")/step2_common.sh"
PY=/opt/venvs/main/bin/python; M=HuggingFaceTB/SmolLM2-135M; J=results/real_step2_c4val.jsonl; CK=$OUT/ckpt
C=$OUT/c4val_256x2048_s0.npy
echo "chain2c start $(date -u)"
id=s2_c4val_cache; [ -f $C ] || { $PY runlog.py $J $id --json $OUT/$id.json -- $PY step2_c4val.py build $C $OUT/$id.json; after_run $id; }
# rebuild the WikiText-calibrated seed-0 checkpoint exactly as Step 1 (must reproduce ppl 17.744842947166127)
id=s2_gptq_w4_135m_wt_s0_rebuild
[ -f $CK/$id/config.json ] || { $PY runlog.py $J $id --json $OUT/$id.json -- $PY quant_eval.py --model $M --source wikitext2 --n 128 --L 512 \
  --eval_n 40 --eval_L 2048 --threads 4 --bits 4 --group_size 128 --seed 0 --dtype float32 --backend torch \
  --save_dir $CK/$id --out $OUT/$id.json; after_run $id; }
for t in dense wt c4; do
  case $t in dense) T=dense ;; wt) T=$CK/s2_gptq_w4_135m_wt_s0_rebuild ;; c4) T=$CK/s2_gptq_w4_135m_c4_s0 ;; esac
  id=s2_c4val_eval_$t; $PY runlog.py $J $id --json $OUT/$id.json -- $PY step2_c4val.py eval $C $OUT/$id.json $T; after_run $id
done
commit_push "c4val chain done"
echo "chain2c end $(date -u)"
