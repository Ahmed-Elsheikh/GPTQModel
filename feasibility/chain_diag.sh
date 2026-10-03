#!/bin/bash
# Diagnosis of the A.5 window-length effect. One job at a time; results -> results/real_diag.jsonl.
#  2: GPTQ W3 g128 135M, 512 x 512 tokens (same 262k tokens as 128 x 2048), seeds 0-2
#  3: GPTQ W4 g128 135M, 128 x 2048, seeds 0-1
#  4: Wanda 50% unstructured 135M, 128 x 512 and 128 x 2048, seeds 0-1 (+ dense reference in the wanda venv)
#  5: Qwen2.5-0.5B GPTQ W3 g128, 128 x 512 and 128 x 2048, seed 0 (+ dense reference)
cd "$(dirname "$0")"
PY=/opt/venvs/main/bin/python; WPY=/opt/venvs/wanda/bin/python
J=results/real_diag.jsonl; R=results/real
M=HuggingFaceTB/SmolLM2-135M; Q=Qwen/Qwen2.5-0.5B
EV="--source wikitext2 --eval_n 40 --eval_L 2048 --threads 4"
done_ok() { [ -f $R/$1.json ] && grep -q '"ppl' $R/$1.json; }
gptq() {  # id model n L bits seed
  done_ok $1 && { echo "skip $1"; return; }
  $PY runlog.py $J $1 --json $R/$1.json -- $PY quant_eval.py --model $2 $EV --n $3 --L $4 --bits $5 --group_size 128 --seed $6 \
     --dtype float32 --backend torch --save_dir ckpt_real/$1 --out $R/$1.json
}
wanda() {  # id seqlen sparsity seed
  done_ok $1 && { echo "skip $1"; return; }
  # absolute paths: wanda_real_driver.py chdirs into third_party/wanda
  $PY runlog.py $J $1 --json $PWD/$R/$1.json -- env WANDA_SEQLEN=$2 $WPY wanda_diag_driver.py $PWD/$R/$1.json --model $M \
     --sparsity_ratio $3 --sparsity_type unstructured --prune_method wanda --nsamples 128 --seed $4 --save $PWD/ckpt_real/$1
}
echo "diag start $(date -u)"
for s in 0 1 2; do gptq diag_w3_135m_n512_L512_s$s $M 512 512 3 $s; done
for s in 0 1; do gptq diag_w4_135m_L2048_s$s $M 128 2048 4 $s; done
wanda diag_wanda_dense_135m 512 0 0
for s in 0 1; do wanda diag_wanda50_135m_L512_s$s 512 0.5 $s; wanda diag_wanda50_135m_L2048_s$s 2048 0.5 $s; done
done_ok diag_qwen05_dense || $PY runlog.py $J diag_qwen05_dense --json $R/diag_qwen05_dense.json -- $PY quant_eval.py --model $Q $EV --dense_only --out $R/diag_qwen05_dense.json
gptq diag_w3_qwen05_L512_s0 $Q 128 512 3 0
gptq diag_w3_qwen05_L2048_s0 $Q 128 2048 3 0
echo "diag end $(date -u)"
