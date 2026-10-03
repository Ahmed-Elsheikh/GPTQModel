#!/bin/bash
# Session S2 chain (resumable: a run with an rc==0 line in its results file is skipped).
#  A. Determinism: anchor (SmolLM2-135M GPTQ W4 g128 s0, 128x512 WikiText, WT-ppl) x5 with eager attention;
#     if not bit-identical (ppl repr + checkpoint sha256): x3 with eager + 1 thread; if still not: STOP.
#  B. S2 on Qwen/Qwen2.5-0.5B with the confirmed configuration: dense fp32; GPTQ W3 g128 128x2048 seeds 0-4;
#     GPTQ W3 g128 128x512 seeds 0-2.
#  C. Determinism caveats in REPORT.md A.3/A.5/A.6.
#  D. Wanda (SmolLM2-135M) 2:4, 60 %, 70 % unstructured, seeds 0-2 + seed-0 repeat each, 128x512 from the id cache.
# Commit + push after every 2 completed runs (logs of running jobs are never committed: commit_done.sh).
cd "$(dirname "$0")"
PY=/opt/venvs/main/bin/python; WPY=/opt/venvs/wanda/bin/python
SMOL=HuggingFaceTB/SmolLM2-135M; QWEN=Qwen/Qwen2.5-0.5B
WORK=${WORK:-/tmp/claude-0/-home-user-GPTQModel/945963cb-7168-5102-8d7d-2c6a90a34ff0/scratchpad/final_S2}; mkdir -p $WORK
JA=results/final_S2_anchor.jsonl; JR=results/final_S2_runs.jsonl; JW=results/final_S2_wanda.jsonl
NDONE=0
done_ok() { [ -f $1 ] && python3 -c "import json,sys;sys.exit(0 if any(json.loads(l)['run_id']=='$2' and json.loads(l)['rc']==0 for l in open('$1')) else 1)"; }
sync_commit() { $PY final_S2_report.py >/dev/null 2>&1; ./commit_done.sh "final S2: $1" || echo "commit failed: $1"; }
after_run() { NDONE=$((NDONE+1)); [ $((NDONE % 2)) = 0 ] && sync_commit "$1"; true; }
gptq() {  # jsonl id cfg threads model n L bits seed [--dense_only]
  local J=$1 id=$2 cfg=$3 thr=$4 model=$5 n=$6 L=$7 bits=$8 seed=$9; shift 9
  done_ok $J $id && { echo "skip $id"; return 0; }
  local method="gptq"; [ "$1" = "--dense_only" ] && method="dense"
  local meta="{\"model\":\"$model\",\"method\":\"$method\",\"bits\":$( [ $method = dense ] && echo null || echo $bits),\"group_size\":$( [ $method = dense ] && echo null || echo 128),\"calib_source\":$( [ $method = dense ] && echo null || echo '"wikitext2-train"'),\"calib_count\":$( [ $method = dense ] && echo null || echo $n),\"calib_length\":$( [ $method = dense ] && echo null || echo $L),\"seed\":$( [ $method = dense ] && echo null || echo $seed),\"config\":{\"final_cfg\":\"$cfg\",\"threads\":$thr,\"backend\":\"torch\",\"quant_dtype\":\"float32\",\"eval_dtype\":$( [ $method = dense ] && echo '"float32 (+bf16)"' || echo '"bfloat16"')},\"eval\":\"WT-ppl 40x2048 test (7da3b34bdebd1923)\"}"
  FINAL_CFG=$cfg $PY runlog.py $J $id --json $WORK/$id.json --meta "$meta" -- env FINAL_CFG=$cfg $PY final_S2_run.py \
     --model $model --source wikitext2 --n $n --L $L --bits $bits --group_size 128 --seed $seed --eval_n 40 --eval_L 2048 \
     --threads $thr --dtype float32 --backend torch $( [ $method = dense ] && echo --dense_only || echo --save_dir $WORK/ckpt_$id ) --out $WORK/$id.json
  after_run $id
}
echo "chain S2 start $(date -u)"
# ---- A. determinism
for r in 1 2 3 4 5; do gptq $JA final_S2_anchor_eager_r$r eager 4 $SMOL 128 512 4 0; done
if $PY final_S2_check.py $JA 'final_S2_anchor_eager_r\d' 5; then CFG=eager; THR=4
else
  for r in 1 2 3; do gptq $JA final_S2_anchor_eager_thr1_r$r eager_threads1 1 $SMOL 128 512 4 0; done
  if $PY final_S2_check.py $JA 'final_S2_anchor_eager_thr1_r\d' 3; then CFG=eager_threads1; THR=1
  else sync_commit "determinism NOT confirmed - STOP"; echo "DETERMINISM_STOP"; exit 1; fi
fi
VAL=$(python3 -c "import json,re;print([repr(json.loads(l)['result']['ppl_quant']) for l in open('$JA') if json.loads(l)['rc']==0 and json.loads(l)['result'].get('config',{}).get('final_cfg')=='$CFG'][0])")
sync_commit "Determinism (confirmed): $CFG, anchor $VAL"
echo "DETERMINISM_CONFIRMED $CFG $VAL"
# ---- B. S2 on Qwen
gptq $JR final_S2_qwen_dense $CFG $THR $QWEN 128 512 3 0 --dense_only
for s in 0 1 2 3 4; do gptq $JR final_S2_qwen_w3_L2048_s$s $CFG $THR $QWEN 128 2048 3 $s; done
for s in 0 1 2; do gptq $JR final_S2_qwen_w3_L512_s$s $CFG $THR $QWEN 128 512 3 $s; done
sync_commit "S2 runs done"
echo "S2_DONE"
# ---- C. caveats in REPORT.md
python3 final_S2_caveat.py && ./commit_done.sh "REPORT.md: determinism caveats in A.3, A.5, A.6"
# ---- D. Wanda sweep
wanda() {  # id type ratio seed
  done_ok $JW $1 && { echo "skip $1"; return 0; }
  local meta="{\"model\":\"$SMOL\",\"method\":\"wanda\",\"sparsity_type\":\"$2\",\"sparsity\":$3,\"group_size\":null,\"calib_source\":\"wikitext2-train\",\"calib_count\":128,\"calib_length\":512,\"seed\":$4,\"config\":{\"venv\":\"wanda (transformers 4.47.1)\",\"dtype\":\"float32\",\"threads\":4,\"attention\":\"default\"},\"eval\":\"WT-ppl 40x2048 test (7da3b34bdebd1923)\"}"
  $PY runlog.py $JW $1 --json $WORK/$1.json --meta "$meta" -- env WANDA_SEQLEN=512 $WPY wanda_diag_driver.py $WORK/$1.json \
     --model $SMOL --sparsity_ratio $3 --sparsity_type $2 --prune_method wanda --nsamples 128 --seed $4 --save $WORK/wsave_$1
  after_run $1
}
for set in "24 2:4 0.5" "u60 unstructured 0.6" "u70 unstructured 0.7"; do
  read tag st ratio <<< "$set"
  for s in 0 1 2; do wanda final_S2_wanda_${tag}_s$s $st $ratio $s; done
  wanda final_S2_wanda_${tag}_s0_rep $st $ratio 0
done
sync_commit "Wanda sweep done"
echo "CHAIN_END $(date -u)"
