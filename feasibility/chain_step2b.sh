#!/bin/bash
# Step 2 tasks 2-6 in order (parallel worker). Waits for chain_step2a.sh; one job at a time so timings are clean.
source "$(dirname "$0")/step2_common.sh"
PY=/opt/venvs/main/bin/python; M=HuggingFaceTB/SmolLM2-135M
COMMON="--n 128 --L 512 --eval_n 40 --eval_L 2048 --threads 4"
CK=$OUT/ckpt
while pgrep -f 'bash ./chain_step2a' >/dev/null; do sleep 30; done
echo "chain2b start $(date -u)"

# 2. C4 calibration, GPTQ W4 g128 seed 0, fp32 quantize, ppl on the WikiText-2 test windows (bf16 TorchLinear)
J=results/real_step2_c4.jsonl; id=s2_gptq_w4_135m_c4_s0
$PY runlog.py $J $id --json $OUT/$id.json -- $PY quant_eval.py --model $M $COMMON --source c4 --eval_source wikitext2 \
  --bits 4 --group_size 128 --seed 0 --dtype float32 --backend torch --save_dir $CK/$id --out $OUT/$id.json; after_run $id

# 3. AWQ W4 on 135M with quantize(backend=AWQ_TORCH): g128 as specified, then g64 (only config that ran in the study)
J=results/real_step2_awq.jsonl
id=s2_awq_w4g128_135m_s0; $PY runlog.py $J $id --json $OUT/$id.json -- $PY quant_eval.py --model $M $COMMON --source wikitext2 \
  --method awq --bits 4 --group_size 128 --seed 0 --quant_backend awq_torch --backend awq_torch --out $OUT/$id.json; after_run $id
id=s2_awq_w4g64_135m_s0; $PY runlog.py $J $id --json $OUT/$id.json -- $PY quant_eval.py --model $M $COMMON --source wikitext2 \
  --method awq --bits 4 --group_size 64 --seed 0 --quant_backend awq_torch --backend awq_torch --out $OUT/$id.json; after_run $id
# 4. Qwen2.5-0.5B GPTQ W4 g128 seed 0 (WikiText-2 calibration), fp32 quantize, ppl 40x2048 (bf16 TorchLinear)
J=results/real_step2_qwen.jsonl; id=s2_gptq_w4_qwen05_s0
$PY runlog.py $J $id --json $OUT/$id.json -- $PY quant_eval.py --model Qwen/Qwen2.5-0.5B $COMMON --source wikitext2 \
  --bits 4 --group_size 128 --seed 0 --dtype float32 --backend torch --eval_dense --out $OUT/$id.json; after_run $id

# 5. lm-eval real arc_easy + hellaswag, 200 items each, 0-shot, bs 8: dense fp32 vs GPTQ W4 (C4 seed-0 ckpt, bf16 TorchLinear)
J=results/real_step2_lmeval.jsonl
id=s2_lmeval_dense_135m; $PY runlog.py $J $id --json $OUT/$id.json -- $PY lmeval_real.py dense $M $OUT/$id.json arc_easy,hellaswag 200 8; after_run $id
id=s2_lmeval_gptq_w4_135m_c4_s0; $PY runlog.py $J $id --json $OUT/$id.json -- $PY lmeval_real.py quant $CK/s2_gptq_w4_135m_c4_s0 $OUT/$id.json arc_easy,hellaswag 200 8; after_run $id

# 6. optimum-benchmark: 2 earlier protocols (Test 3b grid points) x {process (default), inline} launcher, OMP 4
J=results/real_step2_ob.jsonl; OB=/opt/venvs/bench/bin/optimum-benchmark
CFG="--config-dir $PWD/ob_configs --config-name smollm_cpu_real"
for proto in p1 p2; do
  case $proto in
    p1) P="scenario.warmup_runs=10 scenario.input_shapes.batch_size=1 scenario.input_shapes.sequence_length=128" ;;
    p2) P="scenario.warmup_runs=10 scenario.input_shapes.batch_size=4 scenario.input_shapes.sequence_length=512" ;;
  esac
  for L in process inline; do id=s2_ob_${proto}_$L; D=$OUT/ob/$id
    OB_OUT=$D $PY runlog.py $J $id -- env OB_OUT=$D $OB $CFG launcher=$L $P scenario.iterations=5 scenario.duration=0 \
      hydra.run.dir=$D +scenario.generate_kwargs.max_new_tokens=32 +scenario.generate_kwargs.min_new_tokens=32
    $PY - "$J" "$id" "$D" <<'PYEOF'
import json, sys, glob
j, rid, d = sys.argv[1:]
f = glob.glob(d + "/**/benchmark_report.json", recursive=True)
rep = json.load(open(f[0])) if f else None
def g(*k):
    x = rep
    for kk in k: x = (x or {}).get(kk)
    return x
s = {"run_id": rid + "_report", "report_found": bool(f), "report": rep and {
  "prefill_latency_mean_s": g("prefill", "latency", "mean"), "decode_latency_mean_s": g("decode", "latency", "mean"),
  "per_token_latency_mean_s": g("per_token", "latency", "mean"), "decode_throughput": g("decode", "throughput", "value"),
  "prefill_throughput": g("prefill", "throughput", "value"), "load_s": g("load_model", "latency", "mean") or g("load", "latency", "mean")}}
open(j, "a").write(json.dumps(s) + "\n"); print(s)
PYEOF
    after_run $id
  done
done

commit_push "chain b done"
echo "chain2b end $(date -u)"
