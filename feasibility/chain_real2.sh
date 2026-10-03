#!/bin/bash
# Step 2 of the real-data rerun. Runs after chain_real1.sh (waits for it), one job at a time so timings are clean.
cd "$(dirname "$0")"
PY=/opt/venvs/main/bin/python
M=HuggingFaceTB/SmolLM2-135M
J=results/real_step2.jsonl
COMMON="--n 128 --L 512 --eval_n 40 --eval_L 2048 --threads 4"
while pgrep -f chain_real1.sh >/dev/null; do sleep 30; done
echo "chain2 start $(date -u)"
done_ok() { [ -f results/real/$1.json ] && grep -q '"ppl_quant"\|"results"' results/real/$1.json; }

# C4 calibration, GPTQ W4 seed 0 (perplexity still on the WikiText-2 test windows)
id=gptq_w4_135m_c4_s0; done_ok $id || $PY runlog.py $J $id --json results/real/$id.json -- $PY quant_eval.py --model $M $COMMON \
  --source c4 --eval_source wikitext2 --bits 4 --group_size 128 --seed 0 --dtype float32 --backend torch --out results/real/$id.json

# lm-eval real arc_easy + hellaswag, 200 items each, 0-shot: dense fp32 vs GPTQ W4 s0 (in-memory, bf16 TorchLinear)
id=lmeval_dense_135m; done_ok $id || $PY runlog.py $J $id --json results/real/$id.json -- $PY lmeval_real.py dense $M results/real/$id.json arc_easy,hellaswag 200 8
id=lmeval_gptq_w4_135m_s0; done_ok $id || $PY runlog.py $J $id --json results/real/$id.json -- $PY lmeval_real.py quant ckpt_real/gptq_w4_135m_s0 results/real/$id.json arc_easy,hellaswag 200 8

# lm-eval wikitext task on dense 135M (max_length 2048, batch size 1)
id=lmeval_wikitext_dense_135m; done_ok $id || $PY runlog.py $J $id --json results/real/$id.json -- $PY lmeval_real.py dense $M results/real/$id.json wikitext none 1 2048

# optimum-benchmark: 2 protocols x {process (default), inline} launcher, OMP 4
OB=/opt/venvs/bench/bin/optimum-benchmark
CFG="--config-dir $PWD/ob_configs --config-name smollm_cpu_real"
for proto in p1 p2; do
  case $proto in
    p1) P="scenario.warmup_runs=10 scenario.input_shapes.batch_size=1 scenario.input_shapes.sequence_length=128" ;;
    p2) P="scenario.warmup_runs=10 scenario.input_shapes.batch_size=4 scenario.input_shapes.sequence_length=512" ;;
  esac
  for L in process inline; do
    id=ob_${proto}_$L
    [ -f results/real/ob/$id/benchmark_report.json ] && continue
    OB_OUT=$PWD/results/real/ob/$id $PY runlog.py $J $id -- env OB_OUT=$PWD/results/real/ob/$id $OB $CFG launcher=$L \
      $P scenario.iterations=5 scenario.duration=0 hydra.run.dir=$PWD/results/real/ob/$id \
      +scenario.generate_kwargs.max_new_tokens=32 +scenario.generate_kwargs.min_new_tokens=32
  done
done

# Qwen2.5-0.5B GPTQ W4 g128 seed 0
id=gptq_w4_qwen05_s0; done_ok $id || $PY runlog.py $J $id --json results/real/$id.json -- $PY quant_eval.py --model Qwen/Qwen2.5-0.5B $COMMON \
  --source wikitext2 --bits 4 --group_size 128 --seed 0 --dtype float32 --backend torch --out results/real/$id.json

# AWQ W4 on 135M, quantize(backend=AWQ_TORCH): g128 as asked (expected to fail on K=576), then g64 bf16 (the config that ran before)
id=awq_w4g128_135m_s0; [ -f results/real/$id.json ] || $PY runlog.py $J $id --json results/real/$id.json -- $PY quant_eval.py --model $M $COMMON \
  --source wikitext2 --method awq --bits 4 --group_size 128 --seed 0 --quant_backend awq_torch --backend awq_torch --out results/real/$id.json
id=awq_w4g64_135m_s0; done_ok $id || $PY runlog.py $J $id --json results/real/$id.json -- $PY quant_eval.py --model $M $COMMON \
  --source wikitext2 --method awq --bits 4 --group_size 64 --seed 0 --quant_backend awq_torch --backend awq_torch --out results/real/$id.json
echo "chain2 end $(date -u)"
