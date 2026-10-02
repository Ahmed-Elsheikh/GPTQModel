#!/usr/bin/env bash
# Test 2: lm-evaluation-harness on CPU. Usage: run_t2.sh <step...>; appends timing rows to results/t2/timings.tsv
set -u
cd "$(dirname "$0")/.."
source /home/user/venv/bin/activate
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_DATASETS_OFFLINE=0
M=feasibility/models; R=feasibility/results/t2; mkdir -p $R
INC="--include_path feasibility/lm_eval_tasks"
COMMON="--device cpu --seed 0 --log_samples"
run() { # name, args...
  local name=$1; shift
  local t0=$(date -Is)
  /usr/bin/time -v -o $R/$name.time lm_eval "$@" --output_path $R/$name > feasibility/logs/t2_$name.log 2>&1
  local rc=$?
  local wall=$(grep "Elapsed (wall" $R/$name.time | awk '{print $NF}'); local rss=$(grep "Maximum resident" $R/$name.time | awk '{print $NF}')
  printf "%s\t%s\t%s\t%s\t%s\n" "$name" "$t0" "$rc" "$wall" "$rss" | tee -a $R/timings.tsv
}
for step in "$@"; do case $step in
  real)  run a_real_hub --model hf --model_args pretrained=HuggingFaceTB/SmolLM2-135M --tasks arc_easy,hellaswag --num_fewshot 0 --limit 200 --batch_size 8 $COMMON ;;
  a)     run a_dense135 --model hf --model_args pretrained=$M/synth-SmolLM2-135M,dtype=float32 --tasks synth_arc_easy,synth_hellaswag --num_fewshot 0 --limit 200 --batch_size 8 $COMMON $INC ;;
  b)     run b_gptq135 --model hf --model_args pretrained=$M/q-gptq-135m-s0,gptqmodel=True --tasks synth_arc_easy,synth_hellaswag --num_fewshot 0 --limit 200 --batch_size 8 $COMMON $INC ;;
  c0)    run c_base --model hf --model_args pretrained=$M/synth-SmolLM2-135M,dtype=float32,max_length=2048 --tasks synth_arc_easy,synth_hellaswag --num_fewshot 0 --limit 100 --batch_size 8 $COMMON $INC ;;
  c_fs)  run c_fewshot5 --model hf --model_args pretrained=$M/synth-SmolLM2-135M,dtype=float32,max_length=2048 --tasks synth_arc_easy,synth_hellaswag --num_fewshot 5 --limit 100 --batch_size 8 $COMMON $INC ;;
  c_bs)  run c_bs1 --model hf --model_args pretrained=$M/synth-SmolLM2-135M,dtype=float32,max_length=2048 --tasks synth_arc_easy,synth_hellaswag --num_fewshot 0 --limit 100 --batch_size 1 $COMMON $INC ;;
  c_dt)  run c_bf16 --model hf --model_args pretrained=$M/synth-SmolLM2-135M,dtype=bfloat16,max_length=2048 --tasks synth_arc_easy,synth_hellaswag --num_fewshot 0 --limit 100 --batch_size 8 $COMMON $INC ;;
  c_ml)  run c_maxlen512 --model hf --model_args pretrained=$M/synth-SmolLM2-135M,dtype=float32,max_length=512 --tasks synth_arc_easy,synth_hellaswag --num_fewshot 5 --limit 100 --batch_size 8 $COMMON $INC ;;
  d_real) run d_wikitext_hub --model hf --model_args pretrained=$M/synth-SmolLM2-135M,dtype=float32 --tasks wikitext --batch_size 8 $COMMON ;;
  d)     run d_wikitext --model hf --model_args pretrained=$M/synth-SmolLM2-135M,dtype=float32 --tasks synth_wikitext --batch_size 8 $COMMON $INC ;;
esac; done
