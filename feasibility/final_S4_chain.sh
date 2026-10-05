#!/bin/bash
# S4 (trimmed), one container, one job at a time, resumable:
#  A) WT-ppl at context {512,1024,2048} x prepend-BOS {no,yes} for SmolLM2-135M dense fp32 and eager GPTQ W4 g128 s0 (128x512)
#  B) optimum-benchmark 0.6.0 (bench venv): launcher {inline, process} x threads {2,4}, bs 1, seq 128, 32 new tokens,
#     5 iterations, fp32, for SmolLM2-135M and Qwen2.5-0.5B.
cd "$(dirname "$0")"; FEAS=$(pwd); ROOT=$(cd .. && pwd)
export FINAL_CACHE=${FINAL_CACHE:?}; export FINAL_WORK=${FINAL_WORK:?}
PY=/opt/venvs/main/bin/python; RP=results/final_S4_ppl.jsonl; RO=results/final_S4_ob.jsonl
LOGTMP=$FINAL_WORK/logs; mkdir -p $LOGTMP; NDONE=0; BR=$(git rev-parse --abbrev-ref HEAD)
commit_push() {
  git -C $ROOT add feasibility/results/final_S4_*.jsonl feasibility/logs/final_S4_*.log feasibility/final_S4_* 2>/dev/null
  git -C $ROOT commit -q -m "feasibility S4: $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BAoEnHYFNRwMq1jo56pXh4" || return 0
  for d in 2 4 8 16 0; do { git -C $ROOT pull -q --rebase --autostash origin $BR && git -C $ROOT push -q origin $BR; } && return 0; [ $d = 0 ] || sleep $d; done
}
after() { cp $LOGTMP/final_S4_$1.log logs/final_S4_$1.log; NDONE=$((NDONE+1)); [ $((NDONE % 2)) = 0 ] && commit_push "$1"; true; }
fail_line() {  # results file, run_id, rc, log
  $PY - "$@" <<'PYEOF'
import json, sys, datetime
r, i, rc, log = sys.argv[1:]
open(r, "a").write(json.dumps({"run_id": i, "rc": int(rc), "log": log, "utc": datetime.datetime.utcnow().isoformat(),
                               "error_tail": open(log, errors="replace").read()[-2000:]}) + "\n")
PYEOF
}
echo "S4 chain start $(date -u)"
for m in dense w4; do
  n=$(grep -c "\"run_id\": \"${m}_ctx.*\"rc\": 0" $RP 2>/dev/null); n=${n:-0}
  if [ "$n" -ge 6 ]; then echo "skip ppl $m"; continue; fi
  echo "start ppl $m $(date -u +%H:%M:%S)"
  (cd $FINAL_WORK && $PY $FEAS/final_S4_ppl.py $FEAS/$RP $m) > $LOGTMP/final_S4_ppl_$m.log 2>&1; rc=$?
  [ $rc != 0 ] && fail_line $RP ppl_${m}_job $rc $LOGTMP/final_S4_ppl_$m.log
  echo "end ppl $m rc=$rc $(date -u +%H:%M:%S)"; after ppl_$m
done
OB=/opt/venvs/bench/bin/optimum-benchmark; CFG="--config-dir $FEAS/ob_configs --config-name smollm_cpu_real"
for model in HuggingFaceTB/SmolLM2-135M Qwen/Qwen2.5-0.5B; do mk=$(basename $model | tr 'A-Z.' 'a-z_')
for L in inline process; do for t in 2 4; do
  id=ob_${mk}_${L}_t$t; D=$FINAL_WORK/ob/$id
  if [ -f $RO ] && grep -q "\"run_id\": \"$id\", \"rc\": 0" $RO; then echo "skip $id"; continue; fi
  echo "start $id $(date -u +%H:%M:%S)"; rm -rf $D; mkdir -p $D; T0=$(date +%s.%N)
  (cd $D && env OMP_NUM_THREADS=$t OB_OUT=$D $OB $CFG launcher=$L backend.model=$model backend.inter_op_num_threads=$t \
     hydra.job.env_set.OMP_NUM_THREADS=$t scenario.warmup_runs=10 scenario.input_shapes.batch_size=1 \
     scenario.input_shapes.sequence_length=128 scenario.iterations=5 scenario.duration=0 hydra.run.dir=$D \
     +scenario.generate_kwargs.max_new_tokens=32 +scenario.generate_kwargs.min_new_tokens=32) > $LOGTMP/final_S4_$id.log 2>&1
  rc=$?; WALL=$(echo "$(date +%s.%N) - $T0" | bc)
  $PY - $RO $id $rc $D $model $L $t $WALL $LOGTMP/final_S4_$id.log <<'PYEOF'
import json, sys, glob, subprocess, datetime
r, i, rc, d, model, launcher, t, wall, log = sys.argv[1:]
f = glob.glob(d + "/**/benchmark_report.json", recursive=True)
rep = json.load(open(f[0])) if f else None
def g(*k):
    x = rep
    for kk in k: x = (x or {}).get(kk)
    return x
cpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout.splitlines()
gl = lambda k: next((l.split(":", 1)[1].strip() for l in cpu if l.startswith(k)), None)
ok = rep is not None
rec = {"run_id": i, "rc": 0 if ok else int(rc), "exit_code": int(rc), "model": model, "launcher": launcher, "threads": int(t),
       "protocol": {"batch_size": 1, "sequence_length": 128, "new_tokens": 32, "iterations": 5, "warmup_runs": 10, "dtype": "float32"},
       "wall_s": round(float(wall), 1), "utc": datetime.datetime.utcnow().isoformat(), "cpu_model": gl("Model name"), "cpu_flags": gl("Flags"),
       "metrics": ok and {"prefill_latency_mean_s": g("prefill", "latency", "mean"), "decode_latency_mean_s": g("decode", "latency", "mean"),
                          "per_token_latency_mean_s": g("per_token", "latency", "mean"), "decode_throughput": g("decode", "throughput", "value"),
                          "prefill_throughput": g("prefill", "throughput", "value"),
                          "prefill_max_ram_mb": g("prefill", "memory", "max_ram"), "decode_max_ram_mb": g("decode", "memory", "max_ram")},
       "note": "inline launcher exits non-zero after writing the report (JSON-serialisation bug, REPORT_step2 §6)" if ok and rc != "0" else None}
if not ok: rec["error_tail"] = open(log, errors="replace").read()[-2000:]
open(r, "a").write(json.dumps(rec) + "\n"); print(i, rec["rc"], rec["metrics"] and rec["metrics"]["per_token_latency_mean_s"])
PYEOF
  echo "end $id rc=$rc $(date -u +%H:%M:%S)"; after $id
done; done; done
commit_push "chain complete"
echo "S4 chain end $(date -u)"
