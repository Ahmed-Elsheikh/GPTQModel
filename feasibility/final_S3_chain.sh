#!/bin/bash
# S3 resumable chain: {GPTQ W4, W3} g128 x {WikiText-2, C4} calibration (128x512) x seeds 0-2, each evaluated on WT-ppl and C4-ppl.
# Configuration: eager attention, 4 threads (known-imperfect; user chose option 3). Noise measured alongside:
# 2 anchor runs first (W4 s0 WikiText, WT-ppl; recorded, NOT a stop gate), then the 12 runs, then one repeat of each seed-0 cell.
# Runs already in the results file with rc 0 are skipped. Commit+push after every 2 completed runs (only after jobs end).
cd "$(dirname "$0")"; FEAS=$(pwd); ROOT=$(cd .. && pwd)
export FINAL_CACHE=${FINAL_CACHE:?}; export FINAL_WORK=${FINAL_WORK:?}
PY=/opt/venvs/main/bin/python; R=results/final_S3_runs.jsonl
LOGTMP=$FINAL_WORK/logs; mkdir -p $LOGTMP; NDONE=0
BR=$(git rev-parse --abbrev-ref HEAD)
done_ok() { [ -f $R ] && $PY -c "import json,sys; sys.exit(0 if any(json.loads(l).get('run_id')=='$1' and json.loads(l).get('rc')==0 for l in open('$R')) else 1)"; }
commit_push() {
  git -C $ROOT add feasibility/results/final_S3_*.jsonl feasibility/logs/final_S3_*.log feasibility/final_S3_* 2>/dev/null
  git -C $ROOT commit -q -m "feasibility S3: $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BAoEnHYFNRwMq1jo56pXh4" || return 0
  for d in 2 4 8 16 0; do { git -C $ROOT pull -q --rebase --autostash origin $BR && git -C $ROOT push -q origin $BR; } && return 0; [ $d = 0 ] || sleep $d; done
}
run() {  # run_id evals [args]
  local id=$1 ev=$2; shift 2
  if done_ok $id; then echo "skip $id"; return 0; fi
  echo "start $id $(date -u +%H:%M:%S)"
  (cd $FINAL_WORK && $PY $FEAS/final_S3_run.py $FEAS/$R /dev/null $id gptq --evals $ev "$@") > $LOGTMP/final_S3_$id.log 2>&1
  local rc=$?
  cp $LOGTMP/final_S3_$id.log logs/final_S3_$id.log
  if [ $rc != 0 ]; then
    $PY - "$R" "$id" "$rc" "logs/final_S3_$id.log" "$ev $*" <<'PYEOF'
import json, sys, subprocess, datetime
r, i, rc, log, args = sys.argv[1:]
cpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
open(r, "a").write(json.dumps({"run_id": i, "rc": int(rc), "args": args, "log": log, "utc": datetime.datetime.utcnow().isoformat(),
  "cpu_model": next((l.split(":",1)[1].strip() for l in cpu.splitlines() if l.startswith("Model name")), None),
  "error_tail": open(log, errors="replace").read()[-2000:]}) + "\n")
PYEOF
  fi
  echo "end $id rc=$rc $(date -u +%H:%M:%S)"
  NDONE=$((NDONE+1)); [ $((NDONE % 2)) = 0 ] && commit_push "$id"
  return 0
}
echo "S3 chain start $(date -u)"
run anchor_a wt --bits 4 --seed 0 --source wikitext2 --n 128 --L 512
run anchor_b wt --bits 4 --seed 0 --source wikitext2 --n 128 --L 512
for s in 0 1 2; do for b in 4 3; do for src in wikitext2 c4; do
  run w${b}_${src}_s$s wt,c4 --bits $b --seed $s --source $src --n 128 --L 512
done; done; done
for b in 4 3; do for src in wikitext2 c4; do
  run w${b}_${src}_s0_rep wt,c4 --bits $b --seed 0 --source $src --n 128 --L 512
done; done
commit_push "chain complete"
echo "S3 chain end $(date -u)"
