#!/bin/bash
# S1 resumable chain. Anchor gate first (eager, W4 g128 s0, 128x512; each anchor within 0.05 ppl of 17.660599),
# then dense -> W3 s0-4 -> W4 s0-4 -> W3 s5-9 (128x2048 WikiText calibration). C4-ppl + lm-eval on dense and seeds 0-2.
# Runs whose run_id already has rc 0 in the results file are skipped. Commit+push after every 2 completed runs.
cd "$(dirname "$0")"; FEAS=$(pwd); ROOT=$(cd .. && pwd)
export FINAL_CACHE=${FINAL_CACHE:?}; export FINAL_WORK=${FINAL_WORK:?}
PY=/opt/venvs/main/bin/python; R=results/final_S1_runs.jsonl; I=results/final_S1_lmeval_items.jsonl
LOGTMP=$FINAL_WORK/logs; mkdir -p $LOGTMP; NDONE=0
BR=$(git rev-parse --abbrev-ref HEAD)
done_ok() { [ -f $R ] && $PY -c "import json,sys; sys.exit(0 if any(json.loads(l).get('run_id')=='$1' and json.loads(l).get('rc')==0 for l in open('$R')) else 1)"; }
commit_push() {
  git -C $ROOT add feasibility/results/final_S1_*.jsonl feasibility/logs/final_S1_*.log feasibility/final_S1_* 2>/dev/null
  git -C $ROOT commit -q -m "feasibility S1: $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BAoEnHYFNRwMq1jo56pXh4" || return 0
  for d in 2 4 8 16 0; do { git -C $ROOT pull -q --rebase --autostash origin $BR && git -C $ROOT push -q origin $BR; } && return 0; [ $d = 0 ] || sleep $d; done
}
run() {  # run_id kind evals [extra args]
  local id=$1 kind=$2 ev=$3; shift 3
  if done_ok $id; then echo "skip $id"; return 0; fi
  echo "start $id $(date -u +%H:%M:%S)"
  (cd $FINAL_WORK && $PY $FEAS/final_S1_run.py $FEAS/$R $FEAS/$I $id $kind --evals $ev "$@") > $LOGTMP/final_S1_$id.log 2>&1
  local rc=$?
  cp $LOGTMP/final_S1_$id.log logs/final_S1_$id.log   # copy only after the job ended
  if [ $rc != 0 ]; then
    $PY - "$R" "$id" "$rc" "logs/final_S1_$id.log" "$kind $ev $*" <<'PYEOF'
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
  return $rc
}
echo "S1 chain start $(date -u)"
# --- anchor gate ---
run anchor_a gptq wt --bits 4 --seed 0 --n 128 --L 512
run anchor_b gptq wt --bits 4 --seed 0 --n 128 --L 512
GATE=$($PY - "$R" <<'PYEOF'
import json, sys
# Gate (user decision 2026-10-04): an anchor passes if its WT-ppl is within 0.05 of 17.660599 (eager, W4 g128 s0, 128x512).
rows = {json.loads(l)["run_id"]: json.loads(l) for l in open(sys.argv[1])}
res = []
for k in ("anchor_a", "anchor_b"):
    r = rows.get(k, {}); v = r.get("metrics", {}).get("wt_ppl")
    res.append((k, v, r.get("rc") == 0 and v is not None and abs(v - 17.660599) <= 0.05))
print("PASS" if all(ok for *_, ok in res) else "FAIL", res)
PYEOF
)
echo "anchor gate (|ppl-17.660599|<=0.05): $GATE"
if [[ $GATE != PASS* ]]; then commit_push "anchor gate FAILED"; echo "STOP: anchor gate failed"; exit 3; fi
# --- main runs ---
run dense dense wt,c4,lmeval
# noise repeats (WT-ppl + checkpoint sha256 only): W3 s0 repeat right after W3 s0, W4 s0 repeat right after W4 s0
for s in 0 1 2 3 4; do ev=wt; [ $s -le 2 ] && ev=wt,c4,lmeval; run w3_s$s gptq $ev --bits 3 --seed $s --n 128 --L 2048
  [ $s = 0 ] && run w3_s0_rep gptq wt --bits 3 --seed 0 --n 128 --L 2048; done
for s in 0 1 2 3 4; do ev=wt; [ $s -le 2 ] && ev=wt,c4,lmeval; run w4_s$s gptq $ev --bits 4 --seed $s --n 128 --L 2048
  [ $s = 0 ] && run w4_s0_rep gptq wt --bits 4 --seed 0 --n 128 --L 2048; done
for s in 5 6 7 8 9; do run w3_s$s gptq wt --bits 3 --seed $s --n 128 --L 2048; done
commit_push "chain complete"
echo "S1 chain end $(date -u)"
