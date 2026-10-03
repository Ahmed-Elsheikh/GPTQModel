# shared helpers for the Step 2 parallel-worker chains (sourced)
cd "$(dirname "${BASH_SOURCE[0]}")"
OUT=${STEP2_OUT:-/tmp/step2_out}; mkdir -p "$OUT"
NRUN=0
commit_push() {  # commit+push only Step 2 files; retried with backoff on network errors
  git add results/real_step2_*.jsonl logs/real_s2_*.log step2_*.sh chain_step2*.sh wanda_real_driver.py 2>/dev/null
  git commit -q -m "feasibility step2: $1" || return 0
  for d in 2 4 8 16 0; do git push -q -u origin claude/quirky-ramanujan-i0hree && return 0; [ $d = 0 ] || sleep $d; done
}
after_run() { NRUN=$((NRUN+1)); [ $((NRUN % 2)) = 0 ] && commit_push "$1"; true; }
