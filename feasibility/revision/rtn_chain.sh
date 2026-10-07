#!/bin/bash
# Revision step 9 chain (resumable): RTN W4 then W3, each: WT-ppl, C4-ppl, six S4 protocols. Commits after each bit-width.
cd "$(dirname "$0")"; REV=$(pwd); ROOT=$(cd ../.. && pwd); BR=$(git rev-parse --abbrev-ref HEAD)
export FINAL_CACHE=${FINAL_CACHE:?}; PY=/opt/venvs/main/bin/python; R=$REV/rtn_runs.jsonl
echo "RTN chain start $(date -u)"
for b in 4 3; do
  echo "start rtn w$b $(date -u +%H:%M:%S)"; $PY $REV/rtn_eval.py $R $b > $REV/rtn_w$b.log 2>&1; echo "end rtn w$b rc=$? $(date -u +%H:%M:%S)"
  git -C $ROOT add feasibility/revision/rtn_runs.jsonl feasibility/revision/rtn_w$b.log feasibility/revision/rtn_eval.py feasibility/revision/rtn_chain.sh
  git -C $ROOT commit -q -m "revision: RTN W$b g128 evals

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BAoEnHYFNRwMq1jo56pXh4" && for d in 2 4 8 0; do { git -C $ROOT pull -q --rebase --autostash origin $BR && git -C $ROOT push -q origin $BR; } && break; [ $d = 0 ] || sleep $d; done
done
echo "RTN chain end $(date -u)"
