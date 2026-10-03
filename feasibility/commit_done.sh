#!/bin/bash
# Commit + push feasibility/ changes, leaving out files that a running process still has open for writing
# (live run logs): a rebase that rewrites such a file would detach the writer and truncate the committed log.
# usage: commit_done.sh "message"
cd "$(dirname "$0")/.."
open_files=$(for p in /proc/[0-9]*; do ls -l $p/fd 2>/dev/null; done | awk '{print $NF}' | grep "$PWD/feasibility/logs/" | sort -u)
# also skip logs written since the oldest running job started (GPTQModel reopens its gptq_log_* per write)
for pid in $(pgrep -f "runlog.py" 2>/dev/null); do
  [ -d /proc/$pid ] && open_files="$open_files $(find "$PWD/feasibility/logs" -type f -newer /proc/$pid 2>/dev/null)"
done
git add -A feasibility .gitignore
for f in $open_files; do git reset -q -- "$f" 2>/dev/null; done
git diff --cached --quiet && { echo "nothing to commit"; exit 0; }
git commit -q -m "$1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XuYpAPbe9LM1DDrcYJToa8"
for d in 2 4 8 16 0; do
  git pull -q --rebase --autostash origin claude/quirky-ramanujan-i0hree && git push -q origin claude/quirky-ramanujan-i0hree && { echo "pushed $(git log --oneline -1)"; exit 0; }
  [ $d = 0 ] || sleep $d
done
echo "PUSH FAILED"; exit 1
