#!/usr/bin/env bash
cd "$(dirname "$0")/.."
until grep -q "chain_B done" feasibility/logs/chain_B.log 2>/dev/null; do sleep 20; done
source /home/user/venv/bin/activate; export OMP_NUM_THREADS=4
echo "$(date -Is) start t2b_inmemory" >> feasibility/logs/chain_C.log
/usr/bin/time -v -o feasibility/results/t2/b_inmemory.time python feasibility/lmeval_inmemory.py feasibility/models/q-gptq-135m-s0 feasibility/results/t2/b_inmemory.json > feasibility/logs/t2_b_inmemory.log 2>&1
echo "$(date -Is) end t2b_inmemory rc=$?" >> feasibility/logs/chain_C.log
