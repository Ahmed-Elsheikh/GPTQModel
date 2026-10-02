#!/usr/bin/env bash
# Sequential chain: Test 2, Test 3, Test 4 (one job at a time so timings are not contended)
cd "$(dirname "$0")/.."
echo "$(date -Is) chain_A start" >> feasibility/logs/chain_A.log
feasibility/run_t2.sh real a b b_torch c0 c_fs c_bs c_dt c_ml d_real d >> feasibility/logs/chain_A.log 2>&1
echo "$(date -Is) t2 done" >> feasibility/logs/chain_A.log
feasibility/run_t3.sh a >> feasibility/logs/chain_A.log 2>&1; echo "$(date -Is) t3a done" >> feasibility/logs/chain_A.log
feasibility/run_t3.sh b >> feasibility/logs/chain_A.log 2>&1; echo "$(date -Is) t3b done" >> feasibility/logs/chain_A.log
source /home/user/venv/bin/activate
/usr/bin/time -v -o feasibility/results/t4_wanda_tf5.time python feasibility/wanda_synth_driver.py --model $PWD/feasibility/models/synth-SmolLM2-135M \
  --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --nsamples 32 --seed 0 --save $PWD/feasibility/results/t4_wanda_tf5 \
  > feasibility/logs/t4_wanda_tf5.log 2>&1
echo "$(date -Is) t4 (tf5) rc=$? done" >> feasibility/logs/chain_A.log
