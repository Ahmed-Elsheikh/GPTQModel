#!/usr/bin/env bash
cd "$(dirname "$0")/.."
until grep -q "chain_B2 done" feasibility/logs/chain_B2.log 2>/dev/null; do sleep 30; done
feasibility/run_t2.sh d2 >> feasibility/logs/chain_D.log 2>&1
echo "$(date -Is) chain_D done" >> feasibility/logs/chain_D.log
