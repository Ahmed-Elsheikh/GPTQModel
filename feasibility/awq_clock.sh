#!/usr/bin/env bash
log=$1; out=$2; last=""
while true; do
  l=$(grep -oE "AWQProcessor: layer [0-9]+ tracking" "$log" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
  if [ "$l" != "$last" ]; then echo "$(date -Is) layer=$l" >> "$out"; last=$l; fi
  grep -qE "eval_quant:|Traceback" "$log" 2>/dev/null && { echo "$(date -Is) END" >> "$out"; exit 0; }
  sleep 5
done
