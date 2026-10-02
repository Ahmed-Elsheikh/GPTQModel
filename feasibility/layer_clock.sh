#!/usr/bin/env bash
# usage: layer_clock.sh <log> <out>; appends "<iso time> <latest layer index>" whenever it changes
log=$1; out=$2; last=""
while true; do
  l=$(grep -oE "forward_capture_complete layer=[0-9]+" "$log" 2>/dev/null | tail -1 | grep -oE "[0-9]+$")
  if [ "$l" != "$last" ]; then echo "$(date -Is) layer=$l" >> "$out"; last=$l; fi
  grep -qE "eval_quant:|Traceback" "$log" 2>/dev/null && { echo "$(date -Is) END" >> "$out"; exit 0; }
  sleep 5
done
