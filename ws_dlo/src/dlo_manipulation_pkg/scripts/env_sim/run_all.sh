#!/usr/bin/env bash
set +e  # do NOT exit on error

scripts=(
  "GNN_offline_v3_mask_1.py"
  "GNN_offline_v3_mask_variable.py"
  "GNN_offline_v3_mixed.py"
)

mkdir -p logs
fail=0

for s in "${scripts[@]}"; do
  ts=$(date +"%Y%m%d_%H%M%S")
  log="logs/${s%.py}_${ts}.log"

  echo "=== Running: $s at $ts ===" | tee -a "$log"
  python "$s" >>"$log" 2>&1
  rc=$?

  echo "=== Exit code: $rc for $s ===" | tee -a "$log"
  echo | tee -a "$log"

  if [ $rc -ne 0 ]; then
    fail=1
  fi
done

exit $fail

