#!/bin/bash
# Lock GPU power limit on all cards (default 225 W; override via GPU_POWER_LIMIT_W).
set -u
WATTS=${GPU_POWER_LIMIT_W:-225}
for g in 0 1; do
  if nvidia-smi -i "$g" -pl "$WATTS" >/dev/null 2>&1; then
    echo "gpu$g -> ${WATTS} W"
  else
    echo "gpu$g -> FAILED (nvidia-smi -pl ${WATTS})"
  fi
done
nvidia-smi --query-gpu=index,power.limit --format=csv,noheader
