#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
CONFIG="${CONFIG:-configs/wan1p3b_stage2.json}"
if [[ "${1:-}" == "--preflight" ]]; then
  exec "$PYTHON" src/distill/wan1p3b/distributed_train.py stage2 --config "$CONFIG" "$@"
fi
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 src/distill/wan1p3b/distributed_train.py stage2 --config "$CONFIG" "$@"
