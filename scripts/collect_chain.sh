#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
exec "$PYTHON" src/data/collect_chain.py "$@"
