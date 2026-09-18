#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" tests/contract_wan5b.py
exec "$PYTHON" tests/contract_wan1p3b.py
