#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
exec "$PYTHON" src/sft/launch.py wan5b "$@"
