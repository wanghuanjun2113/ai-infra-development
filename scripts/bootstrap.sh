#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

python3 scripts/dev.py check --create-dirs
python3 scripts/dev.py sync clone
python3 scripts/dev.py sync status

echo
echo "ai-infra-development bootstrap complete."
