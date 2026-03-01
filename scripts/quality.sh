#!/bin/bash
# Run code quality checks.
# Usage:
#   ./scripts/quality.sh          # check only (no changes)
#   ./scripts/quality.sh --fix    # apply formatting

set -euo pipefail

FIX=false
for arg in "$@"; do
    [[ "$arg" == "--fix" ]] && FIX=true
done

TARGETS="backend/ main.py"

echo "=== Formatting (black) ==="
if $FIX; then
    uv run black $TARGETS
else
    uv run black --check $TARGETS
fi

echo ""
echo "=== Tests (pytest) ==="
cd backend && uv run pytest tests/ -q
