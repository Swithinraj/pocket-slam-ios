#!/usr/bin/env bash
# Creates the Python env for the offline SLAM / BA tools (WSL Ubuntu).
# Usage (from the repo root): bash tools/setup_env.sh
set -euo pipefail
VENV="$HOME/venvs/slamtogo"
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r "$(dirname "$0")/requirements.txt"
"$VENV/bin/python" -c 'import numpy, scipy, cv2, open3d; print("numpy", numpy.__version__, "| scipy", scipy.__version__, "| cv2", cv2.__version__, "| open3d", open3d.__version__)'
echo "Activate with: source $VENV/bin/activate"
