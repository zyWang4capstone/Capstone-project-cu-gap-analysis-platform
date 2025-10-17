#!/bin/zsh
set -euo pipefail

info()  { echo "\033[1;34m[INFO]\033[0m $*"; }
err()   { echo "\033[1;31m[ERR]\033[0m $*"; }

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" >/dev/null 2>&1 ; pwd -P )"
APP_DIR="$SCRIPT_DIR"

if [[ ! -f "$APP_DIR/environment.yml" || ! -f "$APP_DIR/start.py" ]]; then
  err "run.command must be in the repo root (with environment.yml and start.py)."
  exit 1
fi

ENV_NAME="$(grep -E '^[[:space:]]*name:[[:space:]]*' "$APP_DIR/environment.yml" | head -n1 | awk -F':' '{print $2}' | xargs)"
[[ -z "$ENV_NAME" ]] && ENV_NAME="capstone-py310"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
else
  CONDA_BASE="$HOME/miniforge3"
fi
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate "$ENV_NAME" || { err "Failed to activate env '$ENV_NAME'"; exit 1; }

cd "$APP_DIR"
info "Launching Streamlit (start.py) ..."
exec streamlit run start.py
