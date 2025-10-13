#!/bin/zsh
set -euo pipefail

# change_dataset.command (v2)
# - Uses existing env
# - Lists datasets in ./data and ./data/_pool
# - Moves all zips from ./data to ./data/_pool, then copies chosen one back to ./data (so only one zip remains)
# - Prompts for column/min/max
# - Runs pipeline
# - Does NOT start Streamlit

info()  { echo "\033[1;34m[INFO]\033[0m $*"; }
ok()    { echo "\033[1;32m[OK]\033[0m $*"; }
warn()  { echo "\033[1;33m[WARN]\033[0m $*"; }
err()   { echo "\033[1;31m[ERR]\033[0m $*"; }

pause_on_error() {
  warn "The script encountered an error."
  read -r "?Press Enter to close..."
}
trap pause_on_error ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" >/dev/null 2>&1 ; pwd -P )"
APP_DIR="$SCRIPT_DIR"
if [[ ! -f "$APP_DIR/environment.yml" ]]; then
  err "change_dataset.command must be in the repo root (environment.yml missing)."
  exit 1
fi

# Determine env
ENV_NAME="$(grep -E '^[[:space:]]*name:[[:space:]]*' "$APP_DIR/environment.yml" | head -n1 | awk -F':' '{print $2}' | xargs)"
[[ -z "$ENV_NAME" ]] && ENV_NAME="capstone-py310"

# Init conda
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
else
  CONDA_BASE="$HOME/miniforge3"
fi
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Data dirs
DATA_DIR="$APP_DIR/data"
POOL_DIR="$DATA_DIR/_pool"
mkdir -p "$DATA_DIR" "$POOL_DIR"

zroot=( "$DATA_DIR"/*.zip(N) )
zpool=( "$POOL_DIR"/*.zip(N) )
if (( ${#zroot[@]} + ${#zpool[@]} == 0 )); then
  err "No dataset ZIPs found. Put them in: $DATA_DIR"
  exit 1
fi

echo "Datasets available:"
for z in "${zroot[@]}"; do echo "  - ${z:t} (in data/)"; done
for z in "${zpool[@]}"; do echo "  - ${z:t} (in data/_pool/)"; done
echo ""

# Pick dataset
while true; do
  read -r "DATA_NAME?Enter the dataset ZIP filename to switch to: "
  DATA_NAME="${DATA_NAME%.zip}.zip"
  if [[ -f "$DATA_DIR/$DATA_NAME" ]] || [[ -f "$POOL_DIR/$DATA_NAME" ]]; then
    ok "Selected dataset: $DATA_NAME"
    break
  else
    warn "Not found: $DATA_NAME in data/ or data/_pool/"
  fi
done

# Move all zips from data to pool; then copy chosen back
find "$DATA_DIR" -maxdepth 1 -type f -name "*.zip" -exec mv -f "{}" "$POOL_DIR"/ \; 2>/dev/null || true
if [[ -f "$POOL_DIR/$DATA_NAME" ]]; then
  cp -f "$POOL_DIR/$DATA_NAME" "$DATA_DIR/$DATA_NAME"
else
  err "Selected dataset not found in pool after move."
  exit 1
fi

# Confirm only one zip in data
count_now="$(find "$DATA_DIR" -maxdepth 1 -type f -name "*.zip" | wc -l | tr -d ' ')"
if [[ "$count_now" != "1" ]]; then
  err "Expected exactly one .zip in data/ but found $count_now"
  exit 1
fi

# Prompts for col/min/max
VALUE_COL=""
while [[ -z "$VALUE_COL" ]]; do
  read -r "VALUE_COL?Enter the column name of interest (e.g., CU_PPM): "
  VALUE_COL="${VALUE_COL//[[:space:]]/}"
  if [[ -z "$VALUE_COL" ]]; then warn "Column name cannot be empty."; fi
done
read -r "MIN_VAL?Enter the minimum value (press Enter for default 1e-5): "
read -r "MAX_VAL?Enter the maximum value (press Enter to skip): "
if [[ -z "${MIN_VAL// }" ]]; then MIN_VAL="1e-5"; fi

RAW_REL_PATH="data/$DATA_NAME"
CMD=( python tools/run_all_test.py --value-col "$VALUE_COL" --value-min "$MIN_VAL" --raw-zip "$RAW_REL_PATH" )
if [[ -n "${MAX_VAL// }" ]]; then
  CMD=( python tools/run_all_test.py --value-col "$VALUE_COL" --value-min "$MIN_VAL" --value-max "$MAX_VAL" --raw-zip "$RAW_REL_PATH" )
fi

cd "$APP_DIR"
info "Running pipeline: ${CMD[@]}"
"${CMD[@]}" || warn "Pipeline returned non-zero (continuing)."

ok "Dataset switched successfully to: $DATA_NAME"
echo "Now run: run.command   to start the app."
