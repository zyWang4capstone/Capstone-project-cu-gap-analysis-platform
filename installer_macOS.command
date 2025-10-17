#!/bin/zsh
set -euo pipefail

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

# Verify repo root
if [[ ! -f "$APP_DIR/environment.yml" ]]; then
  err "environment.yml not found; place installer.command in the repo root."
  exit 1
fi
ok "Repo detected at: $APP_DIR"

# Ensure conda (Miniforge)
if ! command -v conda >/dev/null 2>&1; then
  info "Installing Miniforge (conda) ..."
  ARCH="$(uname -m)"
  case "$ARCH" in
    arm64)  MF_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh" ;;
    x86_64) MF_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh" ;;
    *)      MF_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh" ;;
  esac
  TMPDIR="$(mktemp -d)"
  curl -L "$MF_URL" -o "$TMPDIR/miniforge.sh"
  bash "$TMPDIR/miniforge.sh" -b -p "$HOME/miniforge3"
  rm -rf "$TMPDIR"
  ok "Miniforge installed at $HOME/miniforge3"
fi

# Initialize conda
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
else
  CONDA_BASE="$HOME/miniforge3"
fi
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Env name
ENV_NAME="$(grep -E '^[[:space:]]*name:[[:space:]]*' "$APP_DIR/environment.yml" | head -n1 | awk -F':' '{print $2}' | xargs)"
[[ -z "$ENV_NAME" ]] && ENV_NAME="capstone-py310"
ok "Using conda env: $ENV_NAME"

# Create or update env
cd "$APP_DIR"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  info "Updating env '$ENV_NAME' from environment.yml ..."
  conda env update -n "$ENV_NAME" -f environment.yml -q
else
  info "Creating env '$ENV_NAME' from environment.yml ..."
  conda env create -n "$ENV_NAME" -f environment.yml -q
fi

# Activate env
conda activate "$ENV_NAME"
python -V || true

# Data handling: ensure only ONE zip at ./data when running pipeline
DATA_DIR="$APP_DIR/data"
POOL_DIR="$DATA_DIR/_pool"
mkdir -p "$DATA_DIR" "$POOL_DIR"

# Gather zip list from root and pool
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

# Prompt selection
while true; do
  read -r "DATA_NAME?Enter the dataset ZIP filename to use (e.g., MyData.zip): "
  DATA_NAME="${DATA_NAME%.zip}.zip"
  if [[ -f "$DATA_DIR/$DATA_NAME" ]] || [[ -f "$POOL_DIR/$DATA_NAME" ]]; then
    ok "Selected dataset: $DATA_NAME"
    break
  else
    warn "Not found: $DATA_NAME in data/ or data/_pool/"
  fi
done

# Move ALL zips from data/ to data/_pool (so data/ has none)
find "$DATA_DIR" -maxdepth 1 -type f -name "*.zip" -exec mv -f "{}" "$POOL_DIR"/ \; 2>/dev/null || true
# Copy the selected one from pool back to data/
if [[ -f "$POOL_DIR/$DATA_NAME" ]]; then
  cp -f "$POOL_DIR/$DATA_NAME" "$DATA_DIR/$DATA_NAME"
else
  # If the selected file was only in data/ (rare after move), restore it from a temp
  err "Unexpected: selected dataset missing from pool after move."
  exit 1
fi

# Confirm now only one zip at data/
count_now="$(find "$DATA_DIR" -maxdepth 1 -type f -name "*.zip" | wc -l | tr -d ' ')"
if [[ "$count_now" != "1" ]]; then
  err "Expected exactly one .zip in data/ but found $count_now"
  exit 1
fi

# Prompt for column / min / max
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

info "Running pipeline: ${CMD[@]}"
"${CMD[@]}" || warn "Pipeline returned non-zero (continuing)."

ok "Installation complete using dataset: $DATA_NAME"
echo "To start the app, double-click: run.command"
