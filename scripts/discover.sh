#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/images}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/data/blindspots.jsonl}"
DIAGNOSTICS_PATH="${DIAGNOSTICS_PATH:-}"
MAX_IMAGES="${MAX_IMAGES:--1}"
SEED="${SEED:-42}"

args=(
  --model_name "$MODEL_NAME"
  --device "${DEVICE:-cuda}"
  --dtype "${DTYPE:-bfloat16}"
  --data_dir "$DATA_DIR"
  --output_path "$OUTPUT_PATH"
  --image_resize "${IMAGE_RESIZE:-448}"
  --max_images "$MAX_IMAGES"
  --seed "$SEED"
  --probe_answer_tokens "${PROBE_TOKENS:-24}"
  --top_k_logits "${TOP_K:-100}"
  --tau_crop_disagree "${TAU_CROP_DISAGREE:-0.05}"
  --tau_ghost_agree "${TAU_GHOST_AGREE:-0.05}"
  --max_kept_per_image "${MAX_KEPT_PER_IMAGE:-1}"
  --box_min_area_frac "${BOX_MIN_AREA_FRAC:-0.01}"
  --box_max_area_frac "${BOX_MAX_AREA_FRAC:-0.50}"
  --log_every "${LOG_EVERY:-25}"
)

if [[ -n "$DIAGNOSTICS_PATH" ]]; then
  args+=(--diagnostics_path "$DIAGNOSTICS_PATH")
fi

cd "$PROJECT_ROOT"
exec "$PYTHON" discover.py "${args[@]}"
