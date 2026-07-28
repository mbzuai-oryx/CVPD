#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
BLINDSPOTS_JSONL="${BLINDSPOTS_JSONL:-${PROJECT_ROOT}/data/blindspots.jsonl}"
RUN_NAME="${RUN_NAME:-cvpd_8b}"
LOAD_ADAPTER="${LOAD_ADAPTER:-}"
FREEZE_VISION="${FREEZE_VISION:-1}"

args=(
  --model_name "$MODEL_NAME"
  --device "${DEVICE:-cuda}"
  --dtype "${DTYPE:-bfloat16}"
  --blindspots_jsonl "$BLINDSPOTS_JSONL"
  --image_resize "${IMAGE_RESIZE:-448}"
  --output_dir "${OUTPUT_DIR:-${PROJECT_ROOT}/runs}"
  --run_name "$RUN_NAME"
  --checkpoint_root "${CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints}"
  --log_dir "${LOG_DIR:-${PROJECT_ROOT}/logs}"
  --total_steps "${TOTAL_STEPS:--1}"
  --save_every "${SAVE_EVERY:-200}"
  --max_checkpoints "${MAX_CHECKPOINTS:-10}"
  --lr "${LR:-2e-5}"
  --weight_decay "${WEIGHT_DECAY:-0.01}"
  --grad_clip "${GRAD_CLIP:-1.0}"
  --grad_accum "${GRAD_ACCUM:-4}"
  --seed "${SEED:-42}"
  --max_answer_tokens "${MAX_ANSWER_TOKENS:-96}"
  --temperature "${TEMPERATURE:-0.7}"
  --top_p "${TOP_P:-0.9}"
  --top_k_logits "${TOP_K:-100}"
  --lambda_rank "${LAMBDA_RANK:-0.5}"
  --margin "${MARGIN:-0.10}"
  --beta_ref "${BETA_REF:-1e-3}"
  --kl_target "${KL_TARGET:-0.030}"
  --kl_adapt_rate "${KL_ADAPT_RATE:-0.10}"
  --ema_alpha "${EMA_ALPHA:-0.05}"
  --crop_pad_ratio "${CROP_PAD_RATIO:-0.20}"
  --ghost_blur_sigma "${GHOST_SIGMA:-25.0}"
  --ghost_method "${GHOST_METHOD:-blur}"
  --lora_r "${LORA_R:-32}"
  --lora_alpha "${LORA_ALPHA:-64}"
  --lora_dropout "${LORA_DROPOUT:-0.05}"
  --lora_targets "${LORA_TARGETS:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
  --clear_cache_every "${CLEAR_CACHE_EVERY:-10}"
)

if [[ "$FREEZE_VISION" != "1" ]]; then
  args+=(--no_freeze_vision)
fi
if [[ -n "$LOAD_ADAPTER" ]]; then
  args+=(--load_adapter "$LOAD_ADAPTER" --start_step "${START_STEP:-0}")
fi

cd "$PROJECT_ROOT"
exec "$PYTHON" train.py "${args[@]}"
