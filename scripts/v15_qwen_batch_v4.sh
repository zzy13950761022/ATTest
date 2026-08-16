#!/bin/bash
# v15 Qwen v4 batch launcher
set -eo pipefail
export ATTEST_CONFIG_DIR=/root/.attest_cli_qwen
export MODEL=qwen
export MODEL_DISPLAY=qwen3-coder
exec bash /mnt/fangcr/ATTest/scripts/v15_model_batch_v4.sh.template "$@"
