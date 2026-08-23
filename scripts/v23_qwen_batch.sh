#!/bin/bash
set -eo pipefail
export ATTEST_CONFIG_DIR=/root/.attest_cli_qwen
export MODEL=qwen
export MODEL_DISPLAY=qwen3-coder-plus
exec bash /mnt/fangcr/ATTest/scripts/v23_model_batch.sh.template "$@"
