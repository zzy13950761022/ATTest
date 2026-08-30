#!/bin/bash
set -eo pipefail
export ATTEST_CONFIG_DIR=/root/.attest_cli_ds
export MODEL=ds
export MODEL_DISPLAY=deepseek-v4-pro
exec bash /mnt/fangcr/ATTest/scripts/v28_model_batch.sh.template "$@"
