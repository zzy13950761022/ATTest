#!/bin/bash
set -eo pipefail
export ATTEST_CONFIG_DIR=/root/.attest_cli_ds_v4_pro
export MODEL=ds
export MODEL_DISPLAY=deepseek-v4-pro
exec bash /mnt/fangcr/ATTest/scripts/v23_model_batch.sh.template "$@"
