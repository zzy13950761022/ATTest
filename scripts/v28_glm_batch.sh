#!/bin/bash
set -eo pipefail
export ATTEST_CONFIG_DIR=/root/.attest_cli_glm
export MODEL=glm
export MODEL_DISPLAY=glm-5.2
exec bash /mnt/fangcr/ATTest/scripts/v28_model_batch.sh.template "$@"
