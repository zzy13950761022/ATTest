#!/bin/bash
set -eo pipefail
export ATTEST_CONFIG_DIR=/root/.attest_cli_glm5
export MODEL=glm
export MODEL_DISPLAY=glm-5.2
exec bash /mnt/fangcr/ATTest/scripts/v15_model_batch_v7.sh.template "$@"
