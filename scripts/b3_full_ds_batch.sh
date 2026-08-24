#!/bin/bash
set -eo pipefail
export ATTEST_CONFIG_DIR=/root/.attest_cli_ds
export MODEL=ds
case ds in
  qwen) export MODEL_DISPLAY=qwen3-coder-plus ;;
  ds)   export MODEL_DISPLAY=deepseek-v4-pro ;;
  glm)  export MODEL_DISPLAY=glm-5.2 ;;
esac
exec bash /mnt/fangcr/ATTest/scripts/b3_full_model_batch.sh.template "$@"
