#!/usr/bin/env bash
set -euo pipefail

XML_PATH="artifact/rundefinitions/pynguinml-tensorflow.xml"
START_LINE=66
END_LINE=121
MODE="full-auto"
EPOCH=5
WORKSPACE_ROOT="./exam/tensorflow"

modules="$(
  sed -n "${START_LINE},${END_LINE}p" "${XML_PATH}" \
    | awk -F'[<>]' '/<module>/{print $3}' \
    | sed '/^[[:space:]]*$/d'
)"

if [ -z "${modules}" ]; then
  echo "No modules found in ${XML_PATH} lines ${START_LINE}-${END_LINE}." >&2
  exit 1
fi

failures=()

while IFS= read -r module; do
  workspace="${WORKSPACE_ROOT}/${module#tensorflow.}"
  mkdir -p "${workspace}"
  echo "==> ${module}"
  echo "    workspace: ${workspace}"
  if ! testagent run --func "${module}" --workspace "${workspace}" --mode "${MODE}" --epoch "${EPOCH}"; then
    failures+=("${module}")
    echo "    ✗ failed: ${module}" >&2
  fi
done <<< "${modules}"

if [ "${#failures[@]}" -ne 0 ]; then
  echo "Failed modules:" >&2
  printf '  - %s\n' "${failures[@]}" >&2
  exit 1
fi

echo "All modules completed."
