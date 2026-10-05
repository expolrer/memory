#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:?GPU_ID is required}"
TASK_IDS="${TASK_IDS:?TASK_IDS is required (comma-separated)}"
NUM_TRIALS="${NUM_TRIALS:-50}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_spatial}"

EVAL_SCRIPT=/ssd/DySta/dysta_reproduction/scripts/eval_libero_spatial_dysta_gpu.sh
SUMMARY_DIR="${SUMMARY_DIR:-/ssd/DySta/eval-logs/${TASK_SUITE_NAME//_/-}/batch}"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUMMARY_PATH="${SUMMARY_DIR}/${TASK_SUITE_NAME}-gpu${GPU_ID}-${RUN_STAMP}.tsv"

mkdir -p "${SUMMARY_DIR}"
printf 'task_id\tgpu_id\tnum_trials\tstarted_utc\tfinished_utc\telapsed_seconds\texit_code\n' > "${SUMMARY_PATH}"

overall_status=0
IFS=',' read -r -a task_array <<< "${TASK_IDS}"
for task_id in "${task_array[@]}"; do
  started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  started_epoch="$(date +%s)"

  set +e
  GPU_ID="${GPU_ID}" TASK_ID="${task_id}" NUM_TRIALS="${NUM_TRIALS}" \
    TASK_SUITE_NAME="${TASK_SUITE_NAME}" bash "${EVAL_SCRIPT}"
  exit_code=$?
  set -e

  finished_epoch="$(date +%s)"
  finished_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  elapsed_seconds=$((finished_epoch - started_epoch))
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${task_id}" "${GPU_ID}" "${NUM_TRIALS}" "${started_utc}" "${finished_utc}" \
    "${elapsed_seconds}" "${exit_code}" >> "${SUMMARY_PATH}"

  if ((exit_code != 0)); then
    overall_status=1
  fi
done

printf 'Batch summary: %s\n' "${SUMMARY_PATH}"
exit "${overall_status}"
