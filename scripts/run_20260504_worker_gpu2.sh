#!/usr/bin/env bash
set -euo pipefail

ROOT="/data8/cwkang/workspace/edge_attribution_patching"
PY="/home/cwkang/miniconda3/envs/EAP/bin/python"
GPU=2
WORKER=2
LOG_ROOT="${ROOT}/logs/20260504_sh_workers"
STATE_DIR="${LOG_ROOT}/state"
mkdir -p "${LOG_ROOT}/default" "${LOG_ROOT}/logit_diff" "${STATE_DIR}"
cd "${ROOT}"

slugify() { printf '%s' "$1" | tr '/:' '__' | sed 's/\.py$//'; }

run_job() {
  local mode="$1" script="$2" dataset="$3" model="$4"
  local script_slug model_slug log extra=()
  script_slug="$(slugify "$script")"
  model_slug="$(slugify "$model")"
  log="${LOG_ROOT}/${mode}/${script_slug}_${dataset}_${model_slug}_gpu${GPU}.log"
  if [[ "$mode" == "logit_diff" ]]; then
    extra=(--score_function logit_diff --overwrite)
  fi
  echo "[$(date '+%F %T')] START gpu=${GPU} mode=${mode} script=${script} dataset=${dataset} model=${model}" | tee "$log"
  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" OMP_NUM_THREADS=1 "${PY}" -u "src/${script}" \
    --dataset_name "${dataset}" --model_name "${model}" "${extra[@]}" >> "$log" 2>&1
  local status=$?
  set -e
  echo "[$(date '+%F %T')] DONE gpu=${GPU} mode=${mode} script=${script} dataset=${dataset} model=${model} status=${status}" | tee -a "$log"
  return "${status}"
}

wait_for_default_done() {
  touch "${STATE_DIR}/default_done_gpu${GPU}"
  while [[ "$(find "${STATE_DIR}" -maxdepth 1 -name 'default_done_gpu*' | wc -l)" -lt 4 ]]; do
    sleep 30
  done
}

rename_once() {
  if mkdir "${STATE_DIR}/rename.lock" 2>/dev/null; then
    echo "score_function-specific output roots are used; no rename needed" > "${STATE_DIR}/rename.done"
  fi
  while [[ ! -f "${STATE_DIR}/rename.done" ]]; do
    sleep 10
  done
}

run_job default EAP.py known_1000 EleutherAI/pythia-1b
run_job default EAP_IG.py known_1000 AlgorithmicResearchGroup/gpt2-xs
run_job default EAP_IG.py lama_trex EleutherAI/pythia-14m
run_job default EAP_vit.py imagenet deit_tiny_patch16_224
run_job default EAP_IG_vit.py imagenet deit_tiny_patch16_224

wait_for_default_done
rename_once

run_job logit_diff EAP.py known_1000 EleutherAI/pythia-1b
run_job logit_diff EAP_IG.py known_1000 AlgorithmicResearchGroup/gpt2-xs
run_job logit_diff EAP_IG.py lama_trex EleutherAI/pythia-14m
run_job logit_diff EAP_vit.py imagenet deit_tiny_patch16_224
run_job logit_diff EAP_IG_vit.py imagenet deit_tiny_patch16_224

touch "${STATE_DIR}/logit_diff_done_gpu${GPU}"
