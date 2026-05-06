#!/usr/bin/env bash
set -euo pipefail

ROOT="/data8/cwkang/workspace/edge_attribution_patching"
PY="/home/cwkang/miniconda3/envs/EAP/bin/python"
LOG_DIR="${ROOT}/logs/20260504_default_all"
mkdir -p "${LOG_DIR}"
cd "${ROOT}"

declare -a JOBS=(
  "EAP.py|known_1000|AlgorithmicResearchGroup/gpt2-xs"
  "EAP.py|known_1000|EleutherAI/pythia-14m"
  "EAP.py|known_1000|EleutherAI/pythia-1b"
  "EAP.py|lama_trex|AlgorithmicResearchGroup/gpt2-xs"
  "EAP.py|lama_trex|EleutherAI/pythia-14m"
  "EAP.py|lama_trex|EleutherAI/pythia-1b"
  "EAP_IG.py|known_1000|AlgorithmicResearchGroup/gpt2-xs"
  "EAP_IG.py|known_1000|EleutherAI/pythia-14m"
  "EAP_IG.py|known_1000|EleutherAI/pythia-1b"
  "EAP_IG.py|lama_trex|AlgorithmicResearchGroup/gpt2-xs"
  "EAP_IG.py|lama_trex|EleutherAI/pythia-14m"
  "EAP_IG.py|lama_trex|EleutherAI/pythia-1b"
  "EAP_vit.py|imagenet|vit_tiny_patch16_224"
  "EAP_vit.py|officehome|vit_tiny_patch16_224"
  "EAP_vit.py|imagenet|deit_tiny_patch16_224"
  "EAP_vit.py|officehome|deit_tiny_patch16_224"
  "EAP_IG_vit.py|imagenet|vit_tiny_patch16_224"
  "EAP_IG_vit.py|officehome|vit_tiny_patch16_224"
  "EAP_IG_vit.py|imagenet|deit_tiny_patch16_224"
  "EAP_IG_vit.py|officehome|deit_tiny_patch16_224"
)

slugify() {
  printf '%s' "$1" | tr '/:' '__'
}

run_job() {
  local gpu="$1"
  local job="$2"
  local script dataset model script_slug model_slug log
  IFS='|' read -r script dataset model <<< "${job}"
  script_slug="$(slugify "${script%.py}")"
  model_slug="$(slugify "${model}")"
  log="${LOG_DIR}/${script_slug}_${dataset}_${model_slug}_gpu${gpu}.log"
  echo "[$(date '+%F %T')] START gpu=${gpu} script=${script} dataset=${dataset} model=${model} log=${log}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export OMP_NUM_THREADS=1
    stdbuf -oL -eL "${PY}" -u "src/${script}" \
      --dataset_name "${dataset}" \
      --model_name "${model}"
  ) > "${log}" 2>&1
  echo "[$(date '+%F %T')] DONE gpu=${gpu} script=${script} dataset=${dataset} model=${model} status=$?"
}

worker() {
  local worker_id="$1"
  local gpu="$2"
  local i
  for i in "${!JOBS[@]}"; do
    if (( i % 4 == worker_id )); then
      run_job "${gpu}" "${JOBS[$i]}"
    fi
  done
}

echo "[$(date '+%F %T')] launching ${#JOBS[@]} default jobs with 4 GPU workers"
worker 0 0 &
worker 1 1 &
worker 2 2 &
worker 3 3 &
wait
echo "[$(date '+%F %T')] all default jobs finished"
