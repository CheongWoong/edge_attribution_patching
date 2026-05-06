#!/usr/bin/env bash
set -euo pipefail

ROOT="/data8/cwkang/workspace/edge_attribution_patching"
PY="/home/cwkang/miniconda3/envs/EAP/bin/python"
LOG_DIR="${ROOT}/logs/20260504_default"
mkdir -p "${LOG_DIR}"
cd "${ROOT}"

declare -a GPUS=(0 1 2 3)
declare -a DATASETS=(imagenet officehome imagenet officehome)
declare -a MODELS=(vit_tiny_patch16_224 vit_tiny_patch16_224 deit_tiny_patch16_224 deit_tiny_patch16_224)

echo "[$(date '+%F %T')] launching ${#DATASETS[@]} EAP_IG_vit default jobs"

for i in "${!DATASETS[@]}"; do
  gpu="${GPUS[$i]}"
  dataset="${DATASETS[$i]}"
  model="${MODELS[$i]}"
  log="${LOG_DIR}/EAP_IG_vit_${dataset}_${model}_gpu${gpu}.log"
  echo "[$(date '+%F %T')] gpu=${gpu} dataset=${dataset} model=${model} log=${log}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export OMP_NUM_THREADS=1
    stdbuf -oL -eL "${PY}" -u src/EAP_IG_vit.py \
      --dataset_name "${dataset}" \
      --model_name "${model}"
  ) > "${log}" 2>&1 &
done

wait
echo "[$(date '+%F %T')] all EAP_IG_vit default jobs finished"
