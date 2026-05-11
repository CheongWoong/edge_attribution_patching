#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cwkang/workspace/edge_attribution_patching"
PY="${PY:-python}"

cd "${ROOT}"



scripts=(
  "EAP_vit.py"
  "EAP_IG_vit.py"
)

datasets=(
  "imagenet"
  "officehome"
)

models=(
  "vit_tiny_patch16_224"
  "deit_tiny_patch16_224"
)

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for script in "${scripts[@]}"; do
            echo "[$(date '+%F %T')] ${script} dataset=${dataset} model=${model} score_function=logit_diff"
            OMP_NUM_THREADS=1 "${PY}" -u "src/${script}" \
                --score_function logit_diff \
                --dataset_name "${dataset}" \
                --model_name "${model}"
    done
  done
done
