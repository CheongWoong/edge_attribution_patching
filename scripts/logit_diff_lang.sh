#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cwkang/workspace/edge_attribution_patching"
PY="${PY:-python}"

cd "${ROOT}"

scripts=(
  "EAP.py"
  "EAP_IG.py"
)

datasets=(
  "known_1000"
  "lama_trex"
)

models=(
  "AlgorithmicResearchGroup/gpt2-xs"
  "EleutherAI/pythia-14m"
  "EleutherAI/pythia-1b"
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
