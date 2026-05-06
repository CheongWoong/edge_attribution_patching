#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cwkang/workspace/edge_attribution_patching"
PY="${PY:-python}"

cd "${ROOT}"

scripts=(
  "EAP_edge_to_path_conversion.py"
  "EAP_IG_edge_to_path_conversion.py"
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
            echo "[$(date '+%F %T')] ${script} dataset=${dataset} model=${model} score_function=logprob"
            OMP_NUM_THREADS=1 "${PY}" -u "src/${script}" \
                --score_function logprob \
                --dataset_name "${dataset}" \
                --model_name "${model}"
    done
  done
done

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for script in "${scripts[@]}"; do
            echo "[$(date '+%F %T')] ${script} dataset=${dataset} model=${model} score_function=logprob"
            OMP_NUM_THREADS=1 "${PY}" -u "src/${script}" \
                --score_function logprob \
                --dataset_name "${dataset}" \
                --model_name "${model}" \
                --filter_connected
    done
  done
done

scripts=(
  "EAP_vit_edge_to_path_conversion.py"
  "EAP_IG_vit_edge_to_path_conversion.py"
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
            echo "[$(date '+%F %T')] ${script} dataset=${dataset} model=${model} score_function=logprob"
            OMP_NUM_THREADS=1 "${PY}" -u "src/${script}" \
                --score_function logprob \
                --dataset_name "${dataset}" \
                --model_name "${model}"
    done
  done
done

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for script in "${scripts[@]}"; do
            echo "[$(date '+%F %T')] ${script} dataset=${dataset} model=${model} score_function=logprob"
            OMP_NUM_THREADS=1 "${PY}" -u "src/${script}" \
                --score_function logprob \
                --dataset_name "${dataset}" \
                --model_name "${model}" \
                --filter_connected
    done
  done
done
