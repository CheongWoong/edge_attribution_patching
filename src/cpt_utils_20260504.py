import os
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CPT_ROOT = REPO_ROOT / "path_forward_idx_CPT"


LANGUAGE_CPT_NAMES = {
    ("known_1000", "gpt2-xs"): "rome_factual_1000_gpt2-xs",
    ("known_1000", "pythia-14m"): "rome_factual_1000_pythia-14m",
    ("known_1000", "pythia-1b"): "rome_factual_1000_pythia-1b",
    ("lama_trex", "gpt2-xs"): "lama_trex_gpt2-xs",
    ("lama_trex", "pythia-14m"): "lama_trex_pythia-14m",
    ("lama_trex", "pythia-1b"): "lama_trex_pythia-1b",
}

VISION_CPT_NAMES = {
    ("imagenet", "vit_tiny_patch16_224"): "imagenet_vit_tiny",
    ("imagenet", "deit_tiny_patch16_224"): "imagenet_deit_tiny",
    ("officehome", "vit_tiny_patch16_224"): "officehome_vit_tiny",
    ("officehome", "deit_tiny_patch16_224"): "officehome_deit_tiny",
}


def read_cpt_indices(path: Path) -> list[int]:
    text = path.read_text().strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def model_short_name(model_name: str) -> str:
    return model_name.split("/")[-1]


def language_cpt_path(dataset_name: str, model_name: str) -> Path:
    short_name = model_short_name(model_name)
    key = (dataset_name, short_name)
    if key not in LANGUAGE_CPT_NAMES:
        raise ValueError(f"No language CPT subset is defined for {dataset_name}/{short_name}")
    return CPT_ROOT / "language" / f"{LANGUAGE_CPT_NAMES[key]}.txt"


def vision_cpt_path(dataset_name: str, model_name: str) -> Path:
    key = (dataset_name, model_name)
    if key not in VISION_CPT_NAMES:
        raise ValueError(f"No vision CPT subset is defined for {dataset_name}/{model_name}")
    return CPT_ROOT / "vision" / f"{VISION_CPT_NAMES[key]}.txt"


def raw_result_path(out_path: str | Path, idx: int) -> str:
    idx_4 = "%04d" % idx
    idx_6 = "%06d" % idx
    return os.path.join(out_path, "results", f"R{idx_4}", f"raw_C{idx_6}.npy")


def is_valid_raw_result(path: str | Path) -> bool:
    try:
        result = np.load(path, allow_pickle=True).item()
    except Exception:
        return False
    if not isinstance(result, dict) or not result:
        return False
    for value in result.values():
        arr = np.asarray(value)
        if arr.size == 0:
            return False
        if not np.isfinite(arr).all():
            return False
    return True


def pending_cpt_indices(indices: list[int], out_path: str | Path) -> tuple[list[int], list[int]]:
    valid = []
    pending = []
    for idx in indices:
        if is_valid_raw_result(raw_result_path(out_path, idx)):
            valid.append(idx)
        else:
            pending.append(idx)
    return pending, valid
