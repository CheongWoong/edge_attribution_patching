import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch as t
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from verify_cpt_subset_accuracy import (
    CPT_ROOT,
    REPO_ROOT,
    VISION_SPECS,
    read_cpt_indices,
)


VIT_MAIN = REPO_ROOT / "vit_main"


def load_vision_helpers():
    sys.path.insert(0, str(VIT_MAIN))
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from lib.utils import get_data, get_model
    from util import custom_load_tl_model_vision

    return get_data, get_model, custom_load_tl_model_vision


def make_corrupted_images(dataset, num_noise_sample: int, rand_seed: int, patch_size: int):
    t.manual_seed(rand_seed)
    np.random.seed(rand_seed)
    random.seed(rand_seed)

    dataloader_for_corrupt = DataLoader(dataset, batch_size=num_noise_sample, shuffle=True)
    batch_images, _ = next(iter(dataloader_for_corrupt))
    batch_images = batch_images.cpu()

    batch_size, channels, height, width = batch_images.shape
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    total_patches = num_patches_h * num_patches_w

    patches = batch_images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(batch_size, channels, -1, patch_size, patch_size)
    shuffle_indices = t.randperm(batch_size * total_patches)
    corrupted_images = t.zeros_like(batch_images)

    for image_idx in range(batch_size):
        for patch_idx in range(total_patches):
            shuffle_idx = shuffle_indices[image_idx * total_patches + patch_idx]
            src_batch_idx = shuffle_idx // total_patches
            src_patch_idx = shuffle_idx % total_patches
            h_idx = (patch_idx // num_patches_w) * patch_size
            w_idx = (patch_idx % num_patches_w) * patch_size
            corrupted_images[
                image_idx,
                :,
                h_idx : h_idx + patch_size,
                w_idx : w_idx + patch_size,
            ] = patches[src_batch_idx, :, src_patch_idx]
    return corrupted_images


@t.inference_mode()
def mean_corrupt_logits(model, corrupted_images: t.Tensor, device: t.device, batch_size: int):
    logits_sum = None
    total = 0
    for start in range(0, len(corrupted_images), batch_size):
        batch = corrupted_images[start : start + batch_size].to(device)
        logits = model(batch)
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits.float()
        curr_sum = logits.sum(dim=0)
        logits_sum = curr_sum if logits_sum is None else logits_sum + curr_sum
        total += logits.shape[0]
    return logits_sum / total


def topk(logits: t.Tensor, k: int = 5):
    probs = t.softmax(logits.float(), dim=-1)
    top_probs, top_class_ids = t.topk(probs, k=min(k, logits.shape[-1]))
    top_logits = logits[top_class_ids]
    return [
        {
            "rank": rank,
            "class_id": int(class_id),
            "prob": float(prob),
            "logit": float(logit),
        }
        for rank, (class_id, prob, logit) in enumerate(
            zip(top_class_ids.detach().cpu(), top_probs.detach().cpu(), top_logits.detach().cpu()),
            start=1,
        )
    ]


@t.inference_mode()
def verify_spec(
    name: str,
    dataset_name: str,
    model_name: str,
    indices: list[int],
    device: t.device,
    num_noise_sample: int,
    rand_seed: int,
    patch_size: int,
    batch_size: int,
):
    get_data, get_model, custom_load_tl_model_vision = load_vision_helpers()
    dataset, num_classes = get_data(dataset_name)
    invalid_indices = [idx for idx in indices if idx < 0 or idx >= len(dataset)]
    valid_indices = [idx for idx in indices if idx not in invalid_indices]

    mt = get_model(model_name=model_name, num_classes=num_classes, dataset_name=dataset_name)
    mt.model = mt.model.cpu()
    old_state_dict = mt.model.state_dict()
    new_head_state_dict = {
        "head.weight": old_state_dict["model.head.weight"],
        "head.bias": old_state_dict["model.head.bias"],
    }
    model = custom_load_tl_model_vision(
        model_name, dataset_name, new_head_state_dict, num_classes, device
    )

    changed = 0
    unchanged_cases = []
    corrupted_images = make_corrupted_images(
        dataset=dataset,
        num_noise_sample=num_noise_sample,
        rand_seed=rand_seed,
        patch_size=patch_size,
    )
    avg_logits = mean_corrupt_logits(model, corrupted_images, device, batch_size)
    avg_pred = int(t.argmax(avg_logits).detach().cpu())
    avg_probs = t.softmax(avg_logits.float(), dim=-1)
    avg_top5 = topk(avg_logits)

    for idx in tqdm(valid_indices, desc=name):
        image, label = dataset[idx]
        label = int(label)
        clean_logits = model(image.unsqueeze(0).to(device))
        if isinstance(clean_logits, tuple):
            clean_logits = clean_logits[0]
        clean_pred = int(t.argmax(clean_logits[0]).detach().cpu())

        is_changed = avg_pred != label
        changed += int(is_changed)
        if not is_changed:
            unchanged_cases.append(
                {
                    "idx": idx,
                    "label": label,
                    "clean_pred": clean_pred,
                    "avg_corrupt_pred": avg_pred,
                    "label_prob": float(avg_probs[label].detach().cpu()),
                    "label_logit": float(avg_logits[label].detach().cpu()),
                    "top5": avg_top5,
                }
            )

    valid_total = len(valid_indices)
    return {
        "domain": "vision",
        "name": name,
        "dataset": dataset_name,
        "model": model_name,
        "num_cpt_indices": len(indices),
        "num_valid_indices": valid_total,
        "num_invalid_indices": len(invalid_indices),
        "num_noise_sample": num_noise_sample,
        "rand_seed": rand_seed,
        "patch_size": patch_size,
        "batch_size": batch_size,
        "changed": changed,
        "unchanged": len(unchanged_cases),
        "changed_rate": changed / valid_total if valid_total else None,
        "avg_corrupt_pred": avg_pred,
        "avg_corrupt_top5": avg_top5,
        "invalid_indices": invalid_indices,
        "unchanged_cases": unchanged_cases,
    }


def write_reports(results: list[dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "vit_patch_corruption_report.json"
    csv_path = output_dir / "vit_patch_corruption_report.csv"
    unchanged_path = output_dir / "vit_patch_corruption_unchanged_cases.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    fields = [
        "domain",
        "name",
        "dataset",
        "model",
        "num_cpt_indices",
        "num_valid_indices",
        "num_invalid_indices",
        "num_noise_sample",
        "rand_seed",
        "patch_size",
        "batch_size",
        "changed",
        "unchanged",
        "changed_rate",
    ]
    with csv_path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row[field] for field in fields})

    unchanged = [
        {
            "name": row["name"],
            "dataset": row["dataset"],
            "model": row["model"],
            "unchanged_cases": row["unchanged_cases"],
        }
        for row in results
        if row["unchanged"]
    ]
    unchanged_path.write_text(json.dumps(unchanged, indent=2, ensure_ascii=False))
    return json_path, csv_path, unchanged_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional vision spec names without .txt, e.g. imagenet_vit_tiny.",
    )
    parser.add_argument("--num-noise-sample", type=int, default=100)
    parser.add_argument("--rand-seed", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "reports")
    return parser.parse_args()


def main():
    args = parse_args()
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    specs = list(VISION_SPECS.items())
    if args.only:
        only = set(args.only)
        specs = [(name, spec) for name, spec in specs if name in only]

    results = []
    for name, (dataset_name, model_name) in specs:
        cpt_file = CPT_ROOT / "vision" / f"{name}.txt"
        indices = read_cpt_indices(cpt_file)
        print(f"\nVerifying vision/{name}: {len(indices)} CPT indices")
        result = verify_spec(
            name=name,
            dataset_name=dataset_name,
            model_name=model_name,
            indices=indices,
            device=device,
            num_noise_sample=args.num_noise_sample,
            rand_seed=args.rand_seed,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
        )
        results.append(result)
        print(
            f"{name}: changed={result['changed']}/{result['num_valid_indices']} "
            f"unchanged={result['unchanged']}"
        )

    json_path, csv_path, unchanged_path = write_reports(results, args.output_dir)
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {unchanged_path}")


if __name__ == "__main__":
    main()
