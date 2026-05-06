import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch as t
from tqdm.auto import tqdm

from verify_cpt_subset_accuracy import (
    CPT_ROOT,
    DATA_ROOT,
    LANGUAGE_SPECS,
    REPO_ROOT,
    custom_load_tl_model_language,
    get_stopword_ids,
    language_topk,
    read_cpt_indices,
)


def make_corrupt_tokens(tokenizer, prompt: str, num_noise_sample: int, rand_seed: int):
    prng = np.random.RandomState(rand_seed)
    enc_prompt = tokenizer.encode(prompt)
    available_tokens = list(set(range(tokenizer.vocab_size)) - set(enc_prompt))
    selected_tokens = prng.choice(
        available_tokens,
        size=num_noise_sample * len(enc_prompt),
    )
    return enc_prompt, selected_tokens.reshape(num_noise_sample, -1).tolist()


@t.inference_mode()
def mean_corrupt_logits(model, corrupt_tokens: list[list[int]], device: t.device, batch_size: int):
    logits_sum = None
    total = 0
    for start in range(0, len(corrupt_tokens), batch_size):
        batch_tokens = corrupt_tokens[start : start + batch_size]
        batch = t.tensor(batch_tokens, dtype=t.long, device=device)
        logits = model(batch)
        if isinstance(logits, tuple):
            logits = logits[0]
        last_logits = logits[:, -1, :].float()
        curr_sum = last_logits.sum(dim=0)
        logits_sum = curr_sum if logits_sum is None else logits_sum + curr_sum
        total += last_logits.shape[0]
    return logits_sum / total


def prediction_from_logits(logits: t.Tensor, tokenizer):
    token_id = int(t.argmax(logits).detach().cpu())
    return token_id, tokenizer.decode(token_id)


@t.inference_mode()
def verify_spec(
    name: str,
    dataset_name: str,
    model_name: str,
    indices: list[int],
    device: t.device,
    num_noise_sample: int,
    rand_seed: int,
    batch_size: int,
):
    model = custom_load_tl_model_language(model_name, device)
    stopword_ids = get_stopword_ids(model.tokenizer)
    with (DATA_ROOT / f"{dataset_name}.json").open() as fin:
        dataset = json.load(fin)

    invalid_indices = [idx for idx in indices if idx < 0 or idx >= len(dataset)]
    valid_indices = [idx for idx in indices if idx not in invalid_indices]

    changed_filtered = 0
    unchanged_filtered = []
    changed_raw = 0
    unchanged_raw = []

    for idx in tqdm(valid_indices, desc=name):
        line = dataset[idx]
        prompt, label = line["prompt"], line["attribute"]

        clean_inp = model.tokenizer.encode(prompt, return_tensors="pt").to(device)
        clean_logits = model(clean_inp)[0][-1].float()
        clean_filtered_logits = clean_logits.clone()
        clean_filtered_logits[stopword_ids] *= 0
        clean_token_id, clean_token = prediction_from_logits(clean_filtered_logits, model.tokenizer)

        _, corrupt_tokens = make_corrupt_tokens(
            model.tokenizer,
            prompt,
            num_noise_sample=num_noise_sample,
            rand_seed=rand_seed,
        )
        avg_logits = mean_corrupt_logits(model, corrupt_tokens, device, batch_size)

        raw_token_id, raw_token = prediction_from_logits(avg_logits, model.tokenizer)
        filtered_logits = avg_logits.clone()
        filtered_logits[stopword_ids] *= 0
        filtered_token_id, filtered_token = prediction_from_logits(filtered_logits, model.tokenizer)

        raw_is_label = label in raw_token
        filtered_is_label = label in filtered_token
        changed_raw += int(not raw_is_label)
        changed_filtered += int(not filtered_is_label)

        case = {
            "idx": idx,
            "label": label,
            "prompt": prompt,
            "clean_top1_token_id": clean_token_id,
            "clean_top1_token": clean_token,
            "raw_avg_top1_token_id": raw_token_id,
            "raw_avg_top1_token": raw_token,
            "filtered_avg_top1_token_id": filtered_token_id,
            "filtered_avg_top1_token": filtered_token,
            "raw_top5": language_topk(avg_logits, model.tokenizer),
            "filtered_top5": language_topk(filtered_logits, model.tokenizer),
        }
        if raw_is_label:
            unchanged_raw.append(case)
        if filtered_is_label:
            unchanged_filtered.append(case)

    valid_total = len(valid_indices)
    return {
        "domain": "language",
        "name": name,
        "dataset": dataset_name,
        "model": model_name,
        "num_cpt_indices": len(indices),
        "num_valid_indices": valid_total,
        "num_invalid_indices": len(invalid_indices),
        "num_noise_sample": num_noise_sample,
        "rand_seed": rand_seed,
        "batch_size": batch_size,
        "changed_filtered": changed_filtered,
        "unchanged_filtered": len(unchanged_filtered),
        "changed_filtered_rate": changed_filtered / valid_total if valid_total else None,
        "changed_raw": changed_raw,
        "unchanged_raw": len(unchanged_raw),
        "changed_raw_rate": changed_raw / valid_total if valid_total else None,
        "invalid_indices": invalid_indices,
        "unchanged_filtered_cases": unchanged_filtered,
        "unchanged_raw_cases": unchanged_raw,
    }


def write_reports(results: list[dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "token_resampling_corruption_report.json"
    csv_path = output_dir / "token_resampling_corruption_report.csv"
    unchanged_path = output_dir / "token_resampling_unchanged_cases.json"
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
        "batch_size",
        "changed_filtered",
        "unchanged_filtered",
        "changed_filtered_rate",
        "changed_raw",
        "unchanged_raw",
        "changed_raw_rate",
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
            "unchanged_filtered_cases": row["unchanged_filtered_cases"],
            "unchanged_raw_cases": row["unchanged_raw_cases"],
        }
        for row in results
        if row["unchanged_filtered"] or row["unchanged_raw"]
    ]
    unchanged_path.write_text(json.dumps(unchanged, indent=2, ensure_ascii=False))
    return json_path, csv_path, unchanged_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional language spec names without .txt, e.g. lama_trex_gpt2-xs.",
    )
    parser.add_argument("--num-noise-sample", type=int, default=100)
    parser.add_argument("--rand-seed", type=int, default=0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Corrupt-input forward batch size. Defaults to 10 for pythia-1b and 100 otherwise.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "reports")
    return parser.parse_args()


def main():
    args = parse_args()
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    specs = list(LANGUAGE_SPECS.items())
    if args.only:
        only = set(args.only)
        specs = [(name, spec) for name, spec in specs if name in only]

    results = []
    for name, (dataset_name, model_name) in specs:
        cpt_file = CPT_ROOT / "language" / f"{name}.txt"
        indices = read_cpt_indices(cpt_file)
        batch_size = args.batch_size
        if batch_size is None:
            batch_size = 10 if "pythia-1b" in model_name else 100

        print(f"\nVerifying language/{name}: {len(indices)} CPT indices")
        result = verify_spec(
            name=name,
            dataset_name=dataset_name,
            model_name=model_name,
            indices=indices,
            device=device,
            num_noise_sample=args.num_noise_sample,
            rand_seed=args.rand_seed,
            batch_size=batch_size,
        )
        results.append(result)
        print(
            f"{name}: filtered changed={result['changed_filtered']}/"
            f"{result['num_valid_indices']} "
            f"unchanged={result['unchanged_filtered']}; raw changed="
            f"{result['changed_raw']}/{result['num_valid_indices']} "
            f"unchanged={result['unchanged_raw']}"
        )

    json_path, csv_path, unchanged_path = write_reports(results, args.output_dir)
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {unchanged_path}")


if __name__ == "__main__":
    main()
