import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import torch as t
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT.parent / "main" / "data"
CPT_ROOT = REPO_ROOT / "path_forward_idx_CPT"

LANGUAGE_SPECS = {
    "rome_factual_1000_gpt2-xs": ("known_1000", "AlgorithmicResearchGroup/gpt2-xs"),
    "rome_factual_1000_pythia-14m": ("known_1000", "EleutherAI/pythia-14m"),
    "rome_factual_1000_pythia-1b": ("known_1000", "EleutherAI/pythia-1b"),
    "lama_trex_gpt2-xs": ("lama_trex", "AlgorithmicResearchGroup/gpt2-xs"),
    "lama_trex_pythia-14m": ("lama_trex", "EleutherAI/pythia-14m"),
    "lama_trex_pythia-1b": ("lama_trex", "EleutherAI/pythia-1b"),
}


def read_cpt_indices(path: Path) -> list[int]:
    text = path.read_text().strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def dtype_summary(model) -> dict:
    param_counts = Counter(str(param.dtype).replace("torch.", "") for param in model.parameters())
    buffer_counts = Counter(str(buf.dtype).replace("torch.", "") for buf in model.buffers())
    first_param = next(model.parameters(), None)
    first_buffer = next(model.buffers(), None)
    return {
        "parameter_dtype": str(first_param.dtype).replace("torch.", "") if first_param is not None else None,
        "buffer_dtype": str(first_buffer.dtype).replace("torch.", "") if first_buffer is not None else None,
        "parameter_dtype_counts": dict(sorted(param_counts.items())),
        "buffer_dtype_counts": dict(sorted(buffer_counts.items())),
    }


def load_hf_model(model_name: str, device: t.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer


def get_stopword_ids(tokenizer) -> list[int]:
    try:
        from nltk.corpus import stopwords

        stopword_list = stopwords.words("english")
    except LookupError:
        import nltk

        nltk.download("stopwords")
        from nltk.corpus import stopwords

        stopword_list = stopwords.words("english")

    stopword_ids = []
    for stopword in stopword_list:
        for text in (f" {stopword}", stopword):
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) == 1:
                stopword_ids.append(token_ids[0])
    return sorted(set(stopword_ids))


def language_topk(logits: t.Tensor, tokenizer, k: int = 5) -> list[dict]:
    probs = t.softmax(logits.float(), dim=-1)
    top_probs, top_token_ids = t.topk(probs, k=k)
    top_logits = logits[top_token_ids]
    return [
        {
            "rank": rank,
            "token_id": int(token_id),
            "token": tokenizer.decode(token_id),
            "prob": float(prob),
            "logit": float(logit),
        }
        for rank, (token_id, prob, logit) in enumerate(
            zip(top_token_ids.detach().cpu(), top_probs.detach().cpu(), top_logits.detach().cpu()),
            start=1,
        )
    ]


@t.inference_mode()
def verify_language_hf(
    name: str,
    dataset_name: str,
    model_name: str,
    indices: list[int],
    device: t.device,
):
    model, tokenizer = load_hf_model(model_name, device)
    model_dtype = dtype_summary(model)
    stopword_ids = get_stopword_ids(tokenizer)
    with (DATA_ROOT / f"{dataset_name}.json").open() as fin:
        dataset = json.load(fin)

    invalid_indices = [idx for idx in indices if idx < 0 or idx >= len(dataset)]
    correct = 0
    incorrect = []
    for idx in tqdm(indices, desc=name):
        if idx in invalid_indices:
            continue
        line = dataset[idx]
        prompt, label = line["prompt"], line["attribute"]
        inp = tokenizer.encode(prompt, return_tensors="pt").to(device)
        logits = model(inp).logits[0, -1]
        filtered_logits = logits.clone()
        filtered_logits[stopword_ids] *= 0
        pred_token_id = t.argmax(filtered_logits)
        pred_token = tokenizer.decode(pred_token_id)
        is_correct = int(label in pred_token)
        correct += is_correct
        if not is_correct:
            incorrect.append(
                {
                    "idx": idx,
                    "label": label,
                    "label_token_ids_space": tokenizer.encode(
                        f" {label}", add_special_tokens=False
                    ),
                    "label_token_ids_no_space": tokenizer.encode(
                        label, add_special_tokens=False
                    ),
                    "pred_token": pred_token,
                    "top5": language_topk(filtered_logits, tokenizer),
                    "prompt": prompt,
                }
            )

    valid_total = len(indices) - len(invalid_indices)
    result = {
        "domain": "language",
        "loader": "huggingface",
        "name": name,
        "dataset": dataset_name,
        "model": model_name,
        "model_dtype": model_dtype["parameter_dtype"],
        "model_dtype_summary": model_dtype,
        "num_cpt_indices": len(indices),
        "num_valid_indices": valid_total,
        "num_invalid_indices": len(invalid_indices),
        "num_correct": correct,
        "num_incorrect": len(incorrect),
        "accuracy": correct / valid_total if valid_total else None,
        "invalid_indices": invalid_indices,
        "incorrect_cases": incorrect,
        "incorrect_examples": incorrect[:20],
    }
    dtype_row = {
        "loader": "huggingface",
        "name": name,
        "dataset": dataset_name,
        "model": model_name,
        **model_dtype,
    }
    del model
    if device.type == "cuda":
        t.cuda.empty_cache()
    return result, dtype_row


def record_tl_dtype_rows(specs, device: t.device) -> list[dict]:
    from verify_cpt_subset_accuracy import custom_load_tl_model_language

    rows = []
    seen_models = {}
    for name, (dataset_name, model_name) in specs:
        if model_name not in seen_models:
            model = custom_load_tl_model_language(model_name, device)
            seen_models[model_name] = dtype_summary(model)
            del model
            if device.type == "cuda":
                t.cuda.empty_cache()
        rows.append(
            {
                "loader": "transformer_lens",
                "name": name,
                "dataset": dataset_name,
                "model": model_name,
                **seen_models[model_name],
            }
        )
    return rows


def write_dtype_reports(rows: list[dict], output_dir: Path, stem: str):
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    fields = [
        "loader",
        "name",
        "dataset",
        "model",
        "parameter_dtype",
        "buffer_dtype",
        "parameter_dtype_counts",
        "buffer_dtype_counts",
    ]
    with csv_path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "parameter_dtype_counts": json.dumps(row["parameter_dtype_counts"]),
                    "buffer_dtype_counts": json.dumps(row["buffer_dtype_counts"]),
                }
            )
    return json_path, csv_path


def write_reports(results: list[dict], hf_dtype_rows: list[dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "cpt_subset_accuracy_hf_report.json"
    csv_path = output_dir / "cpt_subset_accuracy_hf_report.csv"
    top5_json_path = output_dir / "cpt_subset_incorrect_top5_hf.json"
    top5_csv_path = output_dir / "cpt_subset_incorrect_top5_hf.csv"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    incorrect_results = [
        {
            "domain": row["domain"],
            "loader": row["loader"],
            "name": row["name"],
            "dataset": row["dataset"],
            "model": row["model"],
            "incorrect_cases": row.get("incorrect_cases", row.get("incorrect_examples", [])),
        }
        for row in results
        if row["num_incorrect"]
    ]
    top5_json_path.write_text(json.dumps(incorrect_results, indent=2, ensure_ascii=False))

    fields = [
        "domain",
        "loader",
        "name",
        "dataset",
        "model",
        "model_dtype",
        "num_cpt_indices",
        "num_valid_indices",
        "num_invalid_indices",
        "num_correct",
        "num_incorrect",
        "accuracy",
    ]
    with csv_path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row[field] for field in fields})

    top5_fields = [
        "domain",
        "loader",
        "name",
        "dataset",
        "model",
        "idx",
        "label",
        "pred",
        "rank",
        "top_id",
        "top_text",
        "prob",
        "logit",
    ]
    with top5_csv_path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=top5_fields)
        writer.writeheader()
        for result in incorrect_results:
            for case in result["incorrect_cases"]:
                for top in case["top5"]:
                    writer.writerow(
                        {
                            "domain": result["domain"],
                            "loader": result["loader"],
                            "name": result["name"],
                            "dataset": result["dataset"],
                            "model": result["model"],
                            "idx": case["idx"],
                            "label": case["label"],
                            "pred": case["pred_token"],
                            "rank": top["rank"],
                            "top_id": top["token_id"],
                            "top_text": top["token"],
                            "prob": top["prob"],
                            "logit": top["logit"],
                        }
                    )

    hf_dtype_json, hf_dtype_csv = write_dtype_reports(
        hf_dtype_rows, output_dir, "hf_model_dtype_report"
    )
    return json_path, csv_path, top5_json_path, top5_csv_path, hf_dtype_json, hf_dtype_csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional language spec names without .txt, e.g. lama_trex_gpt2-xs.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "reports")
    parser.add_argument(
        "--record-tl-dtypes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also load models via the existing TL loader and write dtype reports.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    specs = list(LANGUAGE_SPECS.items())
    if args.only:
        only = set(args.only)
        specs = [(name, spec) for name, spec in specs if name in only]

    results = []
    hf_dtype_rows = []
    for name, (dataset_name, model_name) in specs:
        cpt_file = CPT_ROOT / "language" / f"{name}.txt"
        indices = read_cpt_indices(cpt_file)
        print(f"\nVerifying language/{name} with Hugging Face: {len(indices)} CPT indices")
        result, dtype_row = verify_language_hf(name, dataset_name, model_name, indices, device)
        results.append(result)
        hf_dtype_rows.append(dtype_row)
        acc = result["accuracy"]
        acc_text = "n/a" if acc is None else f"{acc * 100:.4f}%"
        print(
            f"{name}: {result['num_correct']}/{result['num_valid_indices']} correct "
            f"({acc_text}), invalid={result['num_invalid_indices']}, dtype={result['model_dtype']}"
        )

    paths = write_reports(results, hf_dtype_rows, args.output_dir)
    for path in paths:
        print(f"Wrote {path}")

    if args.record_tl_dtypes:
        print("\nRecording dtype for existing TL loader")
        tl_rows = record_tl_dtype_rows(specs, device)
        tl_paths = write_dtype_reports(tl_rows, args.output_dir, "tl_model_dtype_report")
        for path in tl_paths:
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
