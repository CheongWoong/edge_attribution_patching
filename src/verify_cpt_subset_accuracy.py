import argparse
import csv
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch as t
import transformer_lens as tl
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
VIT_MAIN = REPO_ROOT / "vit_main"
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

VISION_SPECS = {
    "imagenet_vit_tiny": ("imagenet", "vit_tiny_patch16_224"),
    "imagenet_deit_tiny": ("imagenet", "deit_tiny_patch16_224"),
    "officehome_vit_tiny": ("officehome", "vit_tiny_patch16_224"),
    "officehome_deit_tiny": ("officehome", "deit_tiny_patch16_224"),
}


def read_cpt_indices(path: Path) -> list[int]:
    text = path.read_text().strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def custom_load_tl_model_language(model_name: str, device: t.device):
    try:
        from auto_circuit.experiment_utils import load_tl_model

        if model_name == "openai-community/gpt2":
            model = load_tl_model("gpt2", device)
        else:
            model = load_tl_model(model_name, device)
    except Exception:
        if "gpt2-xs" not in model_name:
            raise

        cfg = tl.HookedTransformerConfig(
            d_model=384,
            n_layers=6,
            n_heads=6,
            d_head=64,
            d_mlp=4 * 384,
            n_ctx=256,
            d_vocab=50257,
            act_fn="gelu",
            normalization_type="LN",
            seed=42,
        )
        model = tl.HookedTransformer(cfg)
        hf_model = AutoModelForCausalLM.from_pretrained(model_name)
        hf_state_dict = hf_model.state_dict()
        new_state_dict = OrderedDict()
        for name, param in hf_state_dict.items():
            if name == "transformer.wte.weight":
                new_state_dict["embed.W_E"] = param
            elif name == "transformer.wpe.weight":
                new_state_dict["pos_embed.W_pos"] = param
            elif name.startswith("transformer.h."):
                parts = name.split(".")
                layer = int(parts[2])
                subname = ".".join(parts[3:])
                if subname == "attn.c_attn.weight":
                    weight = param.T.reshape(3, cfg.d_model, cfg.d_model)
                    for proj, idx in zip(["Q", "K", "V"], [0, 1, 2]):
                        new_state_dict[f"blocks.{layer}.attn.W_{proj}"] = (
                            weight[idx]
                            .reshape(cfg.n_heads, cfg.d_head, cfg.d_model)
                            .permute(0, 2, 1)
                            .contiguous()
                        )
                elif subname == "attn.c_attn.bias":
                    bias = param.reshape(3, cfg.d_model)
                    for proj, idx in zip(["Q", "K", "V"], [0, 1, 2]):
                        new_state_dict[f"blocks.{layer}.attn.b_{proj}"] = (
                            bias[idx].reshape(cfg.n_heads, cfg.d_head).contiguous()
                        )
                elif subname == "attn.c_proj.weight":
                    new_state_dict[f"blocks.{layer}.attn.W_O"] = (
                        param.T.reshape(cfg.d_model, cfg.n_heads, cfg.d_head)
                        .permute(1, 2, 0)
                        .contiguous()
                    )
                elif subname == "attn.c_proj.bias":
                    new_state_dict[f"blocks.{layer}.attn.b_O"] = param
                elif subname == "ln_1.weight":
                    new_state_dict[f"blocks.{layer}.ln1.w"] = param
                elif subname == "ln_1.bias":
                    new_state_dict[f"blocks.{layer}.ln1.b"] = param
                elif subname == "ln_2.weight":
                    new_state_dict[f"blocks.{layer}.ln2.w"] = param
                elif subname == "ln_2.bias":
                    new_state_dict[f"blocks.{layer}.ln2.b"] = param
                elif subname == "mlp.c_fc.weight":
                    new_state_dict[f"blocks.{layer}.mlp.W_in"] = param
                elif subname == "mlp.c_fc.bias":
                    new_state_dict[f"blocks.{layer}.mlp.b_in"] = param
                elif subname == "mlp.c_proj.weight":
                    new_state_dict[f"blocks.{layer}.mlp.W_out"] = param
                elif subname == "mlp.c_proj.bias":
                    new_state_dict[f"blocks.{layer}.mlp.b_out"] = param
                else:
                    raise ValueError(f"Unexpected key in state dict: {name}")
            elif name == "transformer.ln_f.weight":
                new_state_dict["ln_final.w"] = param
            elif name == "transformer.ln_f.bias":
                new_state_dict["ln_final.b"] = param
            elif name == "lm_head.weight":
                new_state_dict["unembed.W_U"] = param.T
            else:
                raise ValueError(f"Unexpected key in state dict: {name}")
        model.load_state_dict(new_state_dict, strict=False)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.tokenizer = tokenizer
    model.cfg.use_attn_result = True
    model.cfg.use_attn_in = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.use_split_qkv_input = False
    model.cfg.tokenizer_prepends_bos = False
    model.cfg.default_prepend_bos = False
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


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
def verify_language(name: str, dataset_name: str, model_name: str, indices: list[int], device: t.device):
    model = custom_load_tl_model_language(model_name, device)
    stopword_ids = get_stopword_ids(model.tokenizer)
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
        inp = model.tokenizer.encode(prompt, return_tensors="pt").to(device)
        out = model(inp)[0][-1]
        out[stopword_ids] *= 0
        pred_token_id = t.argmax(out)
        pred_token = model.tokenizer.decode(pred_token_id)
        is_correct = int(label in pred_token)
        correct += is_correct
        if not is_correct:
            incorrect.append(
                {
                    "idx": idx,
                    "label": label,
                    "label_token_ids_space": model.tokenizer.encode(
                        f" {label}", add_special_tokens=False
                    ),
                    "label_token_ids_no_space": model.tokenizer.encode(
                        label, add_special_tokens=False
                    ),
                    "pred_token": pred_token,
                    "top5": language_topk(out, model.tokenizer),
                    "prompt": prompt,
                }
            )

    valid_total = len(indices) - len(invalid_indices)
    return {
        "domain": "language",
        "name": name,
        "dataset": dataset_name,
        "model": model_name,
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


def load_vision_helpers():
    sys.path.insert(0, str(VIT_MAIN))
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from lib.utils import get_data, get_model
    from util import custom_load_tl_model_vision

    return get_data, get_model, custom_load_tl_model_vision


@t.inference_mode()
def verify_vision(
    name: str,
    dataset_name: str,
    model_name: str,
    indices: list[int],
    device: t.device,
    batch_size: int,
):
    get_data, get_model, custom_load_tl_model_vision = load_vision_helpers()
    dataset, num_classes = get_data(dataset_name)
    invalid_indices = [idx for idx in indices if idx < 0 or idx >= len(dataset)]

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

    correct = 0
    incorrect = []
    valid_indices = [idx for idx in indices if idx not in invalid_indices]
    for start in tqdm(range(0, len(valid_indices), batch_size), desc=name):
        batch_indices = valid_indices[start : start + batch_size]
        images = []
        labels = []
        for idx in batch_indices:
            image, label = dataset[idx]
            images.append(image)
            labels.append(int(label))
        image_input = t.stack(images).to(device)
        outputs = model(image_input)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        probs = t.softmax(outputs.float(), dim=-1)
        top_probs, top_class_ids = t.topk(probs, k=min(5, outputs.shape[-1]), dim=-1)
        top_logits = t.gather(outputs, dim=-1, index=top_class_ids)
        preds = top_class_ids[:, 0].detach().cpu().tolist()
        for row, idx, label, pred in zip(range(len(batch_indices)), batch_indices, labels, preds):
            is_correct = int(pred == label)
            correct += is_correct
            if not is_correct:
                incorrect.append(
                    {
                        "idx": idx,
                        "label": label,
                        "pred": pred,
                        "label_prob": float(probs[row, label].detach().cpu()),
                        "label_logit": float(outputs[row, label].detach().cpu()),
                        "top5": [
                            {
                                "rank": rank,
                                "class_id": int(class_id),
                                "prob": float(prob),
                                "logit": float(logit),
                            }
                            for rank, (class_id, prob, logit) in enumerate(
                                zip(
                                    top_class_ids[row].detach().cpu(),
                                    top_probs[row].detach().cpu(),
                                    top_logits[row].detach().cpu(),
                                ),
                                start=1,
                            )
                        ],
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
        "num_correct": correct,
        "num_incorrect": len(incorrect),
        "accuracy": correct / valid_total if valid_total else None,
        "invalid_indices": invalid_indices,
        "incorrect_cases": incorrect,
        "incorrect_examples": incorrect[:20],
    }


def write_reports(results: list[dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "cpt_subset_accuracy_report.json"
    csv_path = output_dir / "cpt_subset_accuracy_report.csv"
    top5_json_path = output_dir / "cpt_subset_incorrect_top5.json"
    top5_csv_path = output_dir / "cpt_subset_incorrect_top5.csv"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    incorrect_results = [
        {
            "domain": row["domain"],
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
        "name",
        "dataset",
        "model",
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
                pred = case.get("pred", case.get("pred_token"))
                for top in case["top5"]:
                    writer.writerow(
                        {
                            "domain": result["domain"],
                            "name": result["name"],
                            "dataset": result["dataset"],
                            "model": result["model"],
                            "idx": case["idx"],
                            "label": case["label"],
                            "pred": pred,
                            "rank": top["rank"],
                            "top_id": top.get("class_id", top.get("token_id")),
                            "top_text": top.get("token", ""),
                            "prob": top["prob"],
                            "logit": top["logit"],
                        }
                    )
    return json_path, csv_path, top5_json_path, top5_csv_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain",
        choices=["all", "language", "vision"],
        default="all",
        help="Which CPT index directory to verify.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional spec names without .txt, e.g. lama_trex_gpt2-xs imagenet_vit_tiny.",
    )
    parser.add_argument("--vision-batch-size", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "reports")
    return parser.parse_args()


def main():
    args = parse_args()
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    results = []

    specs = []
    if args.domain in ("all", "language"):
        specs.extend(("language", name, spec) for name, spec in LANGUAGE_SPECS.items())
    if args.domain in ("all", "vision"):
        specs.extend(("vision", name, spec) for name, spec in VISION_SPECS.items())
    if args.only:
        only = set(args.only)
        specs = [spec for spec in specs if spec[1] in only]

    for domain, name, (dataset_name, model_name) in specs:
        cpt_file = CPT_ROOT / domain / f"{name}.txt"
        indices = read_cpt_indices(cpt_file)
        print(f"\nVerifying {domain}/{name}: {len(indices)} CPT indices")
        if domain == "language":
            result = verify_language(name, dataset_name, model_name, indices, device)
        else:
            result = verify_vision(
                name, dataset_name, model_name, indices, device, args.vision_batch_size
            )
        results.append(result)
        acc = result["accuracy"]
        acc_text = "n/a" if acc is None else f"{acc * 100:.4f}%"
        print(
            f"{name}: {result['num_correct']}/{result['num_valid_indices']} correct "
            f"({acc_text}), invalid={result['num_invalid_indices']}"
        )

    json_path, csv_path, top5_json_path, top5_csv_path = write_reports(results, args.output_dir)
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {top5_json_path}")
    print(f"Wrote {top5_csv_path}")


if __name__ == "__main__":
    main()
