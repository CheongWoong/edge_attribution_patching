my_path = "vit_main"

import os
import argparse

import torch as t
import numpy as np

from auto_circuit.utils.graph_utils import patchable_model

from pathlib import Path
import json
from collections import defaultdict
from tqdm.auto import tqdm

try:
    from .util import find_knowns_ids
    from .util import custom_load_tl_model_vision as custom_load_tl_model
    from .cpt_utils_20260504 import read_cpt_indices, vision_cpt_path
    from .edge_conversion_utils_20260506 import (
        DEFAULT_PRUNE_RATIOS,
        add_summary_record,
        conversion_out_path,
        path_circuit_and_metadata,
        write_edge_ratio_summaries,
        write_sample_jsons,
    )
except ImportError:
    from util import find_knowns_ids
    from util import custom_load_tl_model_vision as custom_load_tl_model
    from cpt_utils_20260504 import read_cpt_indices, vision_cpt_path
    from edge_conversion_utils_20260506 import (
        DEFAULT_PRUNE_RATIOS,
        add_summary_record,
        conversion_out_path,
        path_circuit_and_metadata,
        write_edge_ratio_summaries,
        write_sample_jsons,
    )

import sys
vit_path = Path(my_path)
if vit_path not in sys.path:
    sys.path.insert(0, str(vit_path))
from lib.utils import get_model, get_data


parser = argparse.ArgumentParser(description='helloworld')
parser.add_argument("--dataset_name", type=str, required=True, choices=["imagenet", "officehome"])
parser.add_argument("--model_name", type=str, required=True, choices=["vit_tiny_patch16_224", "deit_tiny_patch16_224"])
parser.add_argument("--score_function", type=str, default="logit", choices=["logit", "logit_diff", "logprob"])
parser.add_argument("--prune_ratios", type=float, nargs="+", default=DEFAULT_PRUNE_RATIOS)
parser.add_argument("--filter_connected", action="store_true")
args = parser.parse_args()

# for dataset_name in ["imagenet", "officehome"]:
for dataset_name in [args.dataset_name]:
    # for model_name in ["vit_tiny_patch16_224", "deit_tiny_patch16_224"]:
    for model_name in [args.model_name]:
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
        # Load ImageNet dataset
        dataset, num_classes = get_data(dataset_name)
        class_ids = np.arange(num_classes)
        # print data spec
        print(f"Dataset: {dataset_name}")
        print(f"Number of classes: {num_classes}")
        print(f"Number of samples: {len(dataset)}")

        mt = get_model(model_name=model_name, num_classes=num_classes, dataset_name=dataset_name)
        mt.model = mt.model.cpu()
        old_state_dict = mt.model.state_dict()
        new_head_state_dict = {}
        new_head_state_dict["head.weight"] = old_state_dict["model.head.weight"]
        new_head_state_dict["head.bias"] = old_state_dict["model.head.bias"]
        model = custom_load_tl_model(model_name, dataset_name, new_head_state_dict, num_classes, device)

        out_path = os.path.join(f"jobs_EAP_{args.score_function}", dataset_name + "_" + model_name.split("/")[-1])

        try:
            model = patchable_model(
                    model,
                    factorized=True,
                    # slice_output="last_seq",
                    slice_output=None, # use first (cls) token only later
                    separate_qkv=False,
                    device=device,
                )
        except Exception as e:
            print("[Error]", e)

        def extract_node_info(node):
            if node == "Resid Start":
                node_type = "none"
                bidx = -1
                hidx = -1
            elif node == "Resid End":
                node_type = "none"
                bidx = model.cfg.n_layers
                hidx = -1
            elif node.startswith("A"):
                node_type = "Attention"
                bidx = int(node[1:].split(".")[0])
                hidx = int(node[1:].split(".")[1])
            elif node.startswith("MLP"):
                node_type = "MLP"
                bidx = int(node[4:])
                hidx = -1
            else:
                raise Exception
            return {"node_type": node_type, "bidx": bidx, "hidx": hidx}

        cpt_indices = read_cpt_indices(vision_cpt_path(dataset_name, model_name))
        cpt_indices = [idx for idx in cpt_indices if 0 <= idx < len(dataset)]
        class_idx_map = defaultdict(list)
        num_incorrect = 0
        for idx in tqdm(cpt_indices):
            image, label = dataset[idx]
            image_input = image.unsqueeze(0).to(device)
            output = model(image_input)[0]
            pred = t.argmax(output, dim=-1)
            if pred.item() != label:
                num_incorrect += 1

            class_idx_map[label].append(idx)
        print(
            f"CPT conversion indices: total={len(cpt_indices)}, "
            f"selected={sum(len(v) for v in class_idx_map.values())}, "
            f"num_incorrect={num_incorrect}"
        )

        edge_ratio_summaries = defaultdict(list)
        task_indices = []
        task_attribution_scores = None
        for class_id, indices in tqdm(class_idx_map.items()):
            class_attribution_scores = None

            class_indices = []
            for idx in indices:
                idx_4 = "%04d" % idx
                idx_6 = "%06d" % idx
                fin = os.path.join(out_path, "results", f"R{idx_4}", f"raw_C{idx_6}.npy")
                try:
                    attribution_scores = np.load(fin, allow_pickle=True).item()
                    class_indices.append(idx)
                    task_indices.append(idx)
                except Exception as e:
                    print(e)
                    continue

                for key in attribution_scores:
                    attribution_scores[key] = t.from_numpy(attribution_scores[key])

                if class_attribution_scores is None:
                    class_attribution_scores = attribution_scores
                else:
                    for key in attribution_scores:
                        class_attribution_scores[key] += attribution_scores[key]

                if task_attribution_scores is None:
                    task_attribution_scores = attribution_scores
                else:
                    for key in attribution_scores:
                        task_attribution_scores[key] += attribution_scores[key]

                for prune_ratio in args.prune_ratios:
                    causal_subsets, edge_metadata = path_circuit_and_metadata(
                        model, attribution_scores, prune_ratio, args.filter_connected, extract_node_info
                    )
                    edge_metadata.update({"level": "samplewise", "idx": idx})
                    out_path_samplewise = conversion_out_path(
                        out_path, "samplewise", prune_ratio, args.filter_connected
                    )
                    write_sample_jsons(out_path_samplewise, idx, causal_subsets, edge_metadata)
                    add_summary_record(edge_ratio_summaries, out_path_samplewise, edge_metadata)

            attribution_scores = class_attribution_scores
            if attribution_scores is not None:
                for prune_ratio in args.prune_ratios:
                    causal_subsets, edge_metadata = path_circuit_and_metadata(
                        model, attribution_scores, prune_ratio, args.filter_connected, extract_node_info
                    )
                    edge_metadata.update(
                        {"level": "classwise", "class_id": int(class_id), "num_samples": len(class_indices)}
                    )
                    out_path_classwise = conversion_out_path(
                        out_path, "classwise", prune_ratio, args.filter_connected
                    )
                    add_summary_record(edge_ratio_summaries, out_path_classwise, edge_metadata)
                    for idx in class_indices:
                        sample_metadata = edge_metadata.copy()
                        sample_metadata["idx"] = idx
                        write_sample_jsons(out_path_classwise, idx, causal_subsets, sample_metadata)

        attribution_scores = task_attribution_scores
        if attribution_scores is not None:
            for prune_ratio in args.prune_ratios:
                causal_subsets, edge_metadata = path_circuit_and_metadata(
                    model, attribution_scores, prune_ratio, args.filter_connected, extract_node_info
                )
                edge_metadata.update({"level": "taskwise", "num_samples": len(task_indices)})
                out_path_taskwise = conversion_out_path(
                    out_path, "taskwise", prune_ratio, args.filter_connected
                )
                add_summary_record(edge_ratio_summaries, out_path_taskwise, edge_metadata)
                for idx in task_indices:
                    sample_metadata = edge_metadata.copy()
                    sample_metadata["idx"] = idx
                    write_sample_jsons(out_path_taskwise, idx, causal_subsets, sample_metadata)

        write_edge_ratio_summaries(edge_ratio_summaries)
