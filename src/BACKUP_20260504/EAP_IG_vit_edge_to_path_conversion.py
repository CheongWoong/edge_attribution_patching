my_path = "vit_main"

import os
import argparse

import torch as t
import numpy as np

from auto_circuit.utils.graph_utils import patchable_model
from auto_circuit.utils.tensor_ops import prune_scores_threshold

import networkx as nx
from pathlib import Path
import json
from collections import defaultdict
from tqdm.auto import tqdm

from .util import find_knowns_ids
from .util import custom_load_tl_model_vision as custom_load_tl_model

import sys
vit_path = Path(my_path)
if vit_path not in sys.path:
    sys.path.insert(0, str(vit_path))
from lib.utils import get_model, get_data


parser = argparse.ArgumentParser(description='helloworld')
parser.add_argument("--dataset_name", type=str, required=True, choices=["imagenet", "officehome"])
parser.add_argument("--model_name", type=str, required=True, choices=["vit_tiny_patch16_224", "deit_tiny_patch16_224"])
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

        out_path = os.path.join("jobs_EAP_IG", dataset_name + "_" + model_name.split("/")[-1])

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

        known_ids = find_knowns_ids(model_name, dataset_name)

        class_idx_map = defaultdict(list)
        for idx in tqdm(known_ids):
            _, label = dataset[idx]

            class_idx_map[label].append(idx)

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

                for prune_ratio in [0.25]:
                    threshold = prune_scores_threshold(attribution_scores, int(len(model.edges)*prune_ratio))
                    # threshold = t.cat([ps.flatten() for _, ps in attribution_scores.items()]).sort(descending=True).values[int(len(model.edges)*prune_ratio) - 1]

                    edges = {}
                    for edge in model.edges:
                        edges[edge.name] = edge.prune_score(attribution_scores).abs() >= threshold

                    ############################################################
                    ### Pruning disconnected nodes
                    G = nx.DiGraph()

                    for conn, active in edges.items():
                        if active:
                            src, dst = conn.split("->")
                            G.add_edge(src, dst)

                    reachable_from_input = set(nx.descendants(G, "Resid Start")) | {"Resid Start"}

                    reversed_G = G.reverse()
                    reachable_to_output = set(nx.descendants(reversed_G, "Resid End")) | {"Resid End"}

                    connected_nodes = reachable_from_input & reachable_to_output

                    pruned_G = G.subgraph(connected_nodes).copy()

                    pruned_edges = [(u, v) for u, v in pruned_G.edges]
                    ############################################################

                    ############################################################
                    ### Convert edges to paths
                    causal_subsets = defaultdict(set)
                    for u, v in pruned_edges:
                        u_info, v_info = extract_node_info(u), extract_node_info(v)
                        # Attention + MLP
                        if u_info["node_type"] == "Attention" and v_info["node_type"] == "MLP" and u_info["bidx"] == v_info["bidx"]:
                            causal_subsets[u_info["bidx"]].add(u_info["hidx"] + 2 + model.cfg.n_heads)
                        # Attention only
                        if u_info["node_type"] == "Attention" and u_info["bidx"] != v_info["bidx"]:
                            causal_subsets[u_info["bidx"]].add(u_info["hidx"] + 2)
                        # MLP only
                        if v_info["node_type"] == "MLP" and u_info["bidx"] != v_info["bidx"]:
                            causal_subsets[v_info["bidx"]].add(1)
                        # Residual only
                        if v_info["bidx"] - u_info["bidx"] > 1:
                            for bidx in range(u_info["bidx"] + 1, v_info["bidx"]):
                                causal_subsets[bidx].add(0)
                    
                    for bidx in range(model.cfg.n_layers):
                        causal_subsets[bidx] = [sorted(list(causal_subsets[bidx]))]
                    # print(causal_subsets)
                    ############################################################

                    out_path_samplewise = out_path + f"_samplewise_{prune_ratio}"
                    os.makedirs(os.path.join(out_path_samplewise, "results", f"R{idx_4}"), exist_ok=True)
                    with open(os.path.join(out_path_samplewise, "results", f"R{idx_4}", f"C{idx_6}.json"), "w") as fout:
                        json.dump(causal_subsets, fout, indent=2)

            attribution_scores = class_attribution_scores
            if attribution_scores is not None:
                for prune_ratio in [0.25]:
                    threshold = prune_scores_threshold(attribution_scores, int(len(model.edges)*prune_ratio))
                    # threshold = t.cat([ps.flatten() for _, ps in attribution_scores.items()]).sort(descending=True).values[int(len(model.edges)*prune_ratio) - 1]

                    edges = {}
                    for edge in model.edges:
                        edges[edge.name] = edge.prune_score(attribution_scores).abs() >= threshold

                    ############################################################
                    ### Pruning disconnected nodes
                    G = nx.DiGraph()

                    for conn, active in edges.items():
                        if active:
                            src, dst = conn.split("->")
                            G.add_edge(src, dst)

                    reachable_from_input = set(nx.descendants(G, "Resid Start")) | {"Resid Start"}

                    reversed_G = G.reverse()
                    reachable_to_output = set(nx.descendants(reversed_G, "Resid End")) | {"Resid End"}

                    connected_nodes = reachable_from_input & reachable_to_output

                    pruned_G = G.subgraph(connected_nodes).copy()

                    pruned_edges = [(u, v) for u, v in pruned_G.edges]
                    ############################################################

                    ############################################################
                    ### Convert edges to paths
                    causal_subsets = defaultdict(set)
                    for u, v in pruned_edges:
                        u_info, v_info = extract_node_info(u), extract_node_info(v)
                        # Attention + MLP
                        if u_info["node_type"] == "Attention" and v_info["node_type"] == "MLP" and u_info["bidx"] == v_info["bidx"]:
                            causal_subsets[u_info["bidx"]].add(u_info["hidx"] + 2 + model.cfg.n_heads)
                        # Attention only
                        if u_info["node_type"] == "Attention" and u_info["bidx"] != v_info["bidx"]:
                            causal_subsets[u_info["bidx"]].add(u_info["hidx"] + 2)
                        # MLP only
                        if v_info["node_type"] == "MLP" and u_info["bidx"] != v_info["bidx"]:
                            causal_subsets[v_info["bidx"]].add(1)
                        # Residual only
                        if v_info["bidx"] - u_info["bidx"] > 1:
                            for bidx in range(u_info["bidx"] + 1, v_info["bidx"]):
                                causal_subsets[bidx].add(0)
                    
                    for bidx in range(model.cfg.n_layers):
                        causal_subsets[bidx] = [sorted(list(causal_subsets[bidx]))]
                    # print(causal_subsets)
                    ############################################################

                    for idx in class_indices:
                        idx_4 = "%04d" % idx
                        idx_6 = "%06d" % idx
                        out_path_classwise = out_path + f"_classwise_{prune_ratio}"
                        os.makedirs(os.path.join(out_path_classwise, "results", f"R{idx_4}"), exist_ok=True)
                        with open(os.path.join(out_path_classwise, "results", f"R{idx_4}", f"C{idx_6}.json"), "w") as fout:
                            json.dump(causal_subsets, fout, indent=2)

        attribution_scores = task_attribution_scores
        if attribution_scores is not None:
            for prune_ratio in [0.25]:
                threshold = prune_scores_threshold(attribution_scores, int(len(model.edges)*prune_ratio))
                # threshold = t.cat([ps.flatten() for _, ps in attribution_scores.items()]).sort(descending=True).values[int(len(model.edges)*prune_ratio) - 1]

                edges = {}
                for edge in model.edges:
                    edges[edge.name] = edge.prune_score(attribution_scores).abs() >= threshold

                ############################################################
                ### Pruning disconnected nodes
                G = nx.DiGraph()

                for conn, active in edges.items():
                    if active:
                        src, dst = conn.split("->")
                        G.add_edge(src, dst)

                reachable_from_input = set(nx.descendants(G, "Resid Start")) | {"Resid Start"}

                reversed_G = G.reverse()
                reachable_to_output = set(nx.descendants(reversed_G, "Resid End")) | {"Resid End"}

                connected_nodes = reachable_from_input & reachable_to_output

                pruned_G = G.subgraph(connected_nodes).copy()

                pruned_edges = [(u, v) for u, v in pruned_G.edges]
                ############################################################

                ############################################################
                ### Convert edges to paths
                causal_subsets = defaultdict(set)
                for u, v in pruned_edges:
                    u_info, v_info = extract_node_info(u), extract_node_info(v)
                    # Attention + MLP
                    if u_info["node_type"] == "Attention" and v_info["node_type"] == "MLP" and u_info["bidx"] == v_info["bidx"]:
                        causal_subsets[u_info["bidx"]].add(u_info["hidx"] + 2 + model.cfg.n_heads)
                    # Attention only
                    if u_info["node_type"] == "Attention" and u_info["bidx"] != v_info["bidx"]:
                        causal_subsets[u_info["bidx"]].add(u_info["hidx"] + 2)
                    # MLP only
                    if v_info["node_type"] == "MLP" and u_info["bidx"] != v_info["bidx"]:
                        causal_subsets[v_info["bidx"]].add(1)
                    # Residual only
                    if v_info["bidx"] - u_info["bidx"] > 1:
                        for bidx in range(u_info["bidx"] + 1, v_info["bidx"]):
                            causal_subsets[bidx].add(0)
                
                for bidx in range(model.cfg.n_layers):
                    causal_subsets[bidx] = [sorted(list(causal_subsets[bidx]))]
                # print(causal_subsets)
                ############################################################

                for idx in task_indices:
                    idx_4 = "%04d" % idx
                    idx_6 = "%06d" % idx
                    out_path_taskwise = out_path + f"_taskwise_{prune_ratio}"
                    os.makedirs(os.path.join(out_path_taskwise, "results", f"R{idx_4}"), exist_ok=True)
                    with open(os.path.join(out_path_taskwise, "results", f"R{idx_4}", f"C{idx_6}.json"), "w") as fout:
                        json.dump(causal_subsets, fout, indent=2)
