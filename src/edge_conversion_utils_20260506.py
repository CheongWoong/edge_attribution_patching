import json
import os
from collections import defaultdict

import networkx as nx

from auto_circuit.utils.tensor_ops import prune_scores_threshold


DEFAULT_PRUNE_RATIOS = [0.01, 0.025, 0.05, 0.1, 0.25]


def prune_ratio_label(prune_ratio):
    return f"{prune_ratio:g}"


def conversion_out_path(out_path, conversion_level, prune_ratio, filter_connected):
    suffix = f"_{conversion_level}_{prune_ratio_label(prune_ratio)}"
    if filter_connected:
        suffix += "_connected"
    return out_path + suffix


def pruned_edges_and_metadata(model, attribution_scores, prune_ratio, filter_connected):
    total_edges = len(model.edges)
    edge_count = max(1, int(total_edges * prune_ratio))
    threshold = prune_scores_threshold(attribution_scores, edge_count)

    selected_edges = []
    for edge in model.edges:
        active = edge.prune_score(attribution_scores).abs() >= threshold
        if active:
            selected_edges.append(tuple(edge.name.split("->")))

    if filter_connected:
        graph = nx.DiGraph()
        graph.add_nodes_from(["Resid Start", "Resid End"])
        graph.add_edges_from(selected_edges)
        reachable_from_input = set(nx.descendants(graph, "Resid Start")) | {"Resid Start"}
        reachable_to_output = set(nx.ancestors(graph, "Resid End")) | {"Resid End"}
        connected_nodes = reachable_from_input & reachable_to_output
        kept_edges = list(graph.subgraph(connected_nodes).edges)
    else:
        kept_edges = selected_edges

    metadata = {
        "requested_prune_ratio": prune_ratio,
        "requested_edge_count": edge_count,
        "threshold": float(threshold.detach().cpu()) if hasattr(threshold, "detach") else float(threshold),
        "selected_edge_count": len(selected_edges),
        "selected_edge_ratio": len(selected_edges) / total_edges if total_edges else 0.0,
        "kept_edge_count": len(kept_edges),
        "kept_edge_ratio": len(kept_edges) / total_edges if total_edges else 0.0,
        "total_edge_count": total_edges,
        "filter_connected": filter_connected,
    }
    return kept_edges, metadata


def causal_subsets_from_edges(model, pruned_edges, extract_node_info):
    causal_subsets = defaultdict(set)
    for u, v in pruned_edges:
        u_info, v_info = extract_node_info(u), extract_node_info(v)
        if (
            u_info["node_type"] == "Attention"
            and v_info["node_type"] == "MLP"
            and u_info["bidx"] == v_info["bidx"]
        ):
            causal_subsets[u_info["bidx"]].add(u_info["hidx"] + 2 + model.cfg.n_heads)
        if u_info["node_type"] == "Attention" and u_info["bidx"] != v_info["bidx"]:
            causal_subsets[u_info["bidx"]].add(u_info["hidx"] + 2)
        if v_info["node_type"] == "MLP" and u_info["bidx"] != v_info["bidx"]:
            causal_subsets[v_info["bidx"]].add(1)
        if v_info["bidx"] - u_info["bidx"] > 1:
            for bidx in range(u_info["bidx"] + 1, v_info["bidx"]):
                causal_subsets[bidx].add(0)

    for bidx in range(model.cfg.n_layers):
        causal_subsets[bidx] = [sorted(list(causal_subsets[bidx]))]
    return causal_subsets


def path_circuit_and_metadata(model, attribution_scores, prune_ratio, filter_connected, extract_node_info):
    pruned_edges, metadata = pruned_edges_and_metadata(
        model, attribution_scores, prune_ratio, filter_connected
    )
    causal_subsets = causal_subsets_from_edges(model, pruned_edges, extract_node_info)
    return causal_subsets, metadata


def write_sample_jsons(out_root, idx, causal_subsets, metadata):
    idx_4 = "%04d" % idx
    idx_6 = "%06d" % idx
    result_dir = os.path.join(out_root, "results", f"R{idx_4}")
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, f"C{idx_6}.json"), "w") as fout:
        json.dump(causal_subsets, fout, indent=2)
    with open(os.path.join(result_dir, f"M{idx_6}.json"), "w") as fout:
        json.dump(metadata, fout, indent=2)


def add_summary_record(summary_records, out_root, metadata):
    summary_records[out_root].append(metadata.copy())


def summarize_edge_ratio_records(records):
    if not records:
        return {"count": 0}

    stats = {"count": len(records)}
    for key in [
        "selected_edge_count",
        "selected_edge_ratio",
        "kept_edge_count",
        "kept_edge_ratio",
    ]:
        values = [record[key] for record in records if key in record]
        if values:
            stats[f"mean_{key}"] = sum(values) / len(values)
            stats[f"min_{key}"] = min(values)
            stats[f"max_{key}"] = max(values)
    return stats


def write_edge_ratio_summaries(summary_records):
    for out_root, records in summary_records.items():
        os.makedirs(out_root, exist_ok=True)
        with open(os.path.join(out_root, "edge_ratio_summary.json"), "w") as fout:
            json.dump(
                {
                    "stats": summarize_edge_ratio_records(records),
                    "records": records,
                },
                fout,
                indent=2,
            )
