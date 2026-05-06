import os
import argparse

import torch as t
import numpy as np

from auto_circuit.experiment_utils import load_tl_model
from auto_circuit.utils.graph_utils import patchable_model

from nltk.corpus import stopwords
import json
from collections import OrderedDict, defaultdict
import transformer_lens as tl
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from cpt_utils_20260504 import language_cpt_path, read_cpt_indices
from edge_conversion_utils_20260506 import (
    DEFAULT_PRUNE_RATIOS,
    add_summary_record,
    conversion_out_path,
    path_circuit_and_metadata,
    write_edge_ratio_summaries,
    write_sample_jsons,
)

def custom_load_tl_model(model_name, device):
    try:
        if model_name == "openai-community/gpt2":
            model = load_tl_model("gpt2", device)
        else:
            model = load_tl_model(model_name, device)
    except:
        if "gpt2-xs" in model_name:
            cfg = tl.HookedTransformerConfig(
                d_model = 384,
                n_layers = 6,
                n_heads = 6,
                d_head = 64,
                d_mlp = 4*384, # defaults to 4 * d_model
                n_ctx = 256, # cofig.n_positions
                d_vocab = 50257,
                act_fn = "gelu",
                normalization_type="LN",
                seed = 42,
            )
        else:
            raise NotImplementedError

        model = tl.HookedTransformer(cfg)
        hf_model = AutoModelForCausalLM.from_pretrained(model_name)
        hf_state_dict = hf_model.state_dict()
        new_state_dict = OrderedDict()
        for name, param in hf_state_dict.items():
            # embeddings
            if name == "transformer.wte.weight":
                new_state_dict["embed.W_E"] = param
            elif name == "transformer.wpe.weight":
                new_state_dict["pos_embed.W_pos"] = param

            # transformer blocks
            elif name.startswith("transformer.h."):
                parts = name.split(".")
                layer = int(parts[2])
                subname = ".".join(parts[3:])

                # (a) QKV concat weight: [3*d_model, d_model] = [3*384, 384]
                if subname == "attn.c_attn.weight":
                    W = param.T #[3*384, 384]
                    W = W.reshape(3, cfg.d_model, cfg.d_model)
                    for proj, idx in zip(["Q", "K", "V"], [0, 1, 2]):
                        W_proj = W[idx] #[384, 384]
                        W_proj = W_proj.reshape(cfg.n_heads, cfg.d_head, cfg.d_model).permute(0, 2, 1) #[n_heads, d_model, d_head]
                        new_state_dict[f"blocks.{layer}.attn.W_{proj}"] = W_proj.contiguous()
                elif subname == "attn.c_attn.bias":
                    b = param
                    b = b.reshape(3, cfg.d_model)
                    for proj, idx in zip(["Q", "K", "V"], [0, 1, 2]):
                        b_proj = b[idx]
                        b_proj = b[idx].reshape(cfg.n_heads, cfg.d_head)      # [6, 64]
                        new_state_dict[f"blocks.{layer}.attn.b_{proj}"] = b_proj.contiguous()

                # (b) Attention output projection
                elif subname == "attn.c_proj.weight":
                    W_O = (
                        param.T.reshape(cfg.d_model, cfg.n_heads, cfg.d_head)
                        .permute(1, 2, 0)
                    )                                                        # [6, 64, 384]
                    new_state_dict[f"blocks.{layer}.attn.W_O"] = W_O.contiguous()
                elif subname == "attn.c_proj.bias":
                    new_state_dict[f"blocks.{layer}.attn.b_O"] = param

                # (c) LayerNorms
                elif subname == "ln_1.weight":
                    new_state_dict[f"blocks.{layer}.ln1.w"] = param
                elif subname == "ln_1.bias":
                    new_state_dict[f"blocks.{layer}.ln1.b"] = param
                elif subname == "ln_2.weight":
                    new_state_dict[f"blocks.{layer}.ln2.w"] = param
                elif subname == "ln_2.bias":
                    new_state_dict[f"blocks.{layer}.ln2.b"] = param

                # (d) MLP
                elif subname == "mlp.c_fc.weight":
                    new_state_dict[f"blocks.{layer}.mlp.W_in"] = param
                elif subname == "mlp.c_fc.bias":
                    new_state_dict[f"blocks.{layer}.mlp.b_in"] = param
                elif subname == "mlp.c_proj.weight":
                    new_state_dict[f"blocks.{layer}.mlp.W_out"] = param
                elif subname == "mlp.c_proj.bias":
                    new_state_dict[f"blocks.{layer}.mlp.b_out"] = param

            # --- final LayerNorm & unembed ------------------------------------
            elif name == "transformer.ln_f.weight":
                new_state_dict["ln_final.w"] = param
            elif name == "transformer.ln_f.bias":
                new_state_dict["ln_final.b"] = param
            elif name == "lm_head.weight":
                new_state_dict["unembed.W_U"] = param.T
            else:
                raise ValueError(f"Unexpected key in state dict: {name}")
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
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


parser = argparse.ArgumentParser(description='helloworld')
parser.add_argument("--dataset_name", type=str, required=True, choices=["known_1000", "lama_trex", "ioi_matched_samples"])
parser.add_argument("--model_name", type=str, required=True, choices=["AlgorithmicResearchGroup/gpt2-xs", "EleutherAI/pythia-14m", "EleutherAI/pythia-1b", "openai-community/gpt2"])
parser.add_argument("--score_function", type=str, default="logit", choices=["logit", "logit_diff", "logprob"])
parser.add_argument("--prune_ratios", type=float, nargs="+", default=DEFAULT_PRUNE_RATIOS)
parser.add_argument("--filter_connected", action="store_true")
args = parser.parse_args()

rel_ids = {}
rel_ids["known_1000"] = ['P17', 'P19', 'P20', 'P27', 'P30', 'P36', 'P37', 'P39', 'P101', 'P103', 'P106', 'P108', 'P127', 'P131', 'P136', 'P138', 'P140', 'P159', 'P176', 'P178', 'P190', 'P276', 'P364', 'P407', 'P413', 'P449', 'P463', 'P495', 'P641', 'P740', 'P937', 'P1303', 'P1412']
rel_ids["lama_trex"] = ['P17', 'P19', 'P20', 'P27', 'P30', 'P31', 'P36', 'P37', 'P39', 'P47', 'P101', 'P103', 'P106', 'P108', 'P127', 'P131', 'P136', 'P138', 'P140', 'P159', 'P176', 'P178', 'P190', 'P264', 'P276', 'P279', 'P361', 'P364', 'P407', 'P413', 'P449', 'P463', 'P495', 'P527', 'P530', 'P740', 'P937', 'P1001', 'P1303', 'P1376', 'P1412']

# for dataset_name in ["known_1000", "lama_trex"]:
for dataset_name in [args.dataset_name]:
    # for model_name in ["AlgorithmicResearchGroup/gpt2-xs", "EleutherAI/pythia-14m", "EleutherAI/pythia-1b"]:
    for model_name in [args.model_name]:
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
        model = custom_load_tl_model(model_name, device)

        stopword_list = stopwords.words('english')

        stopword_ids = []
        for stopword in stopword_list:
            token_ids = model.tokenizer.encode(' '+stopword, add_special_tokens=False)
            if len(token_ids) == 1:
                stopword_ids.append(token_ids[0])
            token_ids = model.tokenizer.encode(stopword, add_special_tokens=False)
            if len(token_ids) == 1:
                stopword_ids.append(token_ids[0])
                
        stwd_ids = sorted(list(set(stopword_ids)))

        with open(f"../main/data/{dataset_name}.json", "r") as fin:
            dataset = json.load(fin)

        out_path = os.path.join(f"jobs_EAP_IG_{args.score_function}", dataset_name + "_" + model_name.split("/")[-1])
        os.makedirs(os.path.join(out_path, "inp_info"), exist_ok=True)
        os.makedirs(os.path.join(out_path, "results"), exist_ok=True)

        try:
            model = patchable_model(
                    model,
                    factorized=True,
                    slice_output="last_seq",
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

        cpt_indices = read_cpt_indices(language_cpt_path(dataset_name, model_name))
        cpt_indices = [idx for idx in cpt_indices if 0 <= idx < len(dataset)]
        class_idx_map = defaultdict(list)
        num_incorrect = 0
        for idx in tqdm(cpt_indices):
            line = dataset[idx]
            prompt, label = line["prompt"], line["attribute"]
            inp = model.tokenizer.encode(prompt, return_tensors="pt").to(device)
            out = model(inp)[0][-1]
            out[stwd_ids] *= 0
            top_1_token_idx = t.argmax(out)
            top_1_token = model.tokenizer.decode(top_1_token_idx)
            if label not in top_1_token:
                num_incorrect += 1
            rel_id = line.get("relation_id", "NONE")

            class_idx_map[rel_id].append(idx)
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
                    # print(e)
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
                        {"level": "classwise", "class_id": class_id, "num_samples": len(class_indices)}
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
