import time
times = []
import os
import argparse

import torch as t
import numpy as np

from auto_circuit.data import load_datasets_from_json
from auto_circuit.experiment_utils import load_tl_model
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.types import PruneScores
from auto_circuit.utils.graph_utils import patchable_model

from nltk.corpus import stopwords
from pathlib import Path
import json
from collections import OrderedDict
import transformer_lens as tl
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

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
args = parser.parse_args()

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

        out_path = os.path.join("jobs_EAP", dataset_name + "_" + model_name.split("/")[-1])
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

        rand_seed = 0
        num_noise_sample = 100

        for idx, line in enumerate(tqdm(dataset)):
            prompt, label = line["prompt"], line["attribute"]
            inp = model.tokenizer.encode(prompt, return_tensors="pt").to(device)
            out = model(inp)[0][-1]
            out[stwd_ids] *= 0
            top_1_token_idx = t.argmax(out)
            top_1_token = model.tokenizer.decode(top_1_token_idx)
            is_correct = int(label in top_1_token)
            if is_correct < 0.5 and (idx != 2093 or args.dataset_name != "lama_trex" or "gpt2-xs" not in args.model_name):
                continue

            idx_6 = "%06d" % idx
            with open(os.path.join(out_path, "inp_info", f"I{idx_6}.txt"), "w") as fout:
                inp_info = f"li:{idx}\nprompt:{prompt}\ny:{label}\n"
                fout.write(inp_info)

            # make corrupted samples
            prng = np.random.RandomState(rand_seed)
            enc_prompt = model.tokenizer.encode(prompt)
            available_tokens = list(set(range(model.tokenizer.vocab_size)) - set(enc_prompt))
            selected_tokens = prng.choice(available_tokens, size=num_noise_sample*len(enc_prompt))
            corrupt_tokens = selected_tokens.reshape(num_noise_sample, -1).tolist()
            ans = model.tokenizer.encode(" " + label)

            new_samples = []
            for ct in corrupt_tokens:
                new_samples.append({"clean": enc_prompt, "corrupt": ct, "answers": ans, "wrong_answers": ans})
            new_data = {"prompts": new_samples}
            with open(f"temp_{args.dataset_name}_{args.model_name.split('/')[-1]}.json", "w") as fout_temp:
                json.dump(new_data, fout_temp)

            path = Path(f"temp_{args.dataset_name}_{args.model_name.split('/')[-1]}.json")
            train_loader, test_loader = load_datasets_from_json(
                model=None,
                path=path,
                device=device,
                prepend_bos=False,
                batch_size=100 if ("pythia-1b" not in model_name and "openai-community/gpt2" not in model_name) else 10,
                train_test_size=(num_noise_sample, 0),
                shuffle=False,
            )

            start_time = time.time()
            attribution_scores: PruneScores = mask_gradient_prune_scores(
                model=model,
                dataloader=train_loader,
                official_edges=None,
                grad_function="logit",
                answer_function="avg_val",
                mask_val=0.0,
            )
            end_time = time.time()
            time_taken = end_time - start_time
            times.append(time_taken)
            print("^^^", len(times), "samples / Mean:", np.mean(times), "/ Median:", np.median(times), "/ Std:", np.std(times), "&&&")

            for key in attribution_scores:
                attribution_scores[key] = attribution_scores[key].cpu().numpy()

            idx_4 = "%04d" % idx
            idx_6 = "%06d" % idx
            os.makedirs(os.path.join(out_path, "results", f"R{idx_4}"), exist_ok=True)
            fout = os.path.join(out_path, "results", f"R{idx_4}", f"raw_C{idx_6}.npy")
            np.save(fout, attribution_scores)
        print("Finished")
