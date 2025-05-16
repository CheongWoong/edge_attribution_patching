my_path = "/data/cwkang/CausalPathTracing/test/CausalPathTracing_for_ViT/main"

import os
import argparse

import torch as t
import numpy as np
import random

from auto_circuit.data import load_datasets_from_json
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.types import PruneScores
from auto_circuit.utils.graph_utils import patchable_model

from pathlib import Path
from vit_prisma.models.base_vit import HookedViT
from tqdm.auto import tqdm

import sys
vit_path = Path(my_path)
if vit_path not in sys.path:
    sys.path.insert(0, str(vit_path))
from lib.utils import get_model, get_data
from torch.utils.data import DataLoader

def custom_load_tl_model(model_name, dataset_name, new_head_state_dict, num_classes, device):
    assert dataset_name in ["imagenet", "officehome"]
    if dataset_name == "imagenet":
        model = HookedViT.from_pretrained(model_name,
            center_writing_weights=True,
            center_unembed=True,
            fold_ln=True,
            refactor_factored_attn_matrices=True,
        )
    elif dataset_name == "officehome":
        model = HookedViT.from_pretrained(model_name,
            center_writing_weights=True,
            center_unembed=True,
            fold_ln=True,
            refactor_factored_attn_matrices=True,
            new_head_state_dict=new_head_state_dict,
            num_classes=num_classes,
        )
    
    model.cfg.use_attn_result = True
    model.cfg.use_attn_in = True
    model.cfg.use_hook_mlp_in = True

    model.cfg.use_split_qkv_input = False
    model.cfg.tokenizer_prepends_bos = False
    model.cfg.default_prepend_bos = False

    model.cfg.return_type = "logits"

    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


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

        out_path = os.path.join("jobs_edge_patching_baseline", dataset_name + "_" + model_name.split("/")[-1])
        os.makedirs(os.path.join(out_path, "inp_info"), exist_ok=True)
        os.makedirs(os.path.join(out_path, "results"), exist_ok=True)

        try:
            model = patchable_model(
                    model,
                    factorized=True,
                    slice_output=None,
                    separate_qkv=False,
                    device=device,
                )
        except Exception as e:
            print("[Error]", e)

        def find_knowns_ids(model, data="imagenet"):
            assert data in ["imagenet", "officehome"]
            jobs_dir = "jobs" if data=="imagenet" else "jobs_oh"
            print("Finding knowns ids")
            job_root = Path(my_path) / jobs_dir / str(model)
            ids = []
            dir_list = os.listdir(job_root)
            print(len(dir_list))
            for entry in job_root.iterdir():
                if entry.is_dir():
                    name = entry.name
                    idx = name.split("_")[1]

                    results_dir = entry / "results"
                    # Check if the results directory exists and is not empty
                    if results_dir.exists() and any(results_dir.iterdir()):
                        ids.append(int(idx))
                    
            print("Done finding knowns ids")
            print(f"Found {len(ids)} knowns ids")
            ids = sorted(ids)
            return sorted(ids)

        known_ids = find_knowns_ids(model_name, dataset_name)

        rand_seed = 0
        num_noise_sample = 100

        for idx in tqdm(known_ids):
            image, label = dataset[idx]

            image_input = image.unsqueeze(0).to(device)
            output = model(image_input)[0]
            pred = t.argmax(output, dim=-1)
            is_correct = int(pred.item() == label)
            if is_correct < 0.5:
                continue

            idx_6 = "%06d" % idx
            with open(os.path.join(out_path, "inp_info", f"I{idx_6}.txt"), "w") as fout:
                inp_info = f"li:{idx}\nlabel:{label}\npred:{pred.item()}\n"
                fout.write(inp_info)

            # Create corrupted images by replacing random patches
            t.manual_seed(rand_seed)
            np.random.seed(rand_seed)
            random.seed(rand_seed)

            dataloader_for_corrupt = DataLoader(dataset, batch_size=num_noise_sample, shuffle=True)
            for batch_images, _ in dataloader_for_corrupt:
                patch_size = 16
                B, C, H, W = batch_images.shape
                num_patches_h = H // patch_size
                num_patches_w = W // patch_size
                total_patches = num_patches_h * num_patches_w

                # Extract all patches from all images
                patches = batch_images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
                patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)  # [B, C, num_patches, 16, 16]

                # Create random indices for shuffling
                shuffle_indices = t.randperm(B * total_patches)

                # Create new images with shuffled patches
                corrupted_images = t.zeros_like(batch_images)

                for i in range(B):
                    for p in range(total_patches):
                        # Calculate source batch and patch indices
                        shuffle_idx = shuffle_indices[i * total_patches + p]
                        src_batch_idx = shuffle_idx // total_patches
                        src_patch_idx = shuffle_idx % total_patches
                        
                        # Calculate patch positions
                        h_idx = (p // num_patches_w) * patch_size
                        w_idx = (p % num_patches_w) * patch_size
                        
                        # Place shuffled patch in new position
                        corrupted_images[i, :, h_idx:h_idx+patch_size, w_idx:w_idx+patch_size] = \
                            patches[src_batch_idx, :, src_patch_idx]
                break  # Just take the first batch

            new_samples = []
            for corrupt_image in corrupted_images:
                new_samples.append({"clean": image, "corrupt": corrupt_image, "answers": label, "wrong_answers": label})
            new_data = {"prompts": new_samples}

            train_loader, test_loader = load_datasets_from_json(
                model=None,
                path=None,
                device=device,
                prepend_bos=False,
                batch_size=num_noise_sample,
                train_test_size=(num_noise_sample, 0),
                shuffle=False,
                data=new_data,
            )

            attribution_scores: PruneScores = mask_gradient_prune_scores(
                model=model,
                dataloader=train_loader,
                official_edges=None,
                grad_function="logit",
                answer_function="avg_val",
                mask_val=0.0,
            )

            for key in attribution_scores:
                attribution_scores[key] = attribution_scores[key].cpu().numpy()

            idx_4 = "%04d" % idx
            idx_6 = "%06d" % idx
            os.makedirs(os.path.join(out_path, "results", f"R{idx_4}"), exist_ok=True)
            fout = os.path.join(out_path, "results", f"R{idx_4}", f"raw_C{idx_6}.npy")
            np.save(fout, attribution_scores)
