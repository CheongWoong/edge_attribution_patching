my_path = "vit_main"

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
from tqdm.auto import tqdm

try:
    from .util import find_knowns_ids
    from .util import custom_load_tl_model_vision as custom_load_tl_model
    from .cpt_utils_20260504 import pending_cpt_indices, read_cpt_indices, vision_cpt_path
except ImportError:
    from util import find_knowns_ids
    from util import custom_load_tl_model_vision as custom_load_tl_model
    from cpt_utils_20260504 import pending_cpt_indices, read_cpt_indices, vision_cpt_path

import sys
vit_path = Path(my_path)
if vit_path not in sys.path:
    sys.path.insert(0, str(vit_path))
from lib.utils import get_model, get_data
from torch.utils.data import DataLoader


def make_corrupted_images(dataset, num_noise_sample, rand_seed):
    t.manual_seed(rand_seed)
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    generator = t.Generator().manual_seed(rand_seed)

    dataloader_for_corrupt = DataLoader(
        dataset, batch_size=num_noise_sample, shuffle=True, generator=generator
    )
    for batch_images, _ in dataloader_for_corrupt:
        patch_size = 16
        B, C, H, W = batch_images.shape
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        total_patches = num_patches_h * num_patches_w

        patches = batch_images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
        shuffle_indices = t.randperm(B * total_patches, generator=generator)
        corrupted_images = t.zeros_like(batch_images)

        for i in range(B):
            for p in range(total_patches):
                shuffle_idx = shuffle_indices[i * total_patches + p]
                src_batch_idx = shuffle_idx // total_patches
                src_patch_idx = shuffle_idx % total_patches
                h_idx = (p // num_patches_w) * patch_size
                w_idx = (p % num_patches_w) * patch_size
                corrupted_images[i, :, h_idx:h_idx+patch_size, w_idx:w_idx+patch_size] = (
                    patches[src_batch_idx, :, src_patch_idx]
                )
        return corrupted_images
    raise RuntimeError("Could not create corrupted images from an empty dataset")


@t.inference_mode()
def corrupt_vision_decision_changed(model, corrupted_images, label, device, batch_size):
    logits_sum = None
    total = 0
    for start in range(0, len(corrupted_images), batch_size):
        batch = corrupted_images[start : start + batch_size].to(device)
        outputs = model(batch)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        logits = outputs.float()
        curr_sum = logits.sum(dim=0)
        logits_sum = curr_sum if logits_sum is None else logits_sum + curr_sum
        total += logits.shape[0]

    avg_logits = logits_sum / total
    pred = t.argmax(avg_logits, dim=-1)
    return pred.item() != label, pred.item()


parser = argparse.ArgumentParser(description='helloworld')
parser.add_argument("--dataset_name", type=str, required=True, choices=["imagenet", "officehome"])
parser.add_argument("--model_name", type=str, required=True, choices=["vit_tiny_patch16_224", "deit_tiny_patch16_224"])
parser.add_argument("--overwrite", action="store_true", help="Recompute every CPT index even when a valid raw result exists.")
parser.add_argument("--score_function", type=str, default="logit", choices=["logit", "logit_diff", "logprob"])
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

        out_path = os.path.join(f"jobs_EAP_IG_{args.score_function}", dataset_name + "_" + model_name.split("/")[-1])
        os.makedirs(os.path.join(out_path, "inp_info"), exist_ok=True)
        os.makedirs(os.path.join(out_path, "results"), exist_ok=True)

        cpt_indices = read_cpt_indices(vision_cpt_path(dataset_name, model_name))
        cpt_indices = [idx for idx in cpt_indices if 0 <= idx < len(dataset)]
        pending_indices, valid_indices = pending_cpt_indices(cpt_indices, out_path)
        run_indices = cpt_indices if args.overwrite else pending_indices
        print(
            f"CPT subset: total={len(cpt_indices)}, valid_existing={len(valid_indices)}, "
            f"empty_or_missing={len(pending_indices)}, to_run={len(run_indices)}, "
            f"overwrite={args.overwrite}"
        )
        if not run_indices:
            print("No empty or missing CPT results. Finished")
            continue

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

        num_noise_sample = 100

        for idx in tqdm(run_indices):
            image, label = dataset[idx]

            image_input = image.unsqueeze(0).to(device)
            output = model(image_input)[0]
            pred = t.argmax(output, dim=-1)
            is_correct = int(pred.item() == label)
            # if is_correct < 0.5:
            #     continue

            idx_6 = "%06d" % idx
            with open(os.path.join(out_path, "inp_info", f"I{idx_6}.txt"), "w") as fout:
                inp_info = f"li:{idx}\nlabel:{label}\npred:{pred.item()}\n"
                fout.write(inp_info)

            corrupt_seed = 0
            while True:
                corrupted_images = make_corrupted_images(dataset, num_noise_sample, corrupt_seed)
                changed, corrupt_pred = corrupt_vision_decision_changed(
                    model, corrupted_images, label, device, num_noise_sample
                )
                if changed:
                    break
                corrupt_seed += 1
            if corrupt_seed:
                print(f"idx={idx}: corrupted average decision changed with seed={corrupt_seed}")
            with open(os.path.join(out_path, "inp_info", f"I{idx_6}.txt"), "a") as fout:
                fout.write(
                    f"corrupt_seed:{corrupt_seed}\n"
                    f"corrupt_avg_pred:{corrupt_pred}\n"
                    f"score_function:{args.score_function}\n"
                )
            wrong_label = corrupt_pred if args.score_function == "logit_diff" else label
            answer_function = "avg_diff" if args.score_function == "logit_diff" else "avg_val"
            grad_function = "logprob" if args.score_function == "logprob" else "logit"

            new_samples = []
            for corrupt_image in corrupted_images:
                new_samples.append(
                    {
                        "clean": image,
                        "corrupt": corrupt_image,
                        "answers": [label],
                        "wrong_answers": [wrong_label],
                    }
                )
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
                grad_function=grad_function,
                answer_function=answer_function,
                # mask_val=0.0,
                integrated_grad_samples=5,
            )

            for key in attribution_scores:
                attribution_scores[key] = attribution_scores[key].cpu().numpy()

            idx_4 = "%04d" % idx
            idx_6 = "%06d" % idx
            os.makedirs(os.path.join(out_path, "results", f"R{idx_4}"), exist_ok=True)
            fout = os.path.join(out_path, "results", f"R{idx_4}", f"raw_C{idx_6}.npy")
            np.save(fout, attribution_scores)
