my_path = "vit_main"
from pathlib import Path
import os

from vit_prisma.models.base_vit import HookedViT


def find_knowns_ids(model, data="imagenet"):
    assert data in ["imagenet", "officehome"]
    # jobs_dir = "jobs" if data=="imagenet" else "jobs_oh"
    jobs_dir = "jobs"
    print("Finding knowns ids")
    # job_root = Path(my_path) / jobs_dir / str(model)
    job_root = Path(my_path) / jobs_dir / (str(model)+"_"+str(data)) / "results"
    ids = []
    dir_list = os.listdir(job_root)
    print(len(dir_list))
    for entry in job_root.iterdir():
        if entry.is_dir():
            name = entry.name
            # idx = name.split("_")[1]
            idx = name[1:]

            results_dir = entry
            # Check if the results directory exists and is not empty
            if results_dir.exists():
                if any(results_dir.iterdir()):
                    ids.append(int(idx))
                else:
                    print("DD", idx)
            
    print("Done finding knowns ids")
    print(f"Found {len(ids)} knowns ids")
    ids = sorted(ids)
    return sorted(ids)

def custom_load_tl_model_vision(model_name, dataset_name, new_head_state_dict, num_classes, device):
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