# edge_attribution_patching
This is a repository for running EAP (edge attribution patching).

## Installation
### Set up a Conda environment
This setup script creates an environment named "EAP".
```
bash scripts/installation/setup_conda_env.sh
```

## Run the code
Run the following script to run EAP.  
The edge attribution scores are saved in "jobs_EAP/{dataset_name}_{model_name}/results".
```
python src/EAP.py --dataset_name {dataset_name} --model_name {model_name}
```

Then, run the following command to convert edge attribution scores to a path-level circuit.
```
python src/EAP_edge_to_path_conversion.py --dataset_name {dataset_name} --model_name {model_name}
```
