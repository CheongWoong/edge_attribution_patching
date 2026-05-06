#!/usr/bin/env python3
import datetime as dt
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path("/data8/cwkang/workspace/edge_attribution_patching")
PYTHON = Path("/home/cwkang/miniconda3/envs/EAP/bin/python")
GPUS = [0, 1, 2, 3]

LANGUAGE_JOBS = [
    ("EAP.py", "known_1000", "AlgorithmicResearchGroup/gpt2-xs"),
    ("EAP.py", "known_1000", "EleutherAI/pythia-14m"),
    ("EAP.py", "known_1000", "EleutherAI/pythia-1b"),
    ("EAP.py", "lama_trex", "AlgorithmicResearchGroup/gpt2-xs"),
    ("EAP.py", "lama_trex", "EleutherAI/pythia-14m"),
    ("EAP.py", "lama_trex", "EleutherAI/pythia-1b"),
    ("EAP_IG.py", "known_1000", "AlgorithmicResearchGroup/gpt2-xs"),
    ("EAP_IG.py", "known_1000", "EleutherAI/pythia-14m"),
    ("EAP_IG.py", "known_1000", "EleutherAI/pythia-1b"),
    ("EAP_IG.py", "lama_trex", "AlgorithmicResearchGroup/gpt2-xs"),
    ("EAP_IG.py", "lama_trex", "EleutherAI/pythia-14m"),
    ("EAP_IG.py", "lama_trex", "EleutherAI/pythia-1b"),
]

VISION_JOBS = [
    ("EAP_vit.py", "imagenet", "vit_tiny_patch16_224"),
    ("EAP_vit.py", "officehome", "vit_tiny_patch16_224"),
    ("EAP_vit.py", "imagenet", "deit_tiny_patch16_224"),
    ("EAP_vit.py", "officehome", "deit_tiny_patch16_224"),
    ("EAP_IG_vit.py", "imagenet", "vit_tiny_patch16_224"),
    ("EAP_IG_vit.py", "officehome", "vit_tiny_patch16_224"),
    ("EAP_IG_vit.py", "imagenet", "deit_tiny_patch16_224"),
    ("EAP_IG_vit.py", "officehome", "deit_tiny_patch16_224"),
]

JOBS = LANGUAGE_JOBS + VISION_JOBS


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def slug(text: str) -> str:
    return text.replace("/", "_").replace(":", "_").replace(".py", "")


def run_job(job: tuple[str, str, str], gpu: int, mode: str, log_dir: Path) -> int:
    script, dataset, model = job
    log_path = log_dir / f"{slug(script)}_{slug(dataset)}_{slug(model)}_gpu{gpu}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["OMP_NUM_THREADS"] = "1"
    cmd = [
        str(PYTHON),
        "-u",
        str(ROOT / "src" / script),
        "--dataset_name",
        dataset,
        "--model_name",
        model,
    ]
    if mode == "logit_diff":
        cmd.extend(["--score_function", "logit_diff", "--overwrite"])

    with log_path.open("w") as fout:
        fout.write(f"[{now()}] START gpu={gpu} mode={mode} cmd={' '.join(cmd)}\n")
        fout.flush()
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=fout,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        fout.write(f"\n[{now()}] EXIT status={proc.returncode}\n")
        return proc.returncode


def run_stage(mode: str, log_dir: Path, launcher_log) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    launcher_log.write(f"[{now()}] stage={mode} launching {len(JOBS)} jobs\n")
    launcher_log.flush()
    failures = []
    with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        futures = {}
        for i, job in enumerate(JOBS):
            gpu = GPUS[i % len(GPUS)]
            futures[pool.submit(run_job, job, gpu, mode, log_dir)] = (job, gpu)
        for future in as_completed(futures):
            job, gpu = futures[future]
            status = future.result()
            launcher_log.write(
                f"[{now()}] stage={mode} done gpu={gpu} "
                f"script={job[0]} dataset={job[1]} model={job[2]} status={status}\n"
            )
            launcher_log.flush()
            if status != 0:
                failures.append((job, status))
    if failures:
        raise RuntimeError(f"{mode} stage had {len(failures)} failed jobs: {failures}")
    launcher_log.write(f"[{now()}] stage={mode} all jobs finished\n")
    launcher_log.flush()


def move_with_backup(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        suffix = dt.datetime.now().strftime("%H%M%S")
        dst = dst.with_name(f"{dst.name}_{suffix}")
    shutil.move(str(src), str(dst))


def main() -> int:
    log_root = ROOT / "logs" / "20260504_chained"
    log_root.mkdir(parents=True, exist_ok=True)
    launcher_path = log_root / "launcher.log"
    with launcher_path.open("a") as launcher_log:
        launcher_log.write(f"[{now()}] chained scheduler start pid={os.getpid()}\n")
        launcher_log.flush()
        run_stage("default", log_root / "default", launcher_log)

        launcher_log.write(f"[{now()}] score_function-specific output roots are used; no rename needed\n")
        launcher_log.flush()

        run_stage("logit_diff", log_root / "logit_diff", launcher_log)
        launcher_log.write(f"[{now()}] chained scheduler finished\n")
        launcher_log.flush()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[{now()}] FATAL {exc}", file=sys.stderr)
        raise
