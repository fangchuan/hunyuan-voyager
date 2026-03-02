import os
import sys

sys.path.append(".")
sys.path.append("..")

import subprocess
import logging
import os.path as osp
from time import sleep

import hydra
import pandas as pd
import numpy as np
from omegaconf import DictConfig


pd.set_option("display.max_rows", None)
log = logging.getLogger(__name__)


def submit_job(cfg: DictConfig, start: int, end: int, task_id: int) -> None:
    arena_commands = [
        "arena submit pytorch --namespace=mri --image-pull-policy  IfNotPresent --working-dir=/workspace ",
        f"--name={cfg.owner}-{cfg.task_name}-{task_id}",
        f"--workers=1",
        f"--gpus={cfg.resources.gpus}",
        f"--cpu={cfg.resources.cpu}",
        f"--memory={cfg.resources.memory}",
        f"--image={cfg.env.image}",
        f"--sync-mode={cfg.env.sync_mode}",
        f"--data-dir={cfg.env.data_dir}",
        f"--data-dir={cfg.env.data_high_dir}",
        f"--sync-source={cfg.env.sync_source}",
        f"--sync-branch={cfg.env.sync_branch}",
        # f"--selector=kubernetes.io/hostname=cn-hangzhou.10.111.10.20",
        # f"--selector={cfg.env.selector}",
        # f"--toleration mri-monopolize-node=zhenqing:NoSchedule,Equal",
        # f"-l quota.scheduling.koordinator.sh/name=training-zhenqing",
        # f"--toleration user=zhenqing:NoSchedule,Equal",
        f"--toleration all",
    ]
    arena_commands = " ".join(arena_commands)

    args = " ".join([arg for arg in sys.argv[1:] if "output.path=" not in arg])

    dataset_root_dir = cfg.dataset_rootpath
    output_path = cfg.output.path
    workers = cfg.parallel.workers

    os.makedirs(output_path, exist_ok=True)

    if "test_voyager_on_spatialvideo.py" == cfg.task_script:
        exe_cmd_str = f"python {cfg.task_script} \
                        --test_data_dir {dataset_root_dir} \
                        --video_resolution 720p \
                        --video_size 512 512 \
                        --start_index {start} \
                        --end_index {end+1} \
                        --output_dir {output_path} "                   
    else:
        raise NotImplementedError(f"Unknown task script: {cfg.task_script}")

    script_commands = [
        "env",
        "pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple && pip install jaxtyping omegaconf typeguard shapely swanlab pyexr loguru pandas agentscope ",
        "pip install diffusers==0.31.0 tensorboard==2.19.0 transformers==4.39.3 deepspeed==0.15.1 peft==0.13.2 pyexr==0.5.0 decord",
        exe_cmd_str + args,
    ]
    script_commands = '"' + " && ".join(script_commands) + '"'

    command = arena_commands + " " + script_commands
    log.info(f"Running command: {command}")
    # os.system(command)


@hydra.main(version_base="1.2", config_path="./data_engine/configs", config_name="process_caption_video_dataset_h20.yaml")
def main_dispatcher(cfg: DictConfig) -> None:
    dataset_processed_meta_filepath = "/data-nas/data/experiments/zhenqing/HunyuanWorld-Voyager/examples/spatialvideo_test/test_scenes.csv"
    metadata = pd.read_csv(dataset_processed_meta_filepath)
    num_rooms = len(metadata)

    room_indices = np.arange(num_rooms)
    jobs = min(cfg.parallel.jobs, num_rooms)
    if jobs > 1:
        chunks = np.array_split(room_indices, jobs)
        start_end = [(chunk[0], chunk[-1]) for chunk in chunks]
    else:
        start_end = [(0, num_rooms)]
    log.info(f"start_end: {start_end}")

    batcmd = "arena top node --node-selector node-role.kubernetes.io/mri-debug=true"
    result = subprocess.check_output(batcmd, shell=True)
    lines = result.decode("utf-8").splitlines()
    lines = [line.split() for line in lines if line.startswith("cn-hangzhou")]
    column_names = [
        "name",
        "ip_addresss",
        "role",
        "status",
        "GPU_total",
        "GPU_alloc",
        # "GPU_model",
    ]
    nodes_df = pd.DataFrame(lines, columns=column_names)
    # convert column type from string to int
    nodes_df["GPU_total"] = nodes_df["GPU_total"].astype(int)
    nodes_df["GPU_alloc"] = nodes_df["GPU_alloc"].astype(int)
    nodes_df = nodes_df[(nodes_df["status"] == "Ready") & (nodes_df["GPU_alloc"] <= 8)]
    log.info(f"node_df: {nodes_df}")

    exclude_ip = []

    nodes_df = nodes_df[~nodes_df["name"].isin(exclude_ip)]

    # shuffle nodes df
    random_seed = np.random.randint(0, 2**32 - 1)
    nodes_df = nodes_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    for i, (start, end) in enumerate(start_end):
        if cfg.parallel.exclude:
            if i in cfg.parallel.exclude:
                log.info(f"Skipping job {i}")
                continue
        if cfg.parallel.include:
            if i not in cfg.parallel.include:
                log.info(f"Skipping job {i}")
                continue

        submit_job(cfg, start, end, task_id=i)

        sleep(1)

if __name__ == "__main__":
    main_dispatcher()
