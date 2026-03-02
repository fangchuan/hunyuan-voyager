import os
import sys
sys.path.append(".")
sys.path.append("..")
import argparse
from pathlib import Path
from loguru import logger
from datetime import datetime
import subprocess

import torch
import numpy as np

from voyager.utils.file_utils import load_from_txt, load_from_csv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_dir", type=str, default="/data-nas/data/experiments/zhenqing/HunyuanWorld-Voyager/examples/spatialvideo_test")
    parser.add_argument("--video_resolution", type=str, default="720p")
    parser.add_argument("--video_size", type=int, nargs="+", default=(512, 512))
    parser.add_argument("--infer_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./results_spatialvideo")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    test_data_dir = args.test_data_dir
    video_resolution = args.video_resolution
    infer_steps = args.infer_steps
    seed = args.seed
    curr_date_time = datetime.now().strftime("%Y%m%d")
    output_dir = os.path.join(args.output_dir, f"{curr_date_time}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # test_scene_lst = [f for f in os.listdir(test_data_dir) if os.path.isdir(os.path.join(test_data_dir, f))]
    # test_scene_lst = test_scene_lst[args.start_index:args.end_index]
    test_scene_split_filepath = os.path.join(test_data_dir, "test_scenes.csv")
    test_scene_df = load_from_csv(test_scene_split_filepath, start_index=args.start_index, end_index=args.end_index)
    logger.info(f"Processing scenes from index {args.start_index} to {args.end_index}, total {len(test_scene_df)} scenes.")
    
    num_success = 0
    for _, row in test_scene_df.iterrows():
        scene = row["scene_id"]
        
        scene_dir = os.path.join(test_data_dir, scene)
        prompt_filepath = os.path.join(scene_dir, "input_prompt.txt")
        prompt = load_from_txt(prompt_filepath)[0]
        print(f"Scene: {scene}, Prompt: {prompt}")
        
        scene_output_dir = os.path.join(output_dir, scene)
        os.makedirs(scene_output_dir, exist_ok=True)
        
        voyager_output_video_filepath = os.path.join(scene_output_dir, "video_voyager.mp4")
        if os.path.exists(voyager_output_video_filepath):
            logger.info(f"Output video already exists for scene {scene}, skipping inference.")
            num_success += 1
            continue
        
        voyager_cmd_lst = []
        voyager_cmd_lst.append("python sample_image2video.py")
        voyager_cmd_lst.append("--model HYVideo-T/2")
        voyager_cmd_lst.append(f"--input_path \"{scene_dir}\"")
        voyager_cmd_lst.append(f"--prompt \"{prompt}\"")
        voyager_cmd_lst.append("--i2v_stability")
        voyager_cmd_lst.append(f"--i2v_resolution {video_resolution}")
        voyager_cmd_lst.append(f"--video_size 512 512")
        voyager_cmd_lst.append(f"--infer_steps {infer_steps}")
        voyager_cmd_lst.append("--flow_reverse")
        voyager_cmd_lst.append("--flow_shift 7.0")
        voyager_cmd_lst.append(f"--seed {seed}")
        voyager_cmd_lst.append("--embedded_cfg_scale 6.0")
        voyager_cmd_lst.append("--use_cpu_offload")
        voyager_cmd_lst.append("--use_context_block")
        voyager_cmd_lst.append(f"--save_path {scene_output_dir}")
        voyager_cmd_str = " ".join(voyager_cmd_lst)
        
        logger.info(f"Running command: {voyager_cmd_str}")
        try:
            subprocess.run(voyager_cmd_str, shell=True, check=True)
            logger.info(f"Command executed successfully for scene {scene}")
            num_success += 1
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with exit code {e.returncode} for scene {scene}")
            continue
        
    logger.info(f"Finished processing. Total scenes: {len(test_scene_df)}, Success: {num_success}, Failed: {len(test_scene_df) - num_success}")
        
        