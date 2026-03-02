import logging
import os
from pathlib import Path
import json
import tarfile
from collections import defaultdict
from einops import rearrange
from typing import List, Any, Union, Tuple
import torch
import torchvision
import numpy as np
import imageio
import PIL.Image
from PIL import Image
import pandas as pd
import csv

CODE_SUFFIXES = {
    ".py",  # Python codes
    ".sh",  # Shell scripts
    ".yaml",
    ".yml",  # Configuration files
}


def build_pretraining_data_loader():
    pass


def logger_filter(name):
    def filter_(record):
        return record["extra"].get("name") == name

    return filter_


def resolve_resume_path(resume, results_dir):
    # Detect the resume path. Support both the experiment index and the full path.
    if resume.isnumeric():
        tmp_dirs = list(Path(results_dir).glob("*"))
        id2exp_dir = defaultdict(list)
        for tmp_dir in tmp_dirs:
            part0 = tmp_dir.name.split("_")[0]
            if part0.isnumeric():
                id2exp_dir[int(part0)].append(tmp_dir)
        resume_id = int(resume)
        valid_exp_dir = id2exp_dir.get(resume_id)
        if len(valid_exp_dir) == 0:
            raise ValueError(
                f"No valid experiment directories found in {results_dir} with the experiment "
                f"index {resume}."
            )
        elif len(valid_exp_dir) > 1:
            raise ValueError(
                f"Multiple valid experiment directories found in {results_dir} with the experiment "
                f"index {resume}: {valid_exp_dir}."
            )
        resume_path = valid_exp_dir[0] / "checkpoints"
    else:
        resume_path = Path(resume)

    if not resume_path.exists():
        raise FileNotFoundError(f"Resume path {resume_path} not found.")

    return resume_path


def dump_codes(save_path, root, sub_dirs=None, valid_suffixes=None, save_prefix="./"):
    """
    Dump codes to the experiment directory.

    Args:
        save_path (str): Path to the experiment directory.
        root (Path): Path to the root directory of the codes.
        sub_dirs (list): List of subdirectories to be dumped. If None, all files in the root directory will
            be dumped. (default: None)
        valid_suffixes (tuple, optional): Valid suffixes of the files to be dumped. If None, CODE_SUFFIXES will be used.
            (default: None)
        save_prefix (str, optional): Prefix to be added to the files in the tarball. (default: './')
    """
    if valid_suffixes is None:
        valid_suffixes = CODE_SUFFIXES

    # Force to use tar.gz suffix
    save_path = safe_file(save_path)
    assert save_path.name.endswith(
        ".tar.gz"
    ), f"save_path should end with .tar.gz, got {save_path.name}."
    # Make root absolute
    root = Path(root).absolute()
    # Make a tarball of the codes
    with tarfile.open(save_path, "w:gz") as tar:
        # Recursively add all files in the root directory
        if sub_dirs is None:
            sub_dirs = list(root.iterdir())
        for sub_dir in sub_dirs:
            for file in Path(sub_dir).rglob("*"):
                if file.is_file() and file.suffix in valid_suffixes:
                    # make file absolute
                    file = file.absolute()
                    arcname = Path(save_prefix) / file.relative_to(root)
                    tar.add(file, arcname=arcname)
    return root


def dump_args(args, save_path, extra_args=None):
    args_dict = vars(args)
    if extra_args:
        assert isinstance(
            extra_args, dict
        ), f"extra_args should be a dictionary, got {type(extra_args)}."
        args_dict.update(extra_args)
    # Save to file
    with safe_file(save_path).open("w") as f:
        json.dump(args_dict, f, indent=4, sort_keys=True, ensure_ascii=False)


def empty_logger():
    logger = logging.getLogger("hymm_empty_logger")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    return logger


def is_valid_experiment(path):
    path = Path(path)
    if path.is_dir() and path.name.split("_")[0].isdigit():
        return True
    return False


def get_experiment_max_number(experiments):
    valid_experiment_numbers = []
    for exp in experiments:
        if is_valid_experiment(exp):
            valid_experiment_numbers.append(int(Path(exp).name.split("_")[0]))
    if valid_experiment_numbers:
        return max(valid_experiment_numbers)
    return 0


def safe_dir(path):
    """
    Create a directory (or the parent directory of a file) if it does not exist.

    Args:
        path (str or Path): Path to the directory.

    Returns:
        path (Path): Path object of the directory.
    """
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    return path


def safe_file(path):
    """
    Create the parent directory of a file if it does not exist.

    Args:
        path (str or Path): Path to the file.

    Returns:
        path (Path): Path object of the file.
    """
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    return path


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=1, fps=24):
    """save videos by video tensor
    copy from: 
    https://github.com/guoyww/AnimateDiff/blob/e92bd5671ba62c0d774a32951453e328018b7c5b/animatediff/utils/util.py#L61

    Args:
        videos (torch.Tensor): video tensor predicted by the model
        path (str): path to save video
        rescale (bool, optional): rescale the video tensor from [-1, 1] to  . Defaults to False.
        n_rows (int, optional): Defaults to 1.
        fps (int, optional): video save fps. Defaults to 8.
    """
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = torch.clamp(x, 0, 1)
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

def load_from_json(filepath: Union[str, Path]) -> Any:
    """Loads a JSON file and returns the content."""
    filepath = Path(filepath)
    if not filepath.exists():
        return None

    with open(filepath, "r") as f:
        return json.load(f)
def save_to_json(data: Any, filepath: Union[str, Path]) -> None:
    """Saves data to a JSON file."""
    filepath = Path(filepath)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def load_from_txt(filepath: Union[str, Path]) -> List[str]:
    """Reads in a text file and returns lines in a list."""
    filepath = Path(filepath)
    if not filepath.exists():
        return None

    with open(filepath, "r") as f:
        lines = f.read().splitlines()
    # Remove empty lines
    lines = [line for line in lines if line.strip()]
    return lines


def load_from_jsonl(filepath: Union[str, Path]) -> List[dict]:
    """Loads a JSON Lines file and returns a list of dictionaries."""
    filepath = Path(filepath)
    assert filepath.suffix == ".jsonl"
    if not filepath.exists():
        return None

    data = []
    with open(filepath, encoding="utf-8") as f:
        for row in f:
            data.append(json.loads(row))
    return data

def load_from_csv(split_file: str, start_index: int = 0, end_index: int = -1) -> pd.DataFrame:

    # load split
    df = pd.read_csv(split_file)
    num_scenes = len(df)
    if start_index >= num_scenes:
        print(f"Start index {start_index} is out of range for {num_scenes} scenes")
        return None

    end_index = end_index
    if end_index > num_scenes:
        end_index = num_scenes
    if end_index < 0 or end_index > num_scenes:
        end_index = num_scenes
    df = df.iloc[start_index:end_index]
    df.reset_index(drop=True, inplace=True)
    df.index = df.index + 1

    return df

def save_to_csv(data: List[dict], output_file: Union[str, Path], fieldnames: List[str]) -> None:
    """Saves a list of dictionaries to a CSV file."""
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)