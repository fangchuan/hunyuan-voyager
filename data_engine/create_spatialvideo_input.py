import os
import sys
sys.path.append(".")
sys.path.append("..")
import json
import shutil
import argparse


import cv2
import torch
import pyexr
import imageio
import numpy as np
from PIL import Image
import open3d as o3d
import pandas as pd
from loguru import logger
try:
    from moge.model.v1 import MoGeModel
except:
    from MoGe.moge.model.v1 import MoGeModel

from voyager.utils.file_utils import load_from_csv
from voyager.utils.geo_util import rgbd_to_pointcloud

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", type=str, default="/data-nas/data/dataset/qunhe/SpatialVideo/test/processed_wan_vace_data/")
    parser.add_argument("--render_output_dir", type=str, default="./examples/spatialvideo_test/")
    parser.add_argument("--num_frames", type=int, default=49)
    return parser.parse_args()


def camera_list(
    num_frames=49,
    traj_types=["forward"],
    Width=512,
    Height=512,
    fx=256,
    fy=256
):

    cx = Width // 2
    cy = Height // 2

    intrinsic = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    intrinsics = np.stack([intrinsic] * num_frames)

    extrinsics_lst = []
    num_frames_subtraj = num_frames // len(traj_types)
    prev_cam_start_pos = None    # previous camera start position
    prev_cam_end_pos = None      # previous camera end position
    prev_tar_start_pos = None    # previous target start position
    prev_tar_end_pos = None      # previous target end position
    print(f"Generating camera trajectory with types: {traj_types}, total frames: {num_frames}, frames per sub-trajectory: {num_frames_subtraj}")
    for idx, traj_type in enumerate(traj_types):
        assert traj_type in ["forward", "backward", "left", "right", "turn_left", "turn_right", "spiral"], "Invalid camera type"
        if traj_type == "spiral":
            # 螺旋轨迹参数
            radius_start = 0.0  # 起始半径
            radius_end = 1.0    # 结束半径
            height_start = 0.0  # 起始高度
            height_end = 0.0    # 结束高度
            num_cycles = 1      # 螺旋圈数
            
            # 生成螺旋轨迹
            t = np.linspace(0, 1, num_frames_subtraj)
            theta = t * num_cycles * 2 * np.pi  # 角度
            radius = radius_start + (radius_end - radius_start) * t  # 半径逐渐增大
            
            # 相机位置 (螺旋轨迹)
            camera_centers = np.zeros((num_frames_subtraj, 3))
            camera_centers[:, 0] = radius * np.cos(theta)  # x
            camera_centers[:, 1] = height_start + (height_end - height_start) * t  # y (高度)
            camera_centers[:, 2] = radius * np.sin(theta)  # z
            
            # 目标点始终看向螺旋中心稍微前方
            target_points = np.zeros((num_frames_subtraj, 3))
            target_points[:, 0] = 0.0
            target_points[:, 1] = camera_centers[:, 1] - 0.01  # 保持同样的高度
            target_points[:, 2] = 0.0
            
        else:
            # 原有的直线运动轨迹
            start_pos = np.array([0, 0, 0])
            end_pos = np.array([0, 0, 0])
            # for all traj types, we assume the camera moves 1.9m at the fps=49, at the speed of 1m/s 
            if traj_type == "forward":
                end_pos = np.array([0, 0, 2.5]) 
            elif traj_type == "backward":
                end_pos = np.array([0, 0, -2.5])
            elif traj_type == "left":
                end_pos = np.array([-2.5, 0, 0])
            elif traj_type == "right":
                end_pos = np.array([2.5, 0, 0])
            if idx == 0:
                prev_cam_start_pos = start_pos
                prev_cam_end_pos = end_pos
            else:
                # 连续轨迹调整起点和终点
                start_pos = prev_cam_end_pos
                end_pos = prev_cam_start_pos + np.random.randn(3) * 0.1  # slight random noise to avoid exact overlap
                prev_cam_start_pos = start_pos
                prev_cam_end_pos = end_pos
            # Interpolate camera positions along a straight line
            camera_centers = np.linspace(start_pos, end_pos, num_frames_subtraj)
            target_start = np.array([0, 0, 100])  # Target point
            if traj_type == "turn_left":
                target_end = np.array([-100, 0, 0])
            elif traj_type == "turn_right":
                target_end = np.array([100, 0, 0])
            else:
                target_end = np.array([0, 0, 100])
            if idx == 0:
                prev_tar_start_pos = target_start
                prev_tar_end_pos = target_end
            else:
                # 连续轨迹调整起点和终点
                target_start = prev_tar_end_pos
                target_end = prev_tar_start_pos + np.random.randn(3) * 0.1  # slight random noise to avoid exact overlap
                prev_tar_start_pos = target_start
                prev_tar_end_pos = target_end
            target_points = np.linspace(target_start, target_end, num_frames_subtraj * 2)[:num_frames_subtraj]

        for i, (t, target_point) in enumerate(zip(camera_centers, target_points)):
            if traj_type in ["left", "right"]:
                target_point = t + target_point
            
            # 计算朝向
            z = (target_point - t)
            z = z / np.linalg.norm(z)
            x = np.array([1, 0, 0])
            y = np.cross(z, x)
            y = y / np.linalg.norm(y)
            x = np.cross(y, z)

            R = np.stack([x, y, z], axis=0)
            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3] = -R @ t
            extrinsics_lst.append(w2c)
    
    extrinsics = np.stack(extrinsics_lst)
    # Pad extrinsics if needed
    if extrinsics.shape[0] < num_frames:
        num_pad = num_frames - extrinsics.shape[0]
        extrinsics = np.concatenate([extrinsics, np.tile(extrinsics[-1:], (num_pad, 1, 1))], axis=0)
        print(f"intrinsics shape: {intrinsics.shape}, extrinsics shape: {extrinsics.shape}")

    return intrinsics, extrinsics

# from VGGT: https://github.com/facebookresearch/vggt/blob/main/vggt/utils/geometry.py
def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a depth map to camera coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: Camera coordinates (H, W, 3)
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    return cam_coords


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    eps=1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a depth map to world coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).
        extrinsic (np.ndarray): Camera extrinsic matrix of shape (3, 4).

    Returns:
        tuple[np.ndarray, np.ndarray]: World coordinates (H, W, 3) and valid depth mask (H, W).
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # Multiply with the inverse of extrinsic matrix to transform to world coordinates
    # extrinsic_inv is 4x4 (note closed_form_inverse_OpenCV is batched, the output is (N, 4, 4))
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]

    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world  # HxWx3, 3x3 -> HxWx3
    # world_coords_points = np.einsum("ij,hwj->hwi", R_cam_to_world, cam_coords_points) + t_cam_to_world

    return world_coords_points


def render_from_cameras_videos(points, colors, w2c_poses, intrinsics, height, width):
    
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    
    render_list = []
    mask_list = []
    depth_list = []
    # Render from each camera
    for frame_idx in range(len(w2c_poses)):
        # Get corresponding camera parameters
        extrinsic = w2c_poses[frame_idx]
        intrinsic = intrinsics[frame_idx]
        
        camera_coords = (extrinsic @ homogeneous_points.T).T[:, :3]
        projected = (intrinsic @ camera_coords.T).T
        uv = projected[:, :2] / projected[:, 2].reshape(-1, 1)
        depths = projected[:, 2]    
        
        pixel_coords = np.round(uv).astype(int)  # pixel_coords (h*w, 2)      
        valid_pixels = (  # valid_pixels (h*w, )      valid_pixels is the valid pixels in width and height
            (pixel_coords[:, 0] >= 0) & 
            (pixel_coords[:, 0] < width) & 
            (pixel_coords[:, 1] >= 0) & 
            (pixel_coords[:, 1] < height)
        )
        
        pixel_coords_valid = pixel_coords[valid_pixels]  # (h*w, 2) to (valid_count, 2)
        colors_valid = colors[valid_pixels]
        depths_valid = depths[valid_pixels]
        uv_valid = uv[valid_pixels]
        
        
        valid_mask = (depths_valid > 0) & (depths_valid < 60000) # & normal_angle_mask
        colors_valid = colors_valid[valid_mask]
        depths_valid = depths_valid[valid_mask]
        pixel_coords_valid = pixel_coords_valid[valid_mask]

        # Initialize depth buffer
        depth_buffer = np.full((height, width), np.inf)
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Vectorized depth buffer update
        if len(pixel_coords_valid) > 0:
            rows = pixel_coords_valid[:, 1]
            cols = pixel_coords_valid[:, 0]
                            
            # Sort by depth (near to far)
            sorted_idx = np.argsort(depths_valid)
            rows = rows[sorted_idx]
            cols = cols[sorted_idx]
            depths_sorted = depths_valid[sorted_idx]
            # print(f"depths_sorted: min {depths_sorted.min()}, max {depths_sorted.max()}")
            colors_sorted = colors_valid[sorted_idx]

            # Vectorized depth buffer update
            depth_buffer[rows, cols] = np.minimum(
                depth_buffer[rows, cols], 
                depths_sorted
            )
            
            # Get the minimum depth index for each pixel
            flat_indices = rows * width + cols  # Flatten 2D coordinates to 1D index
            unique_indices, idx = np.unique(flat_indices, return_index=True)
            
            # Recover 2D coordinates from flattened indices
            final_rows = unique_indices // width
            final_cols = unique_indices % width
            
            image[final_rows, final_cols] = colors_sorted[idx, :3].astype(np.uint8)

        mask = np.zeros_like(depth_buffer, dtype=np.uint8)
        mask[depth_buffer != np.inf] = 255
        
        render_list.append(image)
        mask_list.append(mask)
        depth_list.append(depth_buffer)
    
    return render_list, mask_list, depth_list


def create_video_input(
    render_list, mask_list, depth_list, render_output_dir,
    separate=True, ref_image=None, ref_depth=None,
    Width=512, Height=512,
    min_percentile=2, max_percentile=98
):
    video_output_dir = os.path.join(render_output_dir)
    os.makedirs(video_output_dir, exist_ok=True)
    video_input_dir = os.path.join(render_output_dir, "video_input")
    os.makedirs(video_input_dir, exist_ok=True)

    value_list = []
    for i, (render, mask, depth) in enumerate(zip(render_list, mask_list, depth_list)):

        # Sky part is the region where depth_max is, also included in mask
        mask = mask > 0
        # depth_max = np.max(depth)
        # non_sky_mask = (depth != depth_max)
        # mask = mask & non_sky_mask
        depth[mask] = 1 / (depth[mask] + 1e-6)
        depth_values = depth[mask]
        
        min_percentile = np.percentile(depth_values, 2)
        max_percentile = np.percentile(depth_values, 98)
        value_list.append((min_percentile, max_percentile))

        depth[mask] = (depth[mask] - min_percentile) / (max_percentile - min_percentile)
        depth[~mask] = depth[mask].min()
        

        # resize to 512x512
        render = cv2.resize(render, (Width, Height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize((mask.astype(np.float32) * 255).astype(np.uint8), \
            (Width, Height), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (Width, Height), interpolation=cv2.INTER_LINEAR)

        # Save mask as png
        mask_path = os.path.join(video_input_dir, f"mask_{i:04d}.png")
        imageio.imwrite(mask_path, mask)
        
        if separate:
            render_path = os.path.join(video_input_dir, f"render_{i:04d}.png")
            imageio.imwrite(render_path, render)
            depth_path = os.path.join(video_input_dir, f"depth_{i:04d}.exr")
            pyexr.write(depth_path, depth)  
        else:
            render = np.concatenate([render, depth], axis=-3)
            render_path = os.path.join(video_input_dir, f"render_{i:04d}.png")
            imageio.imwrite(render_path, render)

        if i == 0:
            if separate:
                ref_image_path = os.path.join(video_output_dir, f"ref_image.png")
                imageio.imwrite(ref_image_path, ref_image)
                ref_depth_path = os.path.join(video_output_dir, f"ref_depth.exr")
                pyexr.write(ref_depth_path, depth) 
            else:
                ref_image = np.concatenate([ref_image, depth], axis=-3)
                ref_image_path = os.path.join(video_output_dir, f"ref_image.png")
                imageio.imwrite(ref_image_path, ref_image)

    with open(os.path.join(video_output_dir, f"depth_range.json"), "w") as f:
        json.dump(value_list, f)


if __name__ == "__main__":
    args = parse_args()
    
    test_data_root_path = args.test_data_path
    output_dir = args.render_output_dir
    num_frames = args.num_frames
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda")
    model = MoGeModel.from_pretrained("/data-nas/models/MoGe/model.pt", local_files_only=True).to(device)

    test_split_filepath = os.path.join(test_data_root_path, "metadata_wan_funcontrol_captioned.csv")
    test_scene_df = load_from_csv(test_split_filepath, start_index=0, end_index=-1)
    print(f"Evaluating on {len(test_scene_df)} scenes.")
    
    test_scene_ids = []
    for idx, row in test_scene_df.iterrows():
        
        scene_id = os.path.basename(os.path.dirname(row["video"]))
        scene_output_dir = os.path.join(output_dir, scene_id)
        os.makedirs(scene_output_dir, exist_ok=True)
        
        ref_rgb_img_path = os.path.join(test_data_root_path, scene_id, "reference_rgb.png")
        image = np.array(Image.open(ref_rgb_img_path).convert("RGB"))
        image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    
        output = model.infer(image_tensor)
        depth = np.array(output['depth'].detach().cpu())
        depth[np.isinf(depth)] = depth[~np.isinf(depth)].max() + 1e4
        logger.info(f"MoGe inference completed for scene {scene_id}, depth range: min {depth.min()}, max {depth.max()}")
        
        Height, Width = image.shape[:2]
        cameras_fileapth = os.path.join(test_data_root_path, scene_id, "cameras.npz")
        # intrinsics, extrinsics = load_camera_info(cameras_fileapth)
        camera_params = np.load(cameras_fileapth, allow_pickle=True)
        c2w_poses = camera_params["c2w_poses"]
        intrinsics = camera_params["intrinsics"]
        depth_min = camera_params["depth_min"]
        depth_max = camera_params["depth_max"]
        scene_scale = camera_params["scene_scale"]
        logger.info(f"original scene_scale: {scene_scale}, depth_min: {depth_min}, depth_max: {depth_max}")
        
        # Backproject point cloud
        intrinsics[:, 0:1] *= Width  # scale intrinsics to image size
        intrinsics[:, 1:2] *= Height
        # normalized_depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        # normalized_depth = (depth - depth_min) / (depth_max - depth_min + 1e-8)
        normalized_depth = depth / scene_scale
        cam_ply = rgbd_to_pointcloud(depth_image=normalized_depth,
                            rgb_image=image,
                            c2w_pose=c2w_poses[0],
                            depth_scale=1.0,
                            intrinsic_mat=intrinsics[0],
        )
        o3d.io.write_point_cloud(os.path.join(scene_output_dir, f"first_frame_pc.ply"), cam_ply)  
        points = np.array(cam_ply.points).reshape(-1, 3)
        colors = np.array(cam_ply.colors).reshape(-1, 3) * 255
        # breakpoint()
        # point_map = depth_to_world_coords_points(depth, extrinsics[0], intrinsics[0])
        # points = point_map.reshape(-1, 3)
        # colors = image.reshape(-1, 3)
        
        # intrinsics, extrinsics = camera_list(
        #     num_frames=num_frames, traj_types=cam_traj_type, Width=Width//2, Height=Height//2, fx=128, fy=128
        # )
        w2c_poses = np.linalg.inv(c2w_poses)  # convert c2w to w2c for rendering
        # render partial RGB-D videos from novel views, in half resolution for faster processing
        intrinsics_half = intrinsics.copy()
        intrinsics_half[:, 0:2] /= 2  # scale intrinsics for half resolution
        render_list, mask_list, depth_list = render_from_cameras_videos(
            points, colors, w2c_poses, intrinsics_half, height=Height//2, width=Width//2
        )
        
        create_video_input(
            render_list, mask_list, depth_list, scene_output_dir, separate=True, 
            ref_image=image, ref_depth=depth, Width=Width, Height=Height)
        
        # copy ground truth RGB video
        gt_rgb_video_path = os.path.join(test_data_root_path, scene_id, "video_rgb.mp4")
        shutil.copy(gt_rgb_video_path, os.path.join(scene_output_dir, "video_rgb.mp4"))
        # input prompt
        txt_prompt = row["prompt"]
        input_prompt_filepath = os.path.join(scene_output_dir, "input_prompt.txt")
        with open(input_prompt_filepath, "w") as f:
            f.write(txt_prompt)
            
        test_scene_ids.append(scene_id)
        
    savev_test_scene_split_filepath = os.path.join(output_dir, "test_scenes.csv")
    df = pd.DataFrame(test_scene_ids, columns=["scene_id"])
    df.to_csv(savev_test_scene_split_filepath, index=False)
