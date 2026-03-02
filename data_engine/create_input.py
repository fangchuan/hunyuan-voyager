import os
import json
import argparse
from typing import List, Dict, Tuple


import cv2
import torch
import pyexr
import imageio
import numpy as np
from PIL import Image

try:
    from moge.model.v1 import MoGeModel
except:
    from MoGe.moge.model.v1 import MoGeModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="./example.png")
    parser.add_argument("--render_output_dir", type=str, default="../demo/example/")
    parser.add_argument("--type", type=str, default="forward",
        choices=["forward", "backward", "left", "right", "turn_left", "turn_right", "spiral"])
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


def render_from_cameras_videos(points, colors, extrinsics, intrinsics, height, width):
    
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    
    render_list = []
    mask_list = []
    depth_list = []
    # Render from each camera
    for frame_idx in range(len(extrinsics)):
        # Get corresponding camera parameters
        extrinsic = extrinsics[frame_idx]
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

def create_gt_video_input(
    rgb_files: List[str], 
    depth_files: List[str], 
    render_output_dir: str,
    depth_scale: float = 1000.0,
    separate: bool = True, 
    Width: int = 512, 
    Height: int = 512,
    min_percentile: int = 2, 
    max_percentile: int = 98,
    ref_image: np.ndarray = None
):
    """
    Create Voyager input data from ground truth RGB and depth files. (Testing with GT data)
    
    Args:
        rgb_files (List[str]): List of file paths to RGB images.
        depth_files (List[str]): List of file paths to depth images.
        render_output_dir (str): Directory to save the output data.
        depth_scale (float): Scale factor to convert depth values to meters.
        separate (bool): Whether to save render and depth separately.
        Width (int): Width to resize images.
        Height (int): Height to resize images.
        min_percentile (int): Minimum percentile for depth normalization.
        max_percentile (int): Maximum percentile for depth normalization.
        ref_image (np.ndarray): Reference image (first frame) for saving.
        
    Returns:
        None
    """
    video_output_dir = os.path.join(render_output_dir)
    os.makedirs(video_output_dir, exist_ok=True)
    video_input_dir = os.path.join(render_output_dir, "video_input")
    os.makedirs(video_input_dir, exist_ok=True)

    value_list = []
    for i, (rgb_filepath, depth_filepath) in enumerate(zip(rgb_files, depth_files)):
        file_name = os.path.basename(rgb_filepath)
        print(f"Processing frame {i}: {file_name}")
        assert os.path.basename(rgb_filepath) == os.path.basename(depth_filepath), \
            f"RGB file and depth file do not match: {rgb_filepath}, {depth_filepath}"
            
        rgb = np.array(Image.open(rgb_filepath).convert("RGB"))
        depth = np.array(Image.open(depth_filepath)).astype(np.float32)
        print(f"Original depth range: min {depth.min()}, max {depth.max()}")
        depth = depth / depth_scale  # scale depth to meters
        
        # skip region with depth == 0
        mask = depth > 0
    
        # mask rgb
        render = np.where(mask[..., None], rgb, 0)
        
        # invevrse depth
        # depth[mask] = 1 / (depth[mask] + 1e-6)
        depth[mask] = 1 / depth[mask]
        depth_values = depth[mask]
        print(f"Inversed depth range before normalization: min {depth_values.min()}, max {depth_values.max()}")
        
        min_percentile = np.percentile(depth_values, 2)
        max_percentile = np.percentile(depth_values, 98)
        value_list.append((min_percentile, max_percentile))

        depth[mask] = (depth[mask] - min_percentile) / (max_percentile - min_percentile)
        depth[~mask] = depth[mask].min()
        

        # resize to 512x512
        render = cv2.resize(render, (Width, Height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize((mask.astype(np.float32) * 255).astype(np.uint8), (Width, Height), interpolation=cv2.INTER_NEAREST)
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

import open3d as o3d
def rgbd_to_pointcloud(
    depth_image: np.array,
    c2w_pose: np.array,
    depth_scale: float = 1.0,
    intrinsic_mat: np.array = None,
    rgb_image: np.array = None,
    normal_image: np.array = None,
) -> o3d.geometry.PointCloud:
    """
    Convert RGBD image to point cloud
    params:
        depth_image: np.array, depth image
        c2w_pose: np.array, camera to world pose
        depth_scale: float, depth scale
        fl_x: float, focal length in x axis
        fl_y: float, focal length in y axis
        rgb_image: np.array, rgb image
    """

    depth_image = (depth_image * depth_scale).astype(np.float32)
    if len(depth_image.shape) == 2:
        depth_image = np.expand_dims(depth_image, axis=2)
    H, W, C = depth_image.shape
    n_pts = H * W
    # Get camera intrinsic
    if intrinsic_mat is not None:
        K = intrinsic_mat
    else:
        hfov = 90.0 * np.pi / 180.0
        fl_x = W / 2.0 / np.tan((hfov / 2.0))
        K = np.array(
            [
                [fl_x, 0.0, W / 2.0],
                [0.0, fl_x, H / 2.0],
                [
                    0.0,
                    0.0,
                    1,
                ],
            ]
        )

    depth_map = depth_image.reshape(1, H, W)

    pts_x = np.linspace(0, W, W)
    pts_y = np.linspace(0, H, H)
    
    pts_xx, pts_yy = np.meshgrid(pts_x, pts_y)

    pts = np.stack((pts_xx, pts_yy, np.ones_like(pts_xx)), axis=0)
    pts = np.linalg.inv(K) @ (pts * depth_map).reshape(3, n_pts)
    # filter out invalid points with large gradient
    points_grad = np.zeros((H, W, 1))
    points_map = pts.T.reshape(H, W, 3)
    points_grad_x = points_map[2:, 1:-1] - points_map[:-2, 1:-1]
    points_grad_x = np.linalg.norm(points_grad_x.reshape(-1, 3), axis=-1)
    points_grad_y = points_map[1:-1, 2:] - points_map[1:-1, :-2]
    points_grad_y = np.linalg.norm(points_grad_y.reshape(-1, 3), axis=-1)
    # print(f"points_grad_x: {points_grad_x.shape}, points_grad_y: {points_grad_y.shape}")
    grad = np.sqrt(points_grad_x**2 + points_grad_y**2)
    # print(f"grad: {grad.shape}")
    points_grad[1:-1, 1:-1, 0] = grad.reshape(H - 2, W - 2)
    grad_thresh = grad.mean() * 2

    # invalid_mask = points_grad.mean(axis=2) > grad_thresh
    # invalid_mask = invalid_mask.reshape(n_pts)
    invalid_mask = np.zeros(n_pts, dtype=bool)

    if rgb_image is not None:
        color = (rgb_image[:, :, :3] / 255.0).clip(0.0, 1.0)
    else:
        color = np.zeros_like(pts[:3].T)

    if normal_image is not None:
        # convert normal to [-1, 1]
        normal = np.clip((normal_image + 0.5) / 255.0, 0.0, 1.0) * 2 - 1
        normal = normal / (np.linalg.norm(normal, axis=2)[:, :, np.newaxis] + 1e-6)

        points, colors, normals = (
            np.transpose(pts)[:, :3],
            color.reshape(n_pts, 3),
            normal.reshape(n_pts, 3),
        )
    else:
        points, colors = np.transpose(pts)[:, :3], color.reshape(n_pts, 3)
        normals = np.zeros_like(points)

    points = points[~invalid_mask]
    colors = colors[~invalid_mask]
    normals = normals[~invalid_mask]
    o3d_ply = o3d.geometry.PointCloud()
    o3d_ply.points = o3d.utility.Vector3dVector(points)
    o3d_ply.colors = o3d.utility.Vector3dVector(colors)
    o3d_ply.normals = o3d.utility.Vector3dVector(normals)

    return o3d_ply.transform(c2w_pose)

def debug_gt_rgbd():
    # debug RGB-D pair
    gt_rgb_filepath = "/data-nas/data/experiments/zhenqing/HunyuanWorld-Voyager/examples/scene_001013_00_0_gt/raw_data/rgbs/0.png"
    gt_depth_filepath = "/data-nas/data/experiments/zhenqing/HunyuanWorld-Voyager/examples/scene_001013_00_0_gt/raw_data/depths/0.png"
    debug_rgb = np.array(Image.open(gt_rgb_filepath).convert("RGB"))
    debug_depth = np.array(Image.open(gt_depth_filepath)).astype(np.float32) / 1000.0
    print(f"Debug GT depth range: min {debug_depth.min()}, max {debug_depth.max()}")
    fovx = 90.0 * np.pi / 180.0
    fl = Width / 2.0 / np.tan(fovx / 2.0)
    vfov = 90.0
    hfov = 106.26020470831195
    intrinsic_mat = np.array([
        [240.0, 0.0, debug_rgb.shape[1] / 2.0],
        [0.0, 240.0, debug_rgb.shape[0] / 2.0],
        [0.0, 0.0, 1.0],
    ])
    debug_point_map = rgbd_to_pointcloud(
        depth_image=debug_depth,
        c2w_pose=np.eye(4),
        depth_scale=1.0,
        rgb_image=debug_rgb,
        intrinsic_mat=intrinsic_mat
    )
    o3d.io.write_point_cloud("/data-nas/data/experiments/zhenqing/HunyuanWorld-Voyager/examples/scene_001013_00_0_gt/debug_debug.ply", debug_point_map)
    
if __name__ == "__main__":
    args = parse_args()
    
    num_frames = args.num_frames
    if num_frames <= 49:
        cam_traj_type = [args.type]
    else:
        if args.type == "forward":
            cam_traj_type = ["forward", "backward"]
        elif args.type == "backward":
            cam_traj_type = ["backward", "forward"]
        elif args.type == "left":
            cam_traj_type = ["left", "right"]
        elif args.type == "right":
            cam_traj_type = ["right", "left"]
        elif args.type == "turn_left":
            cam_traj_type = ["turn_left", "turn_right"]
        elif args.type == "turn_right":
            cam_traj_type = ["turn_right", "turn_left"]
        elif args.type == "spiral":
            cam_traj_type = ["spiral"]
    device = torch.device("cuda")
    model = MoGeModel.from_pretrained("/data-nas/models/MoGe/model.pt", local_files_only=True).to(device)

    image = np.array(Image.open(args.image_path).convert("RGB").resize((1280, 720)))
    image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    
    output = model.infer(image_tensor)
    depth = np.array(output['depth'].detach().cpu())
    depth[np.isinf(depth)] = depth[~np.isinf(depth)].max() + 1e4
    
    Height, Width = image.shape[:2]
    intrinsics, extrinsics = camera_list(
        num_frames=1, traj_types=cam_traj_type[:1], Width=Width, Height=Height, fx=256, fy=256
    )

    # Backproject point cloud
    point_map = depth_to_world_coords_points(depth, extrinsics[0], intrinsics[0])
    points = point_map.reshape(-1, 3)
    colors = image.reshape(-1, 3)
    
    intrinsics, extrinsics = camera_list(
        num_frames=num_frames, traj_types=cam_traj_type, Width=Width//2, Height=Height//2, fx=128, fy=128
    )
    render_list, mask_list, depth_list = render_from_cameras_videos(
        points, colors, extrinsics, intrinsics, height=Height//2, width=Width//2
    )
    
    create_video_input(
        render_list, mask_list, depth_list, args.render_output_dir, separate=True, 
        ref_image=image, ref_depth=depth, Width=Width, Height=Height)

    # # create Voyager input with gt RGB, depth, and mask
    # gt_rgb_dir = "/data-nas/data/experiments/zhenqing/HunyuanWorld-Voyager/examples/scene_001013_00_0_gt/raw_data/rgbs"
    # gt_depth_dir = "/data-nas/data/experiments/zhenqing/HunyuanWorld-Voyager/examples/scene_001013_00_0_gt/raw_data/depths"
    # gt_rgb_files = sorted([os.path.join(gt_rgb_dir, f) for f in os.listdir(gt_rgb_dir) if f.endswith(".png")], key=lambda x: int(os.path.basename(x).split(".")[0]))
    # gt_depth_files = sorted([os.path.join(gt_depth_dir, f) for f in os.listdir(gt_depth_dir) if f.endswith(".png")], key=lambda x: int(os.path.basename(x).split(".")[0]))
    # create_gt_video_input(
    #     rgb_files=gt_rgb_files, 
    #     depth_files=gt_depth_files, 
    #     render_output_dir=args.render_output_dir,
    #     separate=True, 
    #     Width=Width, 
    #     Height=Height,
    #     min_percentile=2, 
    #     max_percentile=98,
    #     ref_image=image
    # )