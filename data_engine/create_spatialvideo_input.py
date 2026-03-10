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
    parser = argparse.ArgumentParser(description="Adapte SpatialVideo data to Hunyuan-Voyager input format")
    parser.add_argument("--test_data_path", type=str, default="/data-nas/data/dataset/qunhe/SpatialVideo/test/processed_wan_vace_data/")
    parser.add_argument("--render_output_dir", type=str, default="./examples/spatialvideo_test/")
    parser.add_argument("--num_frames", type=int, default=49)
    return parser.parse_args()


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
        
        if len(depth_values) == 0:
            min_percentile = 0.0
            max_percentile = 1.0
        else:
            min_percentile = np.percentile(depth_values, 2)
            max_percentile = np.percentile(depth_values, 98)
            
        value_list.append((min_percentile, max_percentile))

        if len(depth_values) > 0 and max_percentile > min_percentile:
            depth[mask] = (depth[mask] - min_percentile) / (max_percentile - min_percentile)
        elif len(depth_values) > 0:
            depth[mask] = 0.0
            
        if len(depth_values) > 0:
            depth[~mask] = depth[mask].min()
        else:
            depth[~mask] = 0.0
        

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
        
        # load reference RGB
        ref_rgb_img_path = os.path.join(test_data_root_path, scene_id, "reference_rgb.png")
        image = np.array(Image.open(ref_rgb_img_path).convert("RGB"))
        image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    
        # run MoGe inference to get reference depth
        output = model.infer(image_tensor)
        depth = np.array(output['depth'].detach().cpu())
        depth[np.isinf(depth)] = depth[~np.isinf(depth)].max() + 1e4
        logger.info(f"MoGe inference completed for scene {scene_id}, depth range: min {depth.min()}, max {depth.max()}")
        
        Height, Width = image.shape[:2]
        # load camera trajectory and intrinsics
        cameras_fileapth = os.path.join(test_data_root_path, scene_id, "cameras.npz")
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
        normalized_depth = depth / scene_scale  # scale depth to align with camera trajectory !!!
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
