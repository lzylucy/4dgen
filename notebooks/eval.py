#%%
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, '4dgen'))
    os.chdir(ROOT_DIR)

from common import transformers_pre_import_mods  # isort:skip
import os
import time

import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from imgaug.augmentables.heatmaps import HeatmapsOnImage

from video_common.pytorch_util import dict_apply
from video_common.svd_util import *
from dataset.spartan_video_dataset import SpartanVideoMultiViewDataset
from workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)

#%% add checkpoint information here
exp_name = '2025.07.19/08.00.04_train_video_diffusion_bimanual_mv'
output_dir = f'4dgen/outputs/{exp_name}'
ckpt_path = f'4dgen/outputs/{exp_name}/checkpoints/epoch=026-train_loss=0.029024.ckpt'
cfg = OmegaConf.load(f'{output_dir}/config.yaml')

for key in cfg:
    if OmegaConf.is_dict(cfg[key]) and 'desc' in cfg[key]:
        cfg[key] = cfg[key]['value']

cls = hydra.utils.get_class(cfg._target_)
cfg.logging.mode = 'offline'
cfg.model.params.ckpt_path = ckpt_path
cfg.training.seed = 42
workspace = cls(cfg)
workspace: BaseWorkspace

#%%
device = "cuda" if torch.cuda.is_available() else "cpu"
model = workspace.lightning_module_wrapper.to(device)
model.eval()
#%%
cfg.task = OmegaConf.load('4dgen/config/task/inference.yaml')
dataset = hydra.utils.instantiate(cfg.task.dataset)
assert isinstance(dataset, SpartanVideoMultiViewDataset)
cfg.dataloader.shuffle = False
cfg.dataloader.batch_size = 1
train_dataloader = DataLoader(dataset, **cfg.dataloader)
print("length of train dataloader:", len(train_dataloader))
#%%
def unnormalize(input, min, max):
    return torch.clamp(((input + 1.) / 2.) * (max - min) + min, min, max)

def compute_MSE_P(outputs, view):
    pred_pointmap_v1 = unnormalize(outputs['video_dict'][f'sampled_video_{view}'][:, :3], -1, 2).cpu().detach()
    gt_pointmap_v1 = unnormalize(outputs['video_dict'][f'gt_video_{view}'][:, :3], -1, 2).cpu().detach()
    MSE_P = torch.mean((gt_pointmap_v1 - pred_pointmap_v1).pow(2))
    return MSE_P

def compute_MSE_I(outputs, view):
    pred_color_v1 = unnormalize(outputs['video_dict'][f'sampled_video_{view}'][:, 3:], 0, 1).cpu().detach()
    gt_color_v1 = unnormalize(outputs['video_dict'][f'gt_video_{view}'][:, 3:], 0, 1).cpu().detach()
    MSE_I = torch.mean((pred_color_v1 - gt_color_v1).pow(2))
    return MSE_I

def compute_psnr(outputs, view):
    pred_color_v1 = unnormalize(outputs['video_dict'][f'sampled_video_{view}'][:, 3:], 0, 1).cpu().detach()
    gt_color_v1 = unnormalize(outputs['video_dict'][f'gt_video_{view}'][:, 3:], 0, 1).cpu().detach()
    
    mse = torch.mean((pred_color_v1 - gt_color_v1).pow(2))
    psnr = 10 * torch.log10(1. / mse)
    
    return psnr

def compute_abs_rel(outputs, view):
    d1 = unnormalize(outputs['video_dict'][f'sampled_video_{view}'][:, 2], -1, 2).cpu().detach()
    d2 = unnormalize(outputs['video_dict'][f'gt_video_{view}'][:, 2], -1, 2).cpu().detach()
    
    valid_mask = (d1 > 0) & (d2 > 0)

    d1 = d1[valid_mask]
    d2 = d2[valid_mask]
    
    abs_rel = np.abs(d1 - d2) / d2

    return torch.mean(abs_rel)

def compute_delta_1(outputs, view, threshold=1.25):
    d1 = unnormalize(outputs['video_dict'][f'sampled_video_{view}'][:, 2], -1, 2).cpu().detach()
    d2 = unnormalize(outputs['video_dict'][f'gt_video_{view}'][:, 2], -1, 2).cpu().detach()
    
    valid_mask = (d1 > 0) & (d2 > 0)

    d1 = d1[valid_mask]
    d2 = d2[valid_mask]

    if len(d1) == 0:
        return 0.0

    # Compute the delta
    delta = torch.maximum(d1 / d2, d2 / d1)

    # Compute percentage of pixels where delta < threshold
    percentage = torch.sum(delta < threshold) / len(delta)

    return percentage

#%%
from metrics.calculate_fvd import calculate_fvd

psnr_left_list, psnr_right_list = [], []
fvd_left_list, fvd_right_list = [], []
abs_rel_left_list, abs_rel_right_list = [], []
delta_1_left_list, delta_1_right_list = [], []

# Store each prediction separately
all_predictions = []

for idx, batch in enumerate(train_dataloader):
    num_video_frames = batch['pointmap'].shape[1]

    if idx % 10 == 0:
        print("Progress {}/{}".format(idx, len(train_dataloader)))
        input_batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        input_batch['num_video_frames'] = num_video_frames
        
        start_time = time.time()
        outputs = model.log_images(input_batch)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        
        # Store this prediction separately
        prediction_data = {
            'idx': idx,
            'val_images_left': [],
            'val_images_right': [],
            'gt_images_left': [],
            'gt_images_right': [],
            'recon_images_left': [],
            'recon_images_right': [],
            'val_depth_images_left': [],
            'val_depth_images_right': [],
            'gt_depth_images_left': [],
            'gt_depth_images_right': [],
            'recon_depth_images_left': [],
            'recon_depth_images_right': []
        }
        
        N, C, H, W = outputs['video_dict']['gt_video_right'].shape
        for n in range(N):
            prediction_data['val_images_left'].append(unnormalize(outputs['video_dict']['sampled_video_left'][n][3:], 0, 255).cpu().detach())
            prediction_data['val_images_right'].append(unnormalize(outputs['video_dict']['sampled_video_right'][n][3:], 0, 255).cpu().detach())
            
            prediction_data['gt_images_left'].append(unnormalize(outputs['video_dict']['gt_video_left'][n][3:], 0, 255).cpu().detach())
            prediction_data['gt_images_right'].append(unnormalize(outputs['video_dict']['gt_video_right'][n][3:], 0, 255).cpu().detach())
    
            prediction_data['recon_images_right'].append(unnormalize(outputs['target_reconstructions'][1][n][3:], 0, 255).cpu().detach())
            prediction_data['recon_images_left'].append(unnormalize(outputs['target_reconstructions'][0][n][3:], 0, 255).cpu().detach())
            
            prediction_data['val_depth_images_left'].append(unnormalize(outputs['video_dict']['sampled_video_left'][n][2], -1, 2).cpu().detach())
            prediction_data['val_depth_images_right'].append(unnormalize(outputs['video_dict']['sampled_video_right'][n][2], -1, 2).cpu().detach())
            
            prediction_data['gt_depth_images_left'].append(unnormalize(outputs['video_dict']['gt_video_left'][n][2], -1, 2).cpu().detach())
            prediction_data['gt_depth_images_right'].append(unnormalize(outputs['video_dict']['gt_video_right'][n][2], -1, 2).cpu().detach())
            
            prediction_data['recon_depth_images_right'].append(unnormalize(outputs['target_reconstructions'][1][n][2], -1, 2).cpu().detach())
            prediction_data['recon_depth_images_left'].append(unnormalize(outputs['target_reconstructions'][0][n][2], -1, 2).cpu().detach())
        
        all_predictions.append(prediction_data)
        
        # images
        psnr_left = compute_psnr(outputs, view='left')
        print("left psnr:", psnr_left)
        psnr_left_list.append(psnr_left)
        psnr_right = compute_psnr(outputs, view='right')
        print("right psnr:", psnr_right)
        psnr_right_list.append(psnr_right)
        
        # depth
        abs_rel_left = compute_abs_rel(outputs, view='left')
        print("left abs_rel:", abs_rel_left)
        abs_rel_left_list.append(abs_rel_left)
        abs_rel_right = compute_abs_rel(outputs, view='right')
        print("right abs_rel:", abs_rel_right)
        abs_rel_right_list.append(abs_rel_right)
        
        delta_1_left = compute_delta_1(outputs, view='left')
        print("left delta_1:", delta_1_left)
        delta_1_left_list.append(delta_1_left)
        delta_1_right = compute_delta_1(outputs, view='right')
        print("right delta_1:", delta_1_right)
        delta_1_right_list.append(delta_1_right)
        
        view = 'left'
        pred_color_v1 = unnormalize(outputs['video_dict'][f'sampled_video_{view}'][:, 3:], 0, 255).cpu().detach()
        recon_color_v1 = unnormalize(outputs['target_reconstructions'][0][:, 3:], 0, 255).cpu().detach()
        gt_color_v1 = unnormalize(outputs['video_dict'][f'gt_video_{view}'][:, 3:], 0, 255).cpu().detach()
        fvd = calculate_fvd(pred_color_v1.unsqueeze(0) / 255., gt_color_v1.unsqueeze(0) / 255., device, method='videogpt', only_final=False)
        fvd_left_list.append(fvd['value'])
        print("left fvd:", fvd)
        
        view = 'right'
        pred_color_v2 = unnormalize(outputs['video_dict'][f'sampled_video_{view}'][:, 3:], 0, 255).cpu().detach()
        recon_color_v2 = unnormalize(outputs['target_reconstructions'][1][:, 3:], 0, 255).cpu().detach()
        gt_color_v2 = unnormalize(outputs['video_dict'][f'gt_video_{view}'][:, 3:], 0, 255).cpu().detach()
        fvd = calculate_fvd(pred_color_v2.unsqueeze(0) / 255., gt_color_v2.unsqueeze(0) / 255., device, method='videogpt', only_final=False)
        fvd_right_list.append(fvd['value'])
        print("right fvd:", fvd)

print("Averaged PSNR_left:", torch.mean(torch.stack(psnr_left_list)))
print("Averaged PSNR_right:", torch.mean(torch.stack(psnr_right_list)))
print("Averaged FVD_left:", np.mean(fvd_left_list))
print("Averaged FVD_right:", np.mean(fvd_right_list))

print("Average abs_rel_left:", np.mean(abs_rel_left_list))
print("Average abs_rel_right:", np.mean(abs_rel_right_list))
print("Average delta_1_left:", np.mean(delta_1_left_list))
print("Average delta_1_right:", np.mean(delta_1_right_list))

# %%
import numpy as np
import matplotlib.pyplot as plt
import cv2

def depth_to_heatmap(depth_images):
    heatmaps = []
    _, depth_max = depth_images.min(), depth_images.max()
    for depth_image in depth_images:
        depth_image[depth_image < 0.] = 0.
        normalized_depth = 255 * (depth_image - 0.) / (depth_max - 0.)
        normalized_depth = normalized_depth.astype(np.uint8)
        heatmap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
        heatmap = heatmap.transpose(2, 0, 1)
        heatmaps.append(heatmap)

    return np.array(heatmaps)

#%% Create separate videos for each prediction
import numpy as np
import cv2

fps = 5
os.system(f'mkdir -p 4dgen/data_local/{exp_name}')

# Try different codecs in order of preference
def get_video_writer(filename, fps, frame_size):
    codecs_to_try = [
        'XVID',  # Usually available and VSCode compatible
        'MJPG',  # Motion JPEG - widely supported
        'mp4v',  # MPEG-4 - fallback option
        'DIVX',  # DivX codec
    ]
    
    for codec in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(filename, fourcc, fps, frame_size)
            if out.isOpened():
                print(f"Using codec: {codec} for {filename}")
                return out
            else:
                out.release()
        except Exception as e:
            continue
    
    # If no codec works, try without fourcc (platform default)
    try:
        out = cv2.VideoWriter(filename, -1, fps, frame_size)
        if out.isOpened():
            print(f"Using platform default codec for {filename}")
            return out
    except:
        pass
    
    print(f"Warning: Could not create video writer for {filename}")
    return None

# Process each prediction separately
for pred_idx, prediction in enumerate(all_predictions):
    print(f"Creating videos for prediction {pred_idx + 1}/{len(all_predictions)}")
    
    # Convert depth images to heatmaps for this prediction
    val_depth_heatmaps_left = depth_to_heatmap(np.array(prediction['val_depth_images_left']))
    val_depth_heatmaps_right = depth_to_heatmap(np.array(prediction['val_depth_images_right']))
    recon_depth_heatmaps_left = depth_to_heatmap(np.array(prediction['recon_depth_images_left']))
    recon_depth_heatmaps_right = depth_to_heatmap(np.array(prediction['recon_depth_images_right']))
    gt_depth_heatmaps_left = depth_to_heatmap(np.array(prediction['gt_depth_images_left']))
    gt_depth_heatmaps_right = depth_to_heatmap(np.array(prediction['gt_depth_images_right']))

    # Combine left and right views
    # pred_heatmap_left = np.concatenate([recon_depth_heatmaps_left, val_depth_heatmaps_left], axis=-1)
    # pred_heatmap_right = np.concatenate([recon_depth_heatmaps_right, val_depth_heatmaps_right], axis=-1)
    pred_heatmaps = np.concatenate([val_depth_heatmaps_left, val_depth_heatmaps_right], axis=-1)
    recon_heatmaps = np.concatenate([recon_depth_heatmaps_left, recon_depth_heatmaps_right], axis=-1)
    gt_heatmaps = np.concatenate([gt_depth_heatmaps_left, gt_depth_heatmaps_right], axis=-1)
    pred_images = np.concatenate([np.array(prediction['val_images_left']), np.array(prediction['val_images_right'])], axis=-1).astype(np.uint8)
    recon_images = np.concatenate([np.array(prediction['recon_images_left']), np.array(prediction['recon_images_right'])], axis=-1).astype(np.uint8)
    gt_images = np.concatenate([np.array(prediction['gt_images_left']), np.array(prediction['gt_images_right'])], axis=-1).astype(np.uint8)

    # Create directory for this prediction
    pred_dir = f'4dgen/data_local/{exp_name}/prediction_{pred_idx:03d}'
    os.system(f'mkdir -p {pred_dir}')
    
    # Save reconstruction video
    if len(recon_images) > 0:
        out = get_video_writer(f'{pred_dir}/recon.mp4', fps, (recon_images.shape[3], recon_images.shape[2]))
        if out is not None:
            for i in range(len(recon_images)):
                rgb_image = cv2.cvtColor(rearrange(recon_images[i], 'c h w -> h w c'), cv2.COLOR_BGR2RGB)
                out.write(rgb_image)
            out.release()
    
    # Save prediction video
    if len(pred_images) > 0:
        out = get_video_writer(f'{pred_dir}/pred.mp4', fps, (pred_images.shape[3], pred_images.shape[2]))
        if out is not None:
            for i in range(len(pred_images)):
                rgb_image = cv2.cvtColor(rearrange(pred_images[i], 'c h w -> h w c'), cv2.COLOR_BGR2RGB)
                out.write(rgb_image)
            out.release()

    if len(gt_images) > 0:
        out = get_video_writer(f'{pred_dir}/gt.mp4', fps, (gt_images.shape[3], gt_images.shape[2]))
        if out is not None:
            for i in range(len(gt_images)):
                rgb_image = cv2.cvtColor(rearrange(gt_images[i], 'c h w -> h w c'), cv2.COLOR_BGR2RGB)
                out.write(rgb_image)
            out.release()

    if len(recon_heatmaps) > 0:
        out = get_video_writer(f'{pred_dir}/recon_depth.mp4', fps, (recon_heatmaps.shape[3], recon_heatmaps.shape[2]))
        if out is not None:
            for i in range(len(recon_heatmaps)):
                depth_image = rearrange(recon_heatmaps[i], 'c h w -> h w c')
                out.write(depth_image)
            out.release()

    if len(pred_heatmaps) > 0:
        out = get_video_writer(f'{pred_dir}/pred_depth.mp4', fps, (pred_heatmaps.shape[3], pred_heatmaps.shape[2]))
        if out is not None:
            for i in range(len(pred_heatmaps)):
                depth_image = rearrange(pred_heatmaps[i], 'c h w -> h w c')
                out.write(depth_image)
            out.release()

    if len(gt_heatmaps) > 0:
        out = get_video_writer(f'{pred_dir}/gt_depth.mp4', fps, (gt_heatmaps.shape[3], gt_heatmaps.shape[2]))
        if out is not None:
            for i in range(len(gt_heatmaps)):
                depth_image = rearrange(gt_heatmaps[i], 'c h w -> h w c')
                out.write(depth_image)
            out.release()

    # # Save left depth video
    # if len(pred_heatmap_left) > 0:
    #     out = get_video_writer(f'{pred_dir}/output_left_depth.mp4', fps, (pred_heatmap_left.shape[3], pred_heatmap_left.shape[2]))
    #     if out is not None:
    #         for i in range(len(pred_heatmap_left)):
    #             depth_image = rearrange(pred_heatmap_left[i], 'c h w -> h w c')
    #             out.write(depth_image)
    #         out.release()
    
    # # Save right depth video
    # if len(pred_heatmap_right) > 0:
    #     out = get_video_writer(f'{pred_dir}/output_right_depth.mp4', fps, (pred_heatmap_right.shape[3], pred_heatmap_right.shape[2]))
    #     if out is not None:
    #         for i in range(len(pred_heatmap_right)):
    #             depth_image = rearrange(pred_heatmap_right[i], 'c h w -> h w c')
    #             out.write(depth_image)
    #         out.release()
    
    # # Save individual view videos (left and right separately)
    # if len(prediction['val_images_left']) > 0:
    #     pred_images_left = np.array(prediction['val_images_left']).astype(np.uint8)
    #     out = get_video_writer(f'{pred_dir}/pred_left.mp4', fps, (pred_images_left.shape[3], pred_images_left.shape[2]))
    #     if out is not None:
    #         for i in range(len(pred_images_left)):
    #             rgb_image = cv2.cvtColor(rearrange(pred_images_left[i], 'c h w -> h w c'), cv2.COLOR_BGR2RGB)
    #             out.write(rgb_image)
    #         out.release()
    
    # if len(prediction['val_images_right']) > 0:
    #     pred_images_right = np.array(prediction['val_images_right']).astype(np.uint8)
    #     out = get_video_writer(f'{pred_dir}/pred_right.mp4', fps, (pred_images_right.shape[3], pred_images_right.shape[2]))
    #     if out is not None:
    #         for i in range(len(pred_images_right)):
    #             rgb_image = cv2.cvtColor(rearrange(pred_images_right[i], 'c h w -> h w c'), cv2.COLOR_BGR2RGB)
    #             out.write(rgb_image)
    #         out.release()
    
    print(f"Saved videos for prediction {pred_idx + 1} to {pred_dir}/")

print(f"All predictions saved to separate directories in 4dgen/data_local/{exp_name}/")

# %%
# Save camera parameters (same for all predictions)
if len(all_predictions) > 0:
    # Use the batch from the first prediction to save camera parameters
    np.savetxt(f'4dgen/data_local/{exp_name}/cam_K.txt', np.array(batch['cam_intr'][0][0], dtype=np.float64))
    np.savetxt(f'4dgen/data_local/{exp_name}/cam_extr_1.txt', np.array(batch['cam_extr'][0][0], dtype=np.float64))
    np.savetxt(f'4dgen/data_local/{exp_name}/cam_extr_2.txt', np.array(batch['cam_extr_right'][0][0], dtype=np.float64))
# %%