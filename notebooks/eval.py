#%%
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, 'video_policy'))
    os.chdir(ROOT_DIR)

from common import transformers_pre_import_mods  # isort:skip
import os

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
from dataset.spartan_video_dataset import SpartanVideoDataset, SpartanVideoMultiViewDataset
from workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)

#%% add checkpoint information here
exp_name = '2025.04.01/11.56.48_train_video_diffusion_bimanual_mv'
output_dir = f'video_policy/outputs/{exp_name}'
ckpt_path = f'video_policy/outputs/{exp_name}/checkpoints/epoch=045-train_loss=0.024706.ckpt'
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
cfg.task = OmegaConf.load('video_policy/config/task/inference.yaml')
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

gt_images_left, gt_images_right = [], []
val_images_left, val_images_right = [], []
recon_images_left, recon_images_right = [], []
gt_depth_images_left, gt_depth_images_right = [], []
val_depth_images_left, val_depth_images_right = [], []
recon_depth_images_left, recon_depth_images_right = [], []

for idx, batch in enumerate(train_dataloader):
    num_video_frames = batch['pointmap'].shape[1]

    if num_video_frames % 10 == 0:
        print("Progress {}/{}".format(idx, len(train_dataloader)))
        input_batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        input_batch['num_video_frames'] = num_video_frames
        outputs = model.log_images(input_batch)
        
        N, C, H, W = outputs['video_dict']['gt_video_right'].shape
        for n in range(N):
            val_images_left.append(unnormalize(outputs['video_dict']['sampled_video_left'][n][3:], 0, 255).cpu().detach())
            val_images_right.append(unnormalize(outputs['video_dict']['sampled_video_right'][n][3:], 0, 255).cpu().detach())
            
            gt_images_left.append(unnormalize(outputs['video_dict']['gt_video_left'][n][3:], 0, 255).cpu().detach())
            gt_images_right.append(unnormalize(outputs['video_dict']['gt_video_right'][n][3:], 0, 255).cpu().detach())
    
            recon_images_right.append(unnormalize(outputs['target_reconstructions'][1][n][3:], 0, 255).cpu().detach())
            recon_images_left.append(unnormalize(outputs['target_reconstructions'][0][n][3:], 0, 255).cpu().detach())
            
            val_depth_images_left.append(unnormalize(outputs['video_dict']['sampled_video_left'][n][2], -1, 2).cpu().detach())
            val_depth_images_right.append(unnormalize(outputs['video_dict']['sampled_video_right'][n][2], -1, 2).cpu().detach())
            
            gt_depth_images_left.append(unnormalize(outputs['video_dict']['gt_video_left'][n][2], -1, 2).cpu().detach())
            gt_depth_images_right.append(unnormalize(outputs['video_dict']['gt_video_right'][n][2], -1, 2).cpu().detach())
            
            recon_depth_images_right.append(unnormalize(outputs['target_reconstructions'][1][n][2], -1, 2).cpu().detach())
            recon_depth_images_left.append(unnormalize(outputs['target_reconstructions'][0][n][2], -1, 2).cpu().detach())
        
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
#%%
# os.system(f"mkdir -p video_policy/data_local/{exp_name}/rgb_1")
# for i in range(len(val_images_left)):
#     val_image_left = val_images_left[i].permute(1, 2, 0).numpy()
#     im = Image.fromarray(val_image_left.astype(np.uint8))
#     im.save(f"video_policy/data_local/{exp_name}/rgb_1/{i}.jpg")
    
# os.system(f"mkdir -p video_policy/data_local/{exp_name}/rgb_2")
# for i in range(len(val_images_right)):
#     val_image_right = val_images_right[i].permute(1, 2, 0).numpy()
#     im = Image.fromarray(val_image_right.astype(np.uint8))
#     im.save(f"video_policy/data_local/{exp_name}/rgb_2/{i}.jpg")

# os.system(f"mkdir -p video_policy/data_local/{exp_name}/gt_rgb_1")
# for i in range(len(gt_images_left)):
#     gt_image_left = gt_images_left[i].permute(1, 2, 0).numpy()
#     im = Image.fromarray(gt_image_left.astype(np.uint8))
#     im.save(f"video_policy/data_local/{exp_name}/gt_rgb_1/{i}.jpg")
    
# os.system(f"mkdir -p video_policy/data_local/{exp_name}/gt_rgb_2")
# for i in range(len(gt_images_right)):
#     gt_image_right = gt_images_right[i].permute(1, 2, 0).numpy()
#     im = Image.fromarray(gt_image_right.astype(np.uint8))
#     im.save(f"video_policy/data_local/{exp_name}/gt_rgb_2/{i}.jpg")

    
# os.system(f"mkdir -p video_policy/data_local/{exp_name}/depth_1")
# for i in range(len(val_depth_images_left)):
#     val_depth_left = val_depth_images_left[i].numpy()
#     val_depth_left = np.clip(val_depth_left, 0, 2) * 1000
#     np.save(f"video_policy/data_local/{exp_name}/depth_1/{i}.npy", val_depth_left)
    
# os.system(f"mkdir -p video_policy/data_local/{exp_name}/depth_2")
# for i in range(len(val_depth_images_right)):
#     val_depth_right = val_depth_images_right[i].numpy()
#     val_depth_right = np.clip(val_depth_right, 0, 2) * 1000
#     np.save(f"video_policy/data_local/{exp_name}/depth_2/{i}.npy", val_depth_right)
    
# os.system(f"mkdir -p video_policy/data_local/{exp_name}/gt_depth_1")
# for i in range(len(gt_depth_images_left)):
#     gt_depth_left = gt_depth_images_left[i].numpy()
#     gt_depth_left = np.clip(gt_depth_left, 0, 2) * 1000
#     np.save(f"video_policy/data_local/{exp_name}/gt_depth_1/{i}.npy", gt_depth_left)
    
# os.system(f"mkdir -p video_policy/data_local/{exp_name}/gt_depth_2")
# for i in range(len(gt_depth_images_right)):
#     gt_depth_right = gt_depth_images_right[i].numpy()
#     gt_depth_right = np.clip(gt_depth_right, 0, 2) * 1000
#     np.save(f"video_policy/data_local/{exp_name}/gt_depth_2/{i}.npy", gt_depth_right)
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

val_depth_heatmaps_left = depth_to_heatmap(np.array(val_depth_images_left))
val_depth_heatmaps_right = depth_to_heatmap(np.array(val_depth_images_right))
recon_depth_heatmaps_left = depth_to_heatmap(np.array(recon_depth_images_left))
recon_depth_heatmaps_right = depth_to_heatmap(np.array(recon_depth_images_right))
# %%
pred_heatmap_left = np.concatenate([np.array(recon_depth_heatmaps_left), np.array(val_depth_heatmaps_left)], axis=-1)
pred_heatmap_right = np.concatenate([np.array(recon_depth_heatmaps_right), np.array(val_depth_heatmaps_right)], axis=-1)
pred_images = np.concatenate([np.array(val_images_left), np.array(val_images_right)], axis=-1).astype(np.uint8)
recon_images = np.concatenate([np.array(recon_images_left), np.array(recon_images_right)], axis=-1).astype(np.uint8)

#%% write predictions to video
import numpy as np
import cv2
fps = 5
out = cv2.VideoWriter(f'video_policy/data_local/{exp_name}/recon.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (pred_images.shape[3], pred_images.shape[2]))
for i in range(len(recon_images)):
    rgb_image = cv2.cvtColor(rearrange(recon_images[i], 'c h w -> h w c'), cv2.COLOR_BGR2RGB)
    out.write(rgb_image)
out.release()

out = cv2.VideoWriter(f'video_policy/data_local/{exp_name}/pred.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (pred_images.shape[3], pred_images.shape[2]))
for i in range(len(pred_images)):
    rgb_image = cv2.cvtColor(rearrange(pred_images[i], 'c h w -> h w c'), cv2.COLOR_BGR2RGB)
    out.write(rgb_image)
out.release()

out = cv2.VideoWriter(f'video_policy/data_local/{exp_name}/output_left_depth.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (pred_heatmap_left.shape[3], pred_heatmap_left.shape[2]))
for i in range(len(pred_heatmap_left)):
    depth_image = rearrange(pred_heatmap_left[i], 'c h w -> h w c')
    out.write(depth_image)
out.release()

out = cv2.VideoWriter(f'video_policy/data_local/{exp_name}/output_right_depth.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (pred_heatmap_left.shape[3], pred_heatmap_left.shape[2]))
for i in range(len(pred_heatmap_right)):
    depth_image = rearrange(pred_heatmap_right[i], 'c h w -> h w c')
    out.write(depth_image)
out.release()
# %%
# save cam_intrinsics to demo_data
np.savetxt(f'video_policy/data_local/{exp_name}/cam_K.txt', np.array(batch['cam_intr'][0][0], dtype=np.float64))
# save cam_extrinsics to demo_data
np.savetxt(f'video_policy/data_local/{exp_name}/cam_extr_1.txt', np.array(batch['cam_extr'][0][0], dtype=np.float64))
# save cam_extrinsics_right to demo_data
np.savetxt(f'video_policy/data_local/{exp_name}/cam_extr_2.txt', np.array(batch['cam_extr_right'][0][0], dtype=np.float64))
# %%
