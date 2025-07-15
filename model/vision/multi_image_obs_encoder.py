import copy
import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F

from common.pytorch_util import dict_apply, replace_submodules
from model.vision.attention_pool_2d import AttentionPool2d
from model.vision.base_encoder import BaseEncoder
from model.vision.crop_randomizer import CropRandomizer
from model.vision.spatial_softmax import SpatialSoftmax


class BatchToTensor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # assuming (N, H, W, C)
        N, H, W, C = x.shape
        default_float_dtype = torch.get_default_dtype()

        # (N, H, W, C) -> (N, C, H, W)
        img = (
            x.permute((0, 3, 1, 2))
            .contiguous()
            .to(
                dtype=default_float_dtype,
                device=x.device,
            )
        )
        return img / 255.0


class MultiImageObsEncoder(BaseEncoder):
    def __init__(
        self,
        shape_meta: dict,
        rgb_model: Union[nn.Module, Dict[str, nn.Module]],
        resize_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        crop_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        random_crop: bool = True,
        color_jitter: Optional[Dict] = None,
        # replace BatchNorm with GroupNorm
        use_group_norm: bool = False,
        # use single rgb model for all rgb inputs
        share_rgb_model: bool = False,
        # renormalize rgb input with imagenet normalization
        # assuming input in [0,1]
        imagenet_norm: bool = False,
        # if > 0, replace avg pool with SpatialSoftmax
        num_keypoints_per_view: int = None,
        # if true, input is (N, H, W, 3) uint8 image.
        raw_rgb_input: bool = False,
        use_attention_pool: bool = False,
    ):
        """
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        # handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map["rgb"] = rgb_model

        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            type = attr.get("type", "low_dim")
            key_shape_map[key] = shape
            if type == "rgb":
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        # have provided model for each key
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_model)

                replace_pooling = (
                    num_keypoints_per_view is not None
                ) or use_attention_pool

                # replace avgpooling.
                if replace_pooling:
                    if this_model is None:
                        enc = rgb_model
                    else:
                        enc = this_model
                    # get feature size.
                    enc.avgpool = torch.nn.Identity()

                    def get_out_channels(module):
                        # Hack for resnet
                        if hasattr(module, "conv3"):
                            return module.conv3.out_channels
                        else:
                            return module.conv2.out_channels

                    if crop_shape is not None:
                        if isinstance(crop_shape, dict):
                            h, w = crop_shape[key]
                        else:
                            h, w = crop_shape
                    else:
                        h, w = shape[1:]
                    out_h = int(math.ceil(h / 32))
                    out_w = int(math.ceil(w / 32))
                    out_c = get_out_channels(enc.layer4[-1])

                    tmp_input = torch.zeros((1, 3, h, w))
                    out_shape = enc(tmp_input).shape
                    # py's resnet's forward does the flattern..
                    assert len(out_shape) == 2
                    assert out_h * out_w * out_c == out_shape[1]

                if use_attention_pool:
                    enc.avgpool = AttentionPool2d(
                        spacial_dim=(out_h, out_w),
                        embed_dim=out_c,
                        # what clip does.
                        num_heads=out_c // 64,
                        # keeps it the same
                        output_dim=None,
                    )

                if num_keypoints_per_view is not None:
                    enc.avgpool = SpatialSoftmax(
                        input_shape=(out_c, out_h, out_w),
                        num_kp=num_keypoints_per_view,
                        temperature=1.0,
                        learnable_temperature=False,
                        output_variance=False,
                        noise_std=0.0,
                    )

                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features // 16,
                                num_channels=x.num_features,
                            ),
                        )
                    key_model_map[key] = this_model

                # configure going from (h, w, 3) uint8 to float32 (3, h, w)
                to_tensor = nn.Identity()
                if raw_rgb_input:
                    to_tensor = BatchToTensor()

                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(size=(h, w))
                    input_shape = (shape[0], h, w)

                # configure random augmentations
                rand_aug = []
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        rand_aug.append(
                            CropRandomizer(
                                input_shape=input_shape,
                                crop_height=h,
                                crop_width=w,
                                num_crops=1,
                                pos_enc=False,
                            )
                        )
                    else:
                        rand_aug.append(
                            torchvision.transforms.CenterCrop(size=(h, w))
                        )
                if color_jitter is not None:
                    rand_aug.append(
                        torchvision.transforms.ColorJitter(
                            brightness=color_jitter["brightness"],
                            contrast=color_jitter["contrast"],
                            saturation=color_jitter["saturation"],
                            hue=tuple(color_jitter["hue"]),
                        )
                    )
                if not rand_aug:
                    this_randomizer = nn.Identity()
                else:
                    this_randomizer = nn.Sequential(*rand_aug)
                # configure normalizer
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    )

                this_transform = nn.Sequential(
                    to_tensor, this_resizer, this_randomizer, this_normalizer
                )
                key_transform_map[key] = this_transform
            elif type == "low_dim":
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.raw_rgb_input = raw_rgb_input

        assert set(self.rgb_keys) == set(self.key_transform_map.keys())

    def forward(self, obs_dict):
        batch_size = None
        features = list()
        # process rgb input
        if self.share_rgb_model:
            # pass all rgb obs to rgb model
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                if self.raw_rgb_input:
                    h, w, c = img.shape[1:]
                    assert (c, h, w) == self.key_shape_map[key]
                else:
                    assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                imgs.append(img)
            # (N*B,C,H,W)
            imgs = torch.cat(imgs, dim=0)
            # (N*B,D)
            feature = self.key_model_map["rgb"](imgs)
            # (N,B,D)
            feature = feature.reshape(-1, batch_size, *feature.shape[1:])
            # (B,N,D)
            feature = torch.moveaxis(feature, 0, 1)
            # (B,N*D)
            feature = feature.reshape(batch_size, -1)
            features.append(feature)
        else:
            # run each rgb obs to independent models
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                if self.raw_rgb_input:
                    h, w, c = img.shape[1:]
                    assert (c, h, w) == self.key_shape_map[key]
                else:
                    assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                features.append(feature)

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)

        # concatenate all features
        result = torch.cat(features, dim=-1)
        return {"features": result}

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta["obs"]
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            this_obs = torch.zeros(
                (batch_size,) + shape, dtype=self.dtype, device=self.device
            )

            if key in self.rgb_keys and self.raw_rgb_input:
                # change this to raw rgb input.
                this_obs = torch.zeros(
                    (batch_size, shape[1], shape[2], shape[0]),
                    dtype=torch.uint8,
                    device=self.device,
                )

            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        output_shape_dict = {}
        for key, value in example_output.items():
            output_shape_dict[key] = value.shape[1:]
        return output_shape_dict
