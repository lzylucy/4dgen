import torch
import torch.nn as nn
from copy import copy, deepcopy
from packaging import version

OPENAIUNETWRAPPER = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapper"
FEATURE_DIM = 4

class IdentityWrapper(nn.Module):
    def __init__(self, diffusion_model, compile_model: bool = False):
        super().__init__()
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0"))
            and compile_model
            else lambda x: x
        )
        
        diffusion_model_2 = copy(diffusion_model)
        diffusion_model_2.output_blocks = deepcopy(diffusion_model.output_blocks)
        self.diffusion_model = compile(diffusion_model)
        self.diffusion_model_2 = compile(diffusion_model_2)

    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


class OpenAIWrapper(IdentityWrapper):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        # separate pointmap and color latents
        x_pointmap = torch.cat((x[:, :FEATURE_DIM], c.get("concat", torch.Tensor([]).type_as(x))[:, :FEATURE_DIM]), dim=1)
        x_color = torch.cat((x[:, FEATURE_DIM:], c.get("concat", torch.Tensor([]).type_as(x))[:, FEATURE_DIM:]), dim=1)

        num_video_frames = kwargs.get("num_video_frames", 0)
        num_chunks = x_pointmap.shape[0] // (2 * num_video_frames)
        x_pointmap_chunks = torch.chunk(x_pointmap, chunks=num_chunks, dim=0)
        x_color_chunks = torch.chunk(x_color, chunks=num_chunks, dim=0)
        t_chunks = torch.chunk(t, chunks=num_chunks, dim=0)

        kwargs['image_only_indicator'] = kwargs['image_only_indicator'][0].unsqueeze(0)

        output_list = []
        for i in range(num_chunks):
            x1_pointmap, hd = self.diffusion_model(
                x_pointmap_chunks[i][:num_video_frames],
                timesteps=t_chunks[i][:num_video_frames],
                context=c.get("crossattn", None)[2 * i].unsqueeze(0),
                y=c.get("vector", None), # .reshape(x.shape[0], -1),
                **kwargs,
            )
            
            x1_color, _ = self.diffusion_model(
                x_color_chunks[i][:num_video_frames],
                timesteps=t_chunks[i][:num_video_frames],
                context=c.get("crossattn", None)[2 * i + 1].unsqueeze(0),
                y=c.get("vector", None), # .reshape(x.shape[0], -1),
                **kwargs,
            )
            
            x2_pointmap, _ = self.diffusion_model_2(
                x_pointmap_chunks[i][num_video_frames:],
                timesteps=t_chunks[i][num_video_frames:],
                context=c.get("crossattn", None)[2 * i].unsqueeze(0),
                spatial_context=hd,
                y=c.get("vector", None), # .reshape(x.shape[0], -1),
                **kwargs,
            )
            
            x2_color, _ = self.diffusion_model_2(
                x_color_chunks[i][num_video_frames:],
                timesteps=t_chunks[i][num_video_frames:],
                context=c.get("crossattn", None)[2 * i + 1].unsqueeze(0),
                y=c.get("vector", None), # .reshape(x.shape[0], -1),
                **kwargs,
            )
            
            x1 = torch.cat([x1_pointmap, x1_color], dim=1)
            x2 = torch.cat([x2_pointmap, x2_color], dim=1)
            
            output_list.append(torch.cat((x1, x2), dim=0))

        return torch.cat(output_list, dim=0)