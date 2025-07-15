import copy

import torch
from torch.nn.modules.batchnorm import _BatchNorm


class EMABaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def step(self, new_model):
        raise NotImplementedError()

    @property
    def model(self):
        raise NotImplementedError()


class VanillaEMAModel(EMABaseModel):
    def __init__(
        self,
        model,
        alpha,
        use_buffers=True,
    ):
        """
        Args:
           model: The model to take an average of.
           alpha: The factor used to weight the averaged model in each update.
               So alpha = 0.99 would mean 0.99 weight on the averaged model
               and (1 - 0.99) weight on the updated model.
           use_buffers: Whether to also average pytorch buffers in the model
               in addition to parameters. This is set to True by default in
               order to properly handle batch norm global statistics.
        """
        super().__init__()
        self._alpah = alpha
        assert alpha > 0.0 and alpha < 1, f"Invalid EMA alpha: {alpha}"

        ema_func = (
            lambda ema_param, model_param, num_averaged: alpha * ema_param
            + (1 - alpha) * model_param
        )

        # AveragedModel does a deepcopy of model.
        self.averaged_model = torch.optim.swa_utils.AveragedModel(
            model,
            # this should be ema = alpha * ema + (1 - alpha) * model
            avg_fn=ema_func,
            use_buffers=use_buffers,
        )
        self.averaged_model.module.eval()
        self.averaged_model.module.requires_grad_(False)

    @torch.no_grad()
    def step(self, new_model):
        self.averaged_model.update_parameters(new_model)

    @property
    def model(self):
        return self.averaged_model.module


class EMAModel(EMABaseModel):
    """
    Exponential Moving Average of models weights
    """

    def __init__(
        self,
        model,
        update_after_step=0,
        inv_gamma=1.0,
        power=2 / 3,
        min_value=0.0,
        max_value=0.9999,
    ):
        """
        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1,
            power=2/3 are good values for models you plan to train for a
            million or more steps (reaches decay factor 0.999 at 31.6K steps,
            0.9999 at 1M steps), gamma=1, power=3/4 for models you plan to
            train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
            215.4k steps).
        Args:
            inv_gamma (float): Inverse multiplicative factor of EMA warmup.
                Default: 1.
            power (float): Exponential factor of EMA warmup. Default: 2/3.
            min_value (float): The minimum EMA decay rate. Default: 0.
        """

        super().__init__()
        self.averaged_model = copy.deepcopy(model)
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        self.register_buffer(
            "optimization_step", torch.zeros(1, dtype=torch.long)
        )

    @property
    def model(self):
        return self.averaged_model

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power

        if step <= 0:
            return 0.0

        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        decay = self.get_decay(self.optimization_step.item())

        # old_all_dataptrs = set()
        # for param in new_model.parameters():
        #     data_ptr = param.data_ptr()
        #     if data_ptr != 0:
        #         old_all_dataptrs.add(data_ptr)

        all_dataptrs = set()
        for module, ema_module in zip(
            new_model.modules(), self.averaged_model.modules()
        ):
            if isinstance(module, _BatchNorm):
                assert isinstance(ema_module, _BatchNorm)
                ema_module.running_mean.mul_(decay)
                ema_module.running_mean.add_(
                    module.running_mean, alpha=1 - decay
                )

                ema_module.running_var.mul_(decay)
                ema_module.running_var.add_(
                    module.running_var, alpha=1 - decay
                )

            for param, ema_param in zip(
                module.parameters(recurse=False),
                ema_module.parameters(recurse=False),
            ):
                # iterative over immediate parameters only.
                if isinstance(param, dict):
                    raise RuntimeError("Dict parameter not supported")

                # data_ptr = param.data_ptr()
                # if data_ptr != 0:
                #     all_dataptrs.add(data_ptr)

                if not param.requires_grad:
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    ema_param.mul_(decay)
                    ema_param.add_(
                        param.data.to(dtype=ema_param.dtype), alpha=1 - decay
                    )

        # verify that iterating over module and then parameters is identical to
        # parameters recursively.
        # assert old_all_dataptrs == all_dataptrs
        self.optimization_step += 1
