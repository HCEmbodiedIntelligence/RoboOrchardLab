# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Qwen HoloBrain training configuration dedicated to the hc_tj dataset."""

import copy
import logging
import math

from config_holobrain_qwen_common import (
    build_model,
    build_processors,
    build_training_dataset,
    build_validation_dataset,
    config as _common_config,
)

logger = logging.getLogger(__file__)

__all__ = [
    "build_model",
    "build_optimizer",
    "build_processors",
    "build_training_dataset",
    "build_validation_dataset",
    "config",
]

config = copy.deepcopy(_common_config)
config.update(
    training_datasets=["hc_tj_14dof"],
    deploy_datasets=["hc_tj_14dof"],
    batch_size=1,
    step_log_freq=50,
    save_step_freq=5000,
    num_workers=16,
    lr=1e-4,
    max_step=int(1e5),
    warmup_steps=300,
    decay_steps=int(1e5),
    decay_lr=2.5e-6,
)


def cosine_decay_with_warmup_scheduler_build(
    optimizer,
    num_warmup_steps,
    num_decay_steps,
    num_training_steps,
    peak_lr,
    decay_lr,
):
    """Used by Physical Intelligence to train Pi0.

    Automatically scales warmup and decay steps if num_training_steps is
    smaller than num_decay_steps. This ensures the learning rate schedule
    completes properly even with shorter training runs.
    """

    from torch import optim

    # Scale the scheduler when training is shorter than its configured decay.
    actual_warmup_steps = num_warmup_steps
    actual_decay_steps = num_decay_steps

    if num_training_steps < num_decay_steps:
        # Fit the schedule into the available training steps.
        scale_factor = num_training_steps / num_decay_steps
        actual_warmup_steps = int(num_warmup_steps * scale_factor)
        actual_decay_steps = num_training_steps

        logging.info(
            f"Auto-scaling LR scheduler: "
            f"num_training_steps ({num_training_steps}) < "
            f"num_decay_steps ({num_decay_steps}). "
            f"Scaling warmup: {num_warmup_steps} → {actual_warmup_steps}, "
            f"decay: {num_decay_steps} → {actual_decay_steps} "
            f"(scale factor: {scale_factor:.3f})"
        )

    def lr_lambda(current_step):
        def linear_warmup_schedule(current_step):
            if current_step <= 0:
                return 1 / (actual_warmup_steps + 1)
            frac = 1 - current_step / actual_warmup_steps
            return (1 / (actual_warmup_steps + 1) - 1) * frac + 1

        def cosine_decay_schedule(current_step):
            step = min(current_step, actual_decay_steps)
            cosine_decay = 0.5 * (
                1 + math.cos(math.pi * step / actual_decay_steps)
            )
            alpha = decay_lr / peak_lr
            decayed = (1 - alpha) * cosine_decay + alpha
            return decayed

        if current_step < actual_warmup_steps:
            return linear_warmup_schedule(current_step)

        return cosine_decay_schedule(current_step)

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, -1)


def build_optimizer(config, model):
    import torch
    from torch import optim

    base_lr = config["lr"]
    max_step = config["max_step"]
    warmup_steps = config.get("warmup_steps", int(max_step * 0.01))
    decay_steps = config.get("decay_steps", max_step)
    decay_lr = config.get("decay_lr", base_lr * 0.01)

    vlm_params = []
    bit16_params = []
    other_params = []
    for name, p in model.named_parameters():
        if "vlm." in name:
            if p.requires_grad:
                vlm_params.append(p)
        elif p.dtype == torch.float16 or p.dtype == torch.bfloat16:
            bit16_params.append(p)
        else:
            other_params.append(p)
    optim_params = [
        {"params": bit16_params},
        {"params": other_params},
    ]
    if len(vlm_params) > 0:
        optim_params.append(
            {"params": vlm_params, "lr": base_lr * 0.1},
        )
    optimizer = optim.AdamW(
        optim_params,
        lr=base_lr,
        weight_decay=config.get("weight_decay", 0.0005),
    )
    lr_scheduler = cosine_decay_with_warmup_scheduler_build(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_decay_steps=decay_steps,
        num_training_steps=max_step,
        peak_lr=base_lr,
        decay_lr=decay_lr,
    )
    return optimizer, lr_scheduler
