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
    training_datasets=["dev_test"],
    deploy_datasets=["dev_test"],
    batch_size=2,
    step_log_freq=50,
    save_step_freq=5000,
    num_workers=16,
    lr=1e-4,
    max_step=int(1e5),
    warmup_steps=300,
    decay_steps=int(1e5),
    decay_lr=2.5e-6,
    task_progress_loss_weight=0.5,
)
config["async_inference_plugin"] = (
    {
        "type": (
            "robo_orchard_lab.models.rtc_plugin.rtc_plugin:RTCInferencePlugin"
        ),
        "max_horizon": 32,
    },
)  # noqa: E501


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


def build_model(config):
    import copy

    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusers.schedulers.scheduling_dpmsolver_multistep import (
        DPMSolverMultistepScheduler,
    )
    from torch import nn

    from robo_orchard_lab.models.bip3d.spatial_enhancer import (
        BatchDepthProbGTGenerator,
        DepthFusionSpatialEnhancer,
    )
    from robo_orchard_lab.models.holobrain import (
        AdaRMSNorm,
        HoloBrain_Qwen2_5_VL_HC_TaskProgress,
        HoloBrain_Qwen2_5_VL_HC_TaskProgressConfig,
        HoloBrainActionDecoder,
        HoloBrainActionLoss,
        HoloBrainDecoderBaseConfig,
        HoloBrainDecoderTransformerConfig,
        HoloBrainEncoderBaseConfig,
        HoloBrainEncoderTransformerConfig,
        HoloBrainRobotStateEncoder,
        HoloBrainTrainingConfig,
        JointGraphAttention,
        RotaryAttention,
        TemporalJointGraphAttention,
        TextTemplate,
        UpsampleHead,
    )
    from robo_orchard_lab.models.layers.data_preprocessors import (
        BaseDataPreprocessor,
    )
    from robo_orchard_lab.models.layers.transformer_layers import FFN
    from robo_orchard_lab.models.modules.swin_transformer import (
        SwinTransformer,
    )

    patch_size = 28
    model_class = HoloBrain_Qwen2_5_VL_HC_TaskProgress
    model_config = HoloBrain_Qwen2_5_VL_HC_TaskProgressConfig

    embed_dims = config["embed_dims"]
    decoder_norm = nn.RMSNorm

    num_chunk = config["pred_steps"] // config["chunk_size"]
    state_dims = 8  # [joint_angle, x, y, z, qw, qx, qy, qz]
    head = dict(
        type=UpsampleHead,
        upsample_sizes=[num_chunk * 2, config["pred_steps"]],
        input_dim=embed_dims,
        dims=[128, 64],
        norm=dict(type=decoder_norm, normalized_shape=embed_dims),
        act=dict(type=nn.SiLU, inplace=True),
        norm_act_idx=[0, 1],
        num_output_layers=2,
        out_dim=state_dims,
    )
    with_mobile = config.get("with_mobile", False)
    if with_mobile:
        mobile_head = copy.deepcopy(head)
        mobile_head.update(out_dim=2)
    else:
        mobile_head = None

    decoder_operation_order = [
        "t_norm",
        "temp_joint_attn",
        "gate_msa",
        "norm",
        "img_cross_attn",
        "norm",
        "text_cross_attn",
        "norm",
        "scale_shift",
        "ffn",
        "gate_mlp",
    ] * config.get("decoder_layers", 6)

    model = model_class(
        cfg=model_config(
            with_cot=config["with_cot"],
            vlm_pretrain=config["vlm_pretrain"],
            num_vlm_layers=config.get("num_vlm_layers"),
            freeze_vlm=config.get("freeze_vlm", True),
            use_state_dict_with_vlm=not config.get("freeze_vlm", True),
            data_preprocessor=dict(
                type=BaseDataPreprocessor,
                # input image should in BGR convention, it will be converted to RGB here  # noqa: E501
                channel_flip=True,
                unsqueeze_depth_channel=True,
                batch_transforms=[
                    dict(
                        type=BatchDepthProbGTGenerator,
                        min_depth=config["min_depth"],
                        max_depth=config["max_depth"],
                        num_depth=config["num_depth"],
                        origin_stride=2,
                        valid_threshold=0.5,
                        stride=(patch_size,),
                    ),
                    dict(
                        type=TextTemplate,
                        with_subtask=config["training_with_subtask"],
                    ),
                ],
            ),
            backbone_3d=(
                dict(
                    type=SwinTransformer,
                    in_channels=1,
                    embed_dims=32,
                    depths=[2, 6, 2],
                    num_heads=[2, 4, 8],
                    window_size=8,
                    patch_size=patch_size // 4,
                    strides=[patch_size // 4, 2, 2],
                    mlp_ratio=4,
                    qkv_bias=True,
                    qk_scale=None,
                    drop_rate=0.0,
                    attn_drop_rate=0.0,
                    out_indices=(2,),
                    with_cp=True,
                    convert_weights=False,
                )
                if config.get("with_depth")
                else None
            ),
            spatial_enhancer=dict(
                type=DepthFusionSpatialEnhancer,
                embed_dims=embed_dims,
                feature_3d_dim=128,
                num_depth_layers=2,
                min_depth=config["min_depth"],
                max_depth=config["max_depth"],
                num_depth=config["num_depth"],
                with_feature_3d=config.get("with_depth"),
                loss_depth_weight=(
                    config.get("loss_depth_weight", 1.0)
                    if config.get("with_depth_loss")
                    else -1
                ),
            ),
            decoder=dict(
                type=HoloBrainActionDecoder,
                async_inference_plugin=config.get("async_inference_plugin"),
                head=head,
                mobile_head=mobile_head,
                transformer_cfg=HoloBrainDecoderTransformerConfig(
                    img_cross_attn=dict(
                        type=RotaryAttention,
                        embed_dims=embed_dims,
                        num_heads=8,
                        max_position_embeddings=32,
                    ),
                    temp_joint_attn=dict(
                        type=TemporalJointGraphAttention,
                        embed_dims=embed_dims,
                        num_heads=8,
                        max_position_embeddings=32,
                    ),
                    norm_layer=dict(
                        type=decoder_norm,
                        normalized_shape=embed_dims,
                    ),
                    ffn=dict(
                        type=FFN,
                        embed_dims=embed_dims,
                        feedforward_channels=2048,
                        act_cfg=dict(type=nn.SiLU, inplace=True),
                    ),
                    joint_self_attn=dict(
                        type=JointGraphAttention,
                        embed_dims=embed_dims,
                        num_heads=8,
                    ),
                    temp_cross_attn=dict(
                        type=RotaryAttention,
                        embed_dims=embed_dims,
                        num_heads=8,
                        max_position_embeddings=32,
                    ),
                    text_cross_attn=dict(
                        type=RotaryAttention,
                        embed_dims=embed_dims,
                        num_heads=8,
                        max_position_embeddings=512,
                    ),
                    timestep_norm_layer=dict(
                        type=AdaRMSNorm,
                        normalized_shape=embed_dims,
                        condition_dims=256,
                        zero=True,
                    ),
                    operation_order=decoder_operation_order,
                ),
                base_cfg=HoloBrainDecoderBaseConfig(
                    chunk_size=config.get("chunk_size", 8),
                    use_joint_mask=True,
                    noise_type="local_joint",
                    pred_scaled_joint=False,
                    prediction_type=config.get(
                        "prediction_type", "relative_joint_relative_pose"
                    ),
                    pred_steps=config["pred_steps"],
                    state_dims=state_dims,
                    embed_dims=embed_dims,
                    with_mobile=with_mobile,
                    training_noise_scheduler=dict(
                        type=DDPMScheduler,
                        num_train_timesteps=1000,
                        beta_schedule="squaredcos_cap_v2",
                        prediction_type="sample",
                        clip_sample=False,
                    ),
                    test_noise_scheduler=dict(
                        type=DPMSolverMultistepScheduler,
                        num_train_timesteps=1000,
                        beta_schedule="squaredcos_cap_v2",
                        prediction_type="sample",
                    ),
                    num_inference_timesteps=10,
                    feature_level=[0],
                    act_cfg=dict(type=nn.SiLU, inplace=True),
                ),
                training_cfg=HoloBrainTrainingConfig(
                    temporal_attn_drop=0.05,
                    num_parallel_training_sample=4,
                    teacher_forcing_rate=0.02,
                    loss=dict(
                        type=HoloBrainActionLoss,
                        timestep_loss_weight=1000,
                        parallel_loss_weight=0.1,
                        smooth_l1_beta=0.04,
                        loss_mode="smooth_l1",
                    ),
                ),
                robot_encoder=dict(
                    type=HoloBrainRobotStateEncoder,
                    transformer_cfg=HoloBrainEncoderTransformerConfig(
                        joint_self_attn=dict(
                            type=JointGraphAttention,
                            embed_dims=embed_dims,
                            num_heads=8,
                        ),
                        norm_layer=dict(
                            type=decoder_norm,
                            normalized_shape=embed_dims,
                        ),
                        ffn=dict(
                            type=FFN,
                            embed_dims=embed_dims,
                            feedforward_channels=2048,
                            act_cfg=dict(type=nn.SiLU, inplace=True),
                        ),
                        temp_self_attn=dict(
                            type=RotaryAttention,
                            embed_dims=embed_dims,
                            num_heads=8,
                            max_position_embeddings=32,
                        ),
                        operation_order=[
                            "norm",
                            "joint_self_attn",
                            None,
                            None,
                            "norm",
                            "ffn",
                        ]
                        * 4
                        + ["norm"],
                    ),
                    base_cfg=HoloBrainEncoderBaseConfig(
                        embed_dims=embed_dims,
                        chunk_size=min(8, config["hist_steps"]),
                        act_cfg=dict(type=nn.SiLU, inplace=True),
                        state_dims=state_dims,
                    ),
                ),
            ),
        )
    )
    return model
