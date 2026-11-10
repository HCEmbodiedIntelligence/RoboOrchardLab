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
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from robo_orchard_lab.models.holobrain import structure


@pytest.fixture()
def model(monkeypatch):
    torch.manual_seed(7)
    config = Qwen2_5_VLConfig(
        text_config=dict(
            vocab_size=32,
            hidden_size=24,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=3,
            num_key_value_heads=1,
            rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
        ),
        vision_config=dict(
            depth=1,
            hidden_size=24,
            intermediate_size=32,
            num_heads=3,
            out_hidden_size=24,
        ),
        image_token_id=29,
        video_token_id=28,
        vision_start_token_id=27,
    )
    vlm = Qwen2_5_VLForConditionalGeneration(config)
    monkeypatch.setattr(
        structure.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        lambda *args, **kwargs: vlm,
    )
    monkeypatch.setattr(
        structure.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            tokenizer=SimpleNamespace(pad_token_id=0)
        ),
    )
    monkeypatch.setattr(
        structure,
        "build",
        lambda cfg: _Decoder() if cfg is not None else None,
    )
    cfg = structure.HoloBrain_Qwen2_5_VLConfig(
        vlm_pretrain="unused",
        decoder={},
    )
    return structure.HoloBrain_Qwen2_5_VL(cfg).float().eval()


class _Decoder(nn.Module):
    embed_dims = 8

    def loss(self, outputs, inputs, **kwargs):
        return {"loss_action": torch.tensor(0.0, requires_grad=True)}

    def post_process(self, outputs, inputs, **kwargs):
        return [{} for _ in outputs["task_progress"]]


class TestHoloBrainProgress:
    def test_multimodal_action_features_exclude_query(self, model):
        ids = torch.tensor([[0, 4, 27, 29, 26, 5]])
        vision = model.vlm.config.vision_config
        pixels = torch.randn(
            4,
            vision.in_channels
            * vision.temporal_patch_size
            * vision.patch_size**2,
        )
        output = model._forward_vlm(
            input_ids=ids,
            attention_mask=ids != 0,
            pixel_values=pixels,
            image_grid_thw=torch.tensor([[1, 2, 2]]),
        )
        size = model.qwen_patch_size
        features, text = model._vlm_outputs_handler(
            output,
            {"input_ids": ids},
            {"imgs": torch.zeros(1, 1, 3, size, size)},
        )
        assert features[0].shape == (1, 1, model.decoder.embed_dims, 1, 1)
        assert text["embedded"].shape == (1, 5, model.decoder.embed_dims)
        assert text["text_token_mask"].tolist() == [
            [False, True, True, True, True]
        ]

    def test_cot_query_keeps_sequence_alignment_and_gradients(
        self, model, monkeypatch
    ):
        sequences = torch.tensor([[0, 4, 5, 6, 7], [8, 9, 5, 0, 0]])
        monkeypatch.setattr(
            model.vlm,
            "generate",
            lambda **kwargs: SimpleNamespace(sequences=sequences),
        )
        output = model._generate_vlm(
            {
                "input_ids": sequences[:, :2],
                "attention_mask": sequences[:, :2] != 0,
            }
        )
        assert output.sequences is sequences
        assert all(h.shape[1] == 4 for h in output.hidden_states)
        direct = model._forward_vlm(
            input_ids=sequences[:, :-1],
            attention_mask=sequences[:, :-1] != 0,
        )
        torch.testing.assert_close(output.task_progress, direct.task_progress)
        output.task_progress.sum().backward()
        assert model.vlm_overall_query.grad.abs().sum() > 0

    def test_query_isolation_padding_and_context(self, model):
        ids = torch.tensor([[0, 0, 4, 5], [0, 6, 7, 8]])
        mask = ids != 0
        first = model._forward_vlm(input_ids=ids, attention_mask=mask)
        assert first.task_progress.shape == (2,)
        assert ((first.task_progress > 0) & (first.task_progress < 1)).all()
        assert all(h.shape[1] == ids.shape[1] for h in first.hidden_states)
        with torch.no_grad():
            model.vlm_overall_query.add_(
                torch.randn_like(model.vlm_overall_query)
            )
        changed = model._forward_vlm(input_ids=ids, attention_mask=mask)
        for before, after in zip(
            first.hidden_states, changed.hidden_states, strict=True
        ):
            torch.testing.assert_close(before[mask], after[mask])
        assert not torch.allclose(first.task_progress, changed.task_progress)
        padded_ids = ids.clone()
        padded_ids[~mask] = 10
        padded = model._forward_vlm(input_ids=padded_ids, attention_mask=mask)
        torch.testing.assert_close(changed.task_progress, padded.task_progress)
        unpadded = model._forward_vlm(input_ids=ids[:1, 2:])
        torch.testing.assert_close(
            changed.task_progress[:1], unpadded.task_progress
        )
        context_ids = ids.clone()
        context_ids[mask] = 12
        context = model._forward_vlm(
            input_ids=context_ids, attention_mask=mask
        )
        assert not torch.allclose(changed.task_progress, context.task_progress)

    def test_frozen_vlm_query_gradients_and_state_dict(self, model):
        output = model._forward_vlm(input_ids=torch.tensor([[4, 5, 6]]))
        nn.functional.mse_loss(output.task_progress, torch.ones(1)).backward()
        assert model.vlm_overall_query.grad.abs().sum() > 0
        assert model.progress_head[0].weight.grad.abs().sum() > 0
        assert model.progress_head[2].weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in model.vlm.parameters())
        state = model.state_dict()
        assert "vlm_overall_query" in state
        assert "progress_head.0.weight" in state
        assert not any(k.startswith("vlm.") for k in state)

    def test_unlabeled_loss_keeps_head_in_backward_graph(
        self, model, monkeypatch
    ):
        output = model._forward_vlm(input_ids=torch.tensor([[4, 5, 6]]))
        monkeypatch.setattr(
            model, "_forward", lambda inputs: (output, None, {}, None)
        )
        losses = model.loss({})
        assert losses["loss_task_progress"].item() == 0
        sum(losses.values()).backward()
        for parameter in model.progress_head.parameters():
            assert parameter.grad is not None
            assert torch.count_nonzero(parameter.grad) == 0

    @pytest.mark.parametrize("column", [False, True])
    def test_loss_and_prediction_contract(self, model, monkeypatch, column):
        progress = torch.tensor([0.2, 0.7], requires_grad=True)
        outputs = {"task_progress": progress}
        monkeypatch.setattr(
            model,
            "_forward",
            lambda inputs: (
                (outputs, None, {}, None)
                if model.training
                else (outputs, None, {})
            ),
        )
        model.train()
        target = torch.tensor([0.0, 1.0])
        losses = model.loss(
            {"task_progress": target[:, None] if column else target}
        )
        torch.testing.assert_close(
            losses["loss_task_progress"], torch.tensor(0.065)
        )
        losses["loss_task_progress"].backward()
        assert progress.grad is not None
        assert model.loss({})["loss_task_progress"].item() == 0
        model.eval()
        results = model.predict({})
        torch.testing.assert_close(
            torch.stack([r["task_progress"] for r in results]), progress
        )

    @pytest.mark.parametrize(
        "target", [[0.2], [[0.1, 0.2]], [float("nan"), 0.5], [-0.1, 1.1]]
    )
    def test_invalid_targets(self, model, monkeypatch, target):
        monkeypatch.setattr(
            model,
            "_forward",
            lambda inputs: (
                {"task_progress": torch.tensor([0.2, 0.7])},
                None,
                {},
                None,
            ),
        )
        with pytest.raises(ValueError, match="task_progress"):
            model.loss({"task_progress": target})
