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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only HTTP client tests for independent inference and action fencing."""

from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from projects.holobrain.scripts import hc_tj_http_async_client as client
from projects.holobrain.scripts.hc_tj_async_client import (
    HcTjAsyncClientConfig,
    _post_inference,
)


def _config():
    return client.HcTjHttpAsyncClientConfig(
        robot_url="http://robot:8765",
        robot_token="0123456789abcdef",
        server_url="http://gpu:2000/holobrain/hc_tj_async/v2",
        instruction="test",
        depth_units_per_meter=1000,
        enable_dagger=True,
        enable_control=True,
    )


@pytest.mark.parametrize("allowed", [False, None, 1, "true"])
def test_decoder_never_promotes_preview_or_ambiguous_permission(allowed):
    session = Mock()
    form = {"control_window": "window", "control_server": "server"}
    session.post.return_value.json.return_value = {
        **form,
        "status": "ok",
        "actions_allowed": allowed,
        "actions": "must not be decoded for execution",
    }
    actions, _ = client._request_inference(session, _config(), {}, form)
    assert actions is None
    assert session.post.call_args.kwargs["headers"] == {
        "Authorization": "Bearer 0123456789abcdef"
    }


def test_direct_client_authenticates_and_discards_preview(
    monkeypatch, tmp_path
):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("<robot/>")
    config = HcTjAsyncClientConfig(
        server_url="http://gpu:2000/holobrain/hc_tj_async/v2",
        instruction="test",
        urdf_path=urdf,
        depth_units_per_meter=1000,
        enable_dagger=True,
        control_token="0123456789abcdef",
    )
    response = Mock()
    response.json.return_value = {
        "status": "ok",
        "actions_allowed": False,
        "actions": "must not be executed",
    }
    post = Mock(return_value=response)
    monkeypatch.setattr(client.requests, "post", post)
    actions, _ = _post_inference(config, {}, {})
    assert actions is None
    assert post.call_args.kwargs["headers"] == {
        "Authorization": "Bearer 0123456789abcdef"
    }


def _run_loop(
    monkeypatch,
    *,
    allowed=False,
    hands_fresh=True,
    hand_values=(0.2, 0.3),
    fresh_observation=True,
    on_tick=None,
    on_observation=None,
    hand_inputs=None,
    inference_actions=None,
    send_error=None,
    invalidations=None,
    ticks=3,
    block_first=False,
):
    """Drive the real loop deterministically without threads or networking."""
    state = {
        "allowed": allowed,
        "window_id": "window-1",
        "hand_fresh": hands_fresh,
        "hand_closures": hand_values,
        "observation_floor": [0, 0, 0]
        if fresh_observation
        else [100, 100, 100],
    }
    observation = SimpleNamespace(
        observation_id=1,
        color_timestamp_ns=10,
        depth_timestamp_ns=10,
        joint_timestamp_ns=10,
    )
    sends, submissions, remaining_inputs = [], [], []
    relay_instance = None
    clock = [0.0]

    class Session:
        def get(self, *_args, **_kwargs):
            return SimpleNamespace(
                raise_for_status=lambda: None, json=lambda: dict(state)
            )

        def close(self):
            pass

    class Relay:
        server_instance = "server"

        def __init__(self, get_snapshot, *_args):
            nonlocal relay_instance
            relay_instance = self
            self.sync = get_snapshot

        def start(self):
            self.sync()

        def ready(self, window):
            return state["allowed"] and state["window_id"] == window

        def close(self):
            pass

    class Executor:
        def __init__(self, *, max_workers):
            assert max_workers == 1

        def submit(self, fn, *args):
            future = Future()
            submissions.append((future, fn, args))
            if not block_first or len(submissions) > 1:
                future.set_result(fn(*args))
            return future

        def shutdown(self, **_kwargs):
            pass

    def build_request(_packet, _config, right, left, remaining, _delay):
        if hand_inputs is not None:
            hand_inputs.append((right, left))
        remaining_inputs.append(remaining)
        return {}, {}, None

    tick = 0
    observation_count = 0

    def request_observation(*_args):
        nonlocal observation_count
        observation_count += 1
        observation.observation_id = observation_count
        observation.color_timestamp_ns = observation_count * 10
        observation.depth_timestamp_ns = observation_count * 10
        observation.joint_timestamp_ns = observation_count * 10
        if on_observation is not None:
            on_observation(observation_count, state)
            relay_instance.sync()
        return observation

    def sleep(_seconds):
        nonlocal tick
        tick += 1
        clock[0] += 0.01
        if on_tick is not None:
            on_tick(tick, state, submissions)
        relay_instance.sync()
        if tick >= ticks:
            raise StopIteration

    def invalidate(_url, *, json, **_kwargs):
        if json["reason"] == "new HTTP client session":
            return SimpleNamespace(raise_for_status=lambda: None)
        if invalidations is not None:
            invalidations.append(json)
        state.update(
            allowed=False,
            window_id="window-2",
            observation_floor=[(observation_count + 1) * 10] * 3,
        )
        return SimpleNamespace(raise_for_status=lambda: None)

    monkeypatch.setattr(client.requests, "Session", Session)
    monkeypatch.setattr(client.requests, "get", Session().get)
    monkeypatch.setattr(client.requests, "post", invalidate)
    monkeypatch.setattr(client, "ControlRelay", Relay)
    monkeypatch.setattr(client, "ThreadPoolExecutor", Executor)
    monkeypatch.setattr(
        client,
        "_check_gateway",
        lambda *_: {"dagger_enabled": True, "hands_enabled": True},
    )
    monkeypatch.setattr(client, "_request_observation", request_observation)
    monkeypatch.setattr(client, "packet_to_snapshot", lambda packet: packet)
    monkeypatch.setattr(client, "_build_request", build_request)
    # Return an apparently executable result to exercise the client's own
    # admission fence independently of the service's actions_allowed fence.
    actions = np.zeros((64, 16), dtype=np.float32)
    actions[:, 7], actions[:, 15] = 0.4, 0.6
    monkeypatch.setattr(
        client,
        "_request_inference",
        lambda *_: (
            actions.copy()
            if inference_actions is None
            else inference_actions(len(submissions), actions.copy()),
            0.01,
        ),
    )
    monkeypatch.setattr(
        client,
        "_request_joint_feedback",
        lambda *_: SimpleNamespace(joint_positions=np.zeros(17)),
    )
    send_attempts = 0

    def send(*args):
        nonlocal send_attempts
        send_attempts += 1
        if send_error is not None:
            error = send_error(send_attempts)
            if error is not None:
                raise error
        sends.append(args)

    monkeypatch.setattr(client, "_send_action_chunk", send)
    monkeypatch.setattr(client.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(client.time, "sleep", sleep)
    with pytest.raises(StopIteration):
        client._run_controlled_client(_config())
    return submissions, sends, remaining_inputs


@pytest.mark.parametrize(
    "conditions",
    [
        {"allowed": False},
        {"allowed": True, "fresh_observation": False},
    ],
)
def test_waiting_for_permission_or_feedback_still_computes(
    monkeypatch, capsys, conditions
):
    submitted, sent, history = _run_loop(monkeypatch, **conditions)
    assert len(submitted) == 3
    assert not sent
    assert all(value is None for value in history)
    output = capsys.readouterr().out
    assert "inference #2: 10.0 ms" in output
    assert "actions=waiting" in output


@pytest.mark.parametrize("initially_allowed", [False, True])
def test_permission_changes_during_inference_do_not_send_old_result(
    monkeypatch, initially_allowed
):
    def change_permission(tick, state, _submissions):
        if tick == 1:
            state["allowed"] = True
            if initially_allowed:
                # Revocation and a later grant have a different window.
                state["window_id"] = "window-2"

    submitted, sent, _ = _run_loop(
        monkeypatch,
        allowed=initially_allowed,
        on_tick=change_permission,
        ticks=2,
    )
    assert len(submitted) == 2
    assert not sent


def test_new_authorized_observation_sends_after_preview(monkeypatch):
    def grant(tick, state, _submissions):
        if tick == 1:
            state["allowed"] = True

    _, sent, history = _run_loop(monkeypatch, on_tick=grant, ticks=4)
    assert len(sent) == 2
    assert [args[2] for args in sent] == [1, 2]
    assert all(args[-1] == "window-1" for args in sent)
    assert history[0] is None and history[1] is None
    assert history[2] is not None


def test_hand_feedback_loss_does_not_block_authorized_send(monkeypatch):
    def lose_hands(tick, state, _submissions):
        if tick == 1:
            state["hand_fresh"] = False

    submitted, sent, _ = _run_loop(
        monkeypatch, allowed=True, on_tick=lose_hands
    )
    assert len(submitted) == 3
    assert len(sent) == 2


@pytest.mark.parametrize("measured", [None, (0.7, 0.8)])
def test_hand_input_matches_v1_command_estimates_without_feedback(
    monkeypatch, measured
):
    inputs = []
    submitted, sent, _ = _run_loop(
        monkeypatch,
        allowed=True,
        hands_fresh=False,
        hand_values=measured,
        hand_inputs=inputs,
    )
    assert len(submitted) == 3
    assert len(sent) == 2
    assert inputs[0] == (0.0, 0.0)
    np.testing.assert_allclose(inputs[1:], [(0.4, 0.6), (0.4, 0.6)])


def test_blocking_observation_uses_new_grant_observation_floor(monkeypatch):
    def new_grant(_count, state):
        state["allowed"] = True
        state["window_id"] = "window-2"
        state["observation_floor"] = [100, 100, 100]

    submitted, sent, _ = _run_loop(monkeypatch, on_observation=new_grant)
    assert len(submitted) == 3
    assert all(
        args[-1]["control_window"] == "window-2" for _, _, args in submitted
    )
    assert not sent


def test_suppressed_inference_has_one_inflight_and_no_backlog(monkeypatch):
    def complete(tick, _state, submissions):
        assert len(submissions) == 1
        if tick == 3:
            future, fn, args = submissions[0]
            future.set_result(fn(*args))

    submitted, sent, _ = _run_loop(
        monkeypatch, on_tick=complete, ticks=3, block_first=True
    )
    assert len(submitted) == 1
    assert not sent


@pytest.mark.parametrize("rejected_by", ["client", "gateway"])
@pytest.mark.parametrize("allow_again", [False, True])
def test_discontinuity_clears_old_trajectory_and_replans(
    monkeypatch, capsys, rejected_by, allow_again
):
    invalidations = []

    def inference(count, actions):
        if rejected_by == "client" and count == 2:
            actions[:, 0] = 0.321770
        return actions

    def send_error(count):
        if rejected_by == "gateway" and count == 2:
            return client.ActionDiscontinuityError("first action jumps")
        return None

    def grant(tick, state, _submissions):
        if tick == 3 and allow_again:
            state["allowed"] = True

    submitted, sent, history = _run_loop(
        monkeypatch,
        allowed=True,
        inference_actions=inference,
        send_error=send_error,
        invalidations=invalidations,
        on_tick=grant,
        ticks=5,
    )
    assert len(submitted) == 5
    assert len(invalidations) == 1
    assert invalidations[0]["invalidate"] is True
    assert history[1] is not None  # A prior accepted continuation existed.
    assert history[2] is None and history[3] is None
    assert [(args[2], args[-1]) for args in sent] == (
        [(1, "window-1"), (1, "window-2")]
        if allow_again
        else [(1, "window-1")]
    )
    assert "actions=discarded; replanning:" in capsys.readouterr().out


def test_invalid_model_values_are_not_treated_as_replannable(monkeypatch):
    invalidations = []

    def inference(_count, actions):
        actions[:, 0] = np.nan
        return actions

    with pytest.raises(ValueError, match="finite"):
        _run_loop(
            monkeypatch,
            allowed=True,
            inference_actions=inference,
            invalidations=invalidations,
        )
    assert not invalidations
