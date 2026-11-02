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

"""Explicit X1 model-to-public mapping for optional HC Motion ServoJ."""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["MotionInterface"]


@dataclass(frozen=True, slots=True)
class MotionInterface:
    """Validated X1 capability snapshot using an explicitly selected mapping.

    ``identity_mapping=True`` acknowledges that the selected X1 profile uses
    no public/model scale or offset overrides. Robot info does not expose
    those overrides; callers must not infer identity from matching dimensions.
    Hands and Cartesian/Move actions are capabilities only, never synthesized
    from the HoloBrain joint action vector.
    """

    revision: str
    joint_names: tuple[str, ...]
    topics: tuple[str, str]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    servo_lease_s: float
    capabilities: dict[str, dict[str, str]]

    @classmethod
    def from_info(
        cls, payload: dict[str, Any], *, identity_mapping: bool = False
    ) -> MotionInterface:
        """Validate public X1 names, units and ServoJ capabilities.

        Raises:
            ValueError: The mapping is unacknowledged or incompatible.
        """
        if not identity_mapping:
            raise ValueError(
                "Motion requires explicit identity mapping opt-in"
            )
        try:
            if (
                payload["schema"] != "hc-robot-api/v1"
                or payload["profile_id"] != "x1"
            ):
                raise ValueError(
                    "Motion mapping supports the verified X1 profile only"
                )
            revision = payload["revision"]
            if not isinstance(revision, str) or not revision:
                raise ValueError("robot info must contain a profile revision")
            all_names, lower, upper, topics, capabilities = [], [], [], [], {}
            for group, prefix, count in (
                ("right_arm", "right_joint_", 7),
                ("left_arm", "left_joint_", 7),
                ("waist", "waist_joint_", 3),
            ):
                spec = payload["groups"][group]
                names = [f"{prefix}{i}" for i in range(1, count + 1)]
                if (
                    spec["joint_names"] != names
                    or spec["units"] != ["rad"] * count
                ):
                    raise ValueError(
                        f"incompatible Motion {group} names or units"
                    )
                all_names.extend(names)
                endpoints = spec["endpoints"]
                if not isinstance(endpoints, dict) or any(
                    not isinstance(v, str) or not v.startswith("/")
                    for v in endpoints.values()
                ):
                    raise ValueError("invalid Motion endpoint")
                capabilities[group] = dict(endpoints)
                if group == "waist":
                    continue
                topics.append(endpoints["servo_j"])
                limits = spec["limits"]
                if len(limits) != count:
                    raise ValueError("Motion limit count differs from joints")
                for limit in limits:
                    lo, hi = float(limit["lower"]), float(limit["upper"])
                    if (
                        not math.isfinite(lo)
                        or not math.isfinite(hi)
                        or lo > hi
                    ):
                        raise ValueError("invalid Motion joint limits")
                    lower.append(lo)
                    upper.append(hi)
            lease = float(payload["servo"]["servo_lease_ms"]) / 1000
            if (
                not math.isfinite(lease)
                or lease <= 0
                or topics[0] == topics[1]
            ):
                raise ValueError(
                    "invalid Motion servo lease or duplicate endpoints"
                )
            return cls(
                revision,
                tuple(all_names),
                tuple(topics),
                tuple(lower),
                tuple(upper),
                lease,
                capabilities,
            )
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError(
                "incomplete HC Motion capability document"
            ) from error

    def feedback(self, names: list[str], positions: list[float]) -> np.ndarray:
        """Map named SI feedback to both arms followed by three waist axes."""
        values = np.asarray(positions, dtype=np.float64)
        if (
            values.ndim != 1
            or len(names) != len(values)
            or len(set(names)) != len(names)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("invalid named Motion joint feedback")
        indexed = dict(zip(names, values, strict=True))
        if not set(self.joint_names) <= indexed.keys():
            raise ValueError(
                "Motion feedback requires both arms and measured waist"
            )
        return np.asarray([indexed[name] for name in self.joint_names])

    def commands(
        self, arms: np.ndarray
    ) -> tuple[tuple[str, tuple[str, ...], list[float]], ...]:
        """Split canonical 14 arm positions into two public ServoJ messages."""
        values = np.asarray(arms, dtype=np.float64)
        if (
            values.shape != (14,)
            or not np.all(np.isfinite(values))
            or np.any(values < self.lower)
            or np.any(values > self.upper)
        ):
            raise ValueError("Motion arm target is invalid or outside limits")
        return tuple(
            (
                topic,
                self.joint_names[offset : offset + 7],
                values[offset : offset + 7].tolist(),
            )
            for topic, offset in zip(self.topics, (0, 7), strict=True)
        )
