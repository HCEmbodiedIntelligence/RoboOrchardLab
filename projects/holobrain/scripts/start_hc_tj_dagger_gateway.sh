#!/usr/bin/env bash
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

set -Eeuo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "用法: $0；在已运行的 HC DAgger 仲裁器旁启动网关，不启动硬件驱动。"
  echo "默认保留 own 的 /hc_teleop 话题。Motion 需显式设置 HC_TJ_ROBOT_INTERFACE=hc_motion、HC_TJ_MOTION_IDENTITY_MAPPING=1、HC_TJ_ENABLE_HANDS=0。"
  exit 0
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
CONDA_BASE="${HC_TJ_CONDA_BASE:-${HOME}/miniconda3}"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${HC_TJ_CONDA_ENV:-hc-teleop-controller}"
source /opt/ros/humble/setup.bash
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-13}"
export PYTHONPATH="${REPOSITORY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
TOKEN_FILE="${HC_TJ_GATEWAY_TOKEN_FILE:-${HOME}/.config/holobrain/hc_tj_gateway_token}"
if [[ -z "${HC_TJ_GATEWAY_TOKEN:-}" && -f "${TOKEN_FILE}" ]]; then
  IFS= read -r HC_TJ_GATEWAY_TOKEN <"${TOKEN_FILE}"
fi
if [[ -z "${HC_TJ_GATEWAY_TOKEN:-}" ]]; then
  mkdir -p -- "$(dirname -- "${TOKEN_FILE}")"
  umask 077
  HC_TJ_GATEWAY_TOKEN="$(python -c 'import secrets; print(secrets.token_hex(32))')"
  printf '%s\n' "${HC_TJ_GATEWAY_TOKEN}" >"${TOKEN_FILE}"
fi
export HC_TJ_GATEWAY_TOKEN
args=(--enable-dagger --enable-control
  --port "${HC_TJ_GATEWAY_PORT:-8765}"
  --urdf-path "${HC_TJ_URDF_PATH:-${REPOSITORY_ROOT}/hc_tj_robot.urdf}"
  --robot-interface "${HC_TJ_ROBOT_INTERFACE:-legacy}"
  --calibration-topic /observation/cameras/head/calibration
  --extrinsics-topic /observation/cameras/head/extrinsics
  --extrinsics-durability volatile
  --state-timeout-s "${HC_TJ_STATE_TIMEOUT_S:-0.5}"
  --first-point-jump-rad "${HC_TJ_FIRST_POINT_JUMP_RAD:-0.3}"
  --max-arm-velocity-rad-s "${HC_TJ_MAX_ARM_VELOCITY_RAD_S:-0.25}")
if [[ "${HC_TJ_ENABLE_HANDS:-1}" == "1" ]]; then args+=(--enable-hands); fi
if [[ "${HC_TJ_MOTION_IDENTITY_MAPPING:-0}" == "1" ]]; then args+=(--motion-identity-mapping); fi
exec python "${SCRIPT_DIR}/hc_tj_http_robot_gateway.py" "${args[@]}" "$@"
