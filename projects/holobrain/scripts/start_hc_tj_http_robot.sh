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

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "用法: $0"
  echo "启动 HC_X1（腰部只读、双臂+双手）和 HoloBrain HTTP 网关。"
  exit 0
fi
if (( $# != 0 )); then
  echo "[robot-launcher] 不支持位置参数；使用 --help 查看用法。" >&2
  exit 2
fi

LAUNCHER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${LAUNCHER_DIR}/hc_tj_http_robot_gateway.py" ]]; then
  GATEWAY_SCRIPT_DIR="${LAUNCHER_DIR}"
  GATEWAY_ROOT="$(cd -- "${LAUNCHER_DIR}/../../.." && pwd)"
elif [[ -f "${LAUNCHER_DIR}/projects/holobrain/scripts/hc_tj_http_robot_gateway.py" ]]; then
  GATEWAY_ROOT="${LAUNCHER_DIR}"
  GATEWAY_SCRIPT_DIR="${GATEWAY_ROOT}/projects/holobrain/scripts"
else
  echo "[robot-launcher] 找不到 hc_tj_http_robot_gateway.py。" >&2
  exit 2
fi

HC_X1_DIR="${HC_X1_DIR:-${HOME}/HC_X1}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-13}"
GATEWAY_PORT="${HC_TJ_GATEWAY_PORT:-8765}"
TOKEN_FILE="${HC_TJ_GATEWAY_TOKEN_FILE:-${HOME}/.config/holobrain/hc_tj_gateway_token}"
URDF_PATH="${HC_TJ_URDF_PATH:-${GATEWAY_ROOT}/hc_tj_robot.urdf}"
HC_X1_PID=""
GATEWAY_PID=""

load_token() {
  if [[ -f "${TOKEN_FILE}" ]]; then
    IFS= read -r HC_TJ_GATEWAY_TOKEN <"${TOKEN_FILE}"
  fi
  if [[ -z "${HC_TJ_GATEWAY_TOKEN:-}" ]]; then
    HC_TJ_GATEWAY_TOKEN="$(/usr/bin/python3 -c \
      'import secrets; print(secrets.token_hex(32))')"
    mkdir -p -- "$(dirname -- "${TOKEN_FILE}")"
    umask 077
    printf '%s\n' "${HC_TJ_GATEWAY_TOKEN}" >"${TOKEN_FILE}"
    echo "[robot-launcher] 已自动生成并保存网关 TOKEN。"
  fi
  if (( ${#HC_TJ_GATEWAY_TOKEN} < 16 )); then
    echo "[robot-launcher] TOKEN 必须至少包含 16 个字符。" >&2
    exit 2
  fi
  export HC_TJ_GATEWAY_TOKEN
}

stop_pid() {
  local pid="$1"
  [[ -n "${pid}" ]] || return 0
  if kill -0 "${pid}" 2>/dev/null; then
    kill -TERM "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  echo "[robot-launcher] 正在停止 HTTP 网关和 HC_X1..."
  stop_pid "${GATEWAY_PID}"
  stop_pid "${HC_X1_PID}"
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ ! -x "${HC_X1_DIR}/start.sh" ]]; then
  echo "[robot-launcher] 找不到可执行的 ${HC_X1_DIR}/start.sh。" >&2
  exit 2
fi
if [[ ! -f "${URDF_PATH}" ]]; then
  echo "[robot-launcher] 找不到 ${URDF_PATH}。" >&2
  exit 2
fi
if ss -ltnH "sport = :${GATEWAY_PORT}" | grep -q .; then
  echo "[robot-launcher] 端口 ${GATEWAY_PORT} 已被占用，请先停止旧网关。" >&2
  exit 2
fi

set +u
source /opt/ros/humble/setup.bash
set -u
export ROS_DOMAIN_ID
export PYTHONPATH="${GATEWAY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
load_token

echo "[robot-launcher] 预先获取 sudo 凭据，用于 OmniHand UART/pinmux 检查..."
sudo -v

echo "[robot-launcher] 启动 HC_X1：腰部只读，双臂和双手启用..."
(
  cd -- "${HC_X1_DIR}"
  exec ./start.sh waist_control:=false
) &
HC_X1_PID=$!

echo "[robot-launcher] 等待 HC_X1 ROS 数据就绪..."
deadline=$((SECONDS + 90))
while true; do
  if ! kill -0 "${HC_X1_PID}" 2>/dev/null; then
    wait "${HC_X1_PID}" || true
    echo "[robot-launcher] HC_X1 在 ROS 数据就绪前退出。" >&2
    exit 1
  fi
  topics="$(ros2 topic list 2>/dev/null || true)"
  if grep -Fxq "/hc_teleop/joint_states" <<<"${topics}" \
    && grep -Fxq "/hc_teleop/camera_head/color/compressed" <<<"${topics}" \
    && grep -Fxq "/hc_teleop/camera_head/depth/compressed" <<<"${topics}"; then
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "[robot-launcher] 等待 HC_X1 ROS 话题超时。" >&2
    exit 1
  fi
  sleep 1
done

echo "[robot-launcher] 启动双臂+双手 HTTP 网关，腰部不进入命令..."
/usr/bin/python3 "${GATEWAY_SCRIPT_DIR}/hc_tj_http_robot_gateway.py" \
  --host 0.0.0.0 \
  --port "${GATEWAY_PORT}" \
  --enable-control \
  --enable-hands \
  --urdf-path "${URDF_PATH}" \
  --first-point-jump-rad "${HC_TJ_FIRST_POINT_JUMP_RAD:-0.3}" \
  --state-timeout-s "${HC_TJ_STATE_TIMEOUT_S:-0.5}" \
  --action-timeout-s "${HC_TJ_ACTION_TIMEOUT_S:-0.75}" \
  --max-arm-velocity-rad-s "${HC_TJ_MAX_ARM_VELOCITY_RAD_S:-0.25}" &
GATEWAY_PID=$!

echo "[robot-launcher] 等待网关观测数据就绪..."
deadline=$((SECONDS + 90))
last_health=""
while true; do
  if ! kill -0 "${GATEWAY_PID}" 2>/dev/null; then
    wait "${GATEWAY_PID}" || true
    echo "[robot-launcher] HTTP 网关在就绪前退出。" >&2
    exit 1
  fi
  last_health="$(curl -sS --max-time 1 "http://127.0.0.1:${GATEWAY_PORT}/healthz" || true)"
  if HC_TJ_HEALTH_PAYLOAD="${last_health}" /usr/bin/python3 - <<'PY'
import json
import os

try:
    health = json.loads(os.environ["HC_TJ_HEALTH_PAYLOAD"])
    observations = health["observations"]
    ready = (
        health["status"] == "ok"
        and health["fault"] is None
        and health["control_enabled"] is True
        and health["hands_enabled"] is True
        and observations["color_frames"] > 0
        and observations["depth_frames"] > 0
        and observations["joint_samples"] > 0
        and observations["calibration"] is True
        and observations["extrinsics"] is True
    )
except (KeyError, TypeError, ValueError, json.JSONDecodeError):
    ready = False
raise SystemExit(0 if ready else 1)
PY
  then
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "[robot-launcher] 等待网关数据超时，最后状态: ${last_health}" >&2
    exit 1
  fi
  sleep 1
done

echo "[robot-launcher] 机器人侧已就绪: http://0.0.0.0:${GATEWAY_PORT}"
echo "[robot-launcher] 腰部只读；双臂和双手控制已开启。按 Ctrl+C 全部停止。"
set +e
wait -n "${HC_X1_PID}" "${GATEWAY_PID}"
status=$?
set -e
exit "${status}"
