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
  echo "用法: $0 ['任务指令']"
  echo "启动本机 HoloBrain 推理服务和真机 HTTP 控制客户端。"
  exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${HC_TJ_PYTHON:-${REPOSITORY_ROOT}/.venv/bin/python}"
MODEL_DIR="${HC_TJ_MODEL_DIR:-${REPOSITORY_ROOT}/260823_w/export/model}"
ROBOT_URL="${HC_TJ_ROBOT_URL:-http://192.168.2.33:8765}"
ROBOT_SSH="${HC_TJ_ROBOT_SSH:-niic@192.168.2.33}"
INFERENCE_PORT="${HC_TJ_INFERENCE_PORT:-2000}"
SERVER_URL="http://127.0.0.1:${INFERENCE_PORT}/holobrain/hc_tj_async/v1"
TOKEN_FILE="${HC_TJ_GATEWAY_TOKEN_FILE:-${HOME}/.config/holobrain/hc_tj_gateway_token}"
INSTRUCTION="${*:-${HC_TJ_INSTRUCTION:-Pick up the object in front of the robot.}}"
RUNTIME_DIR="${REPOSITORY_ROOT}/runtime/holobrain_http"
INFERENCE_LOG="${RUNTIME_DIR}/inference_$(date +'%Y%m%d_%H%M%S').log"
INFERENCE_PID=""

fetch_token() {
  mkdir -p -- "$(dirname -- "${TOKEN_FILE}")"
  umask 077
  echo "[local-launcher] 正在从 ${ROBOT_SSH} 自动获取网关 TOKEN..."
  if ! scp -q \
    "${ROBOT_SSH}:.config/holobrain/hc_tj_gateway_token" \
    "${TOKEN_FILE}"; then
    echo "[local-launcher] 获取 TOKEN 失败，请先启动机器人侧脚本。" >&2
    exit 1
  fi
  chmod 600 "${TOKEN_FILE}"
  IFS= read -r HC_TJ_GATEWAY_TOKEN <"${TOKEN_FILE}"
  export HC_TJ_GATEWAY_TOKEN
}

load_token() {
  if [[ -f "${TOKEN_FILE}" ]]; then
    IFS= read -r HC_TJ_GATEWAY_TOKEN <"${TOKEN_FILE}"
  fi
  if [[ -z "${HC_TJ_GATEWAY_TOKEN:-}" ]]; then
    fetch_token
  fi
  if (( ${#HC_TJ_GATEWAY_TOKEN} < 16 )); then
    echo "[local-launcher] TOKEN 必须至少包含 16 个字符。" >&2
    exit 2
  fi
  export HC_TJ_GATEWAY_TOKEN
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "${INFERENCE_PID}" ]] && kill -0 "${INFERENCE_PID}" 2>/dev/null; then
    echo "[local-launcher] 正在停止本机推理服务..."
    kill -TERM "${INFERENCE_PID}" 2>/dev/null || true
    wait "${INFERENCE_PID}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ ! -x "${PYTHON}" ]]; then
  echo "[local-launcher] 找不到 Python: ${PYTHON}" >&2
  exit 2
fi
if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "[local-launcher] 找不到模型目录: ${MODEL_DIR}" >&2
  exit 2
fi
if ss -ltnH "sport = :${INFERENCE_PORT}" | grep -q .; then
  echo "[local-launcher] 端口 ${INFERENCE_PORT} 已被占用，请先停止旧推理服务。" >&2
  exit 2
fi

load_token
health="$(curl -fsS --max-time 3 "${ROBOT_URL}/healthz")" || {
  echo "[local-launcher] 无法连接机器人网关: ${ROBOT_URL}" >&2
  exit 1
}
if ! HC_TJ_HEALTH_PAYLOAD="${health}" "${PYTHON}" - <<'PY'
import json
import os

try:
    health = json.loads(os.environ["HC_TJ_HEALTH_PAYLOAD"])
    ready = (
        health["status"] == "ok"
        and health["fault"] is None
        and health["control_enabled"] is True
        and health["hands_enabled"] is True
    )
except (KeyError, TypeError, ValueError, json.JSONDecodeError):
    ready = False
raise SystemExit(0 if ready else 1)
PY
then
  echo "[local-launcher] 机器人网关未处于双臂+双手可控状态: ${health}" >&2
  exit 1
fi
if ! curl -fsS --max-time 3 \
  -H "Authorization: Bearer ${HC_TJ_GATEWAY_TOKEN}" \
  "${ROBOT_URL}/hc_tj/joint_feedback/v1" >/dev/null; then
  echo "[local-launcher] 本地 TOKEN 已失效，自动从机器人刷新..."
  fetch_token
  if ! curl -fsS --max-time 3 \
    -H "Authorization: Bearer ${HC_TJ_GATEWAY_TOKEN}" \
    "${ROBOT_URL}/hc_tj/joint_feedback/v1" >/dev/null; then
    echo "[local-launcher] TOKEN 自动刷新后仍无法通过网关鉴权。" >&2
    exit 1
  fi
fi

mkdir -p -- "${RUNTIME_DIR}"
echo "[local-launcher] 启动 GPU 推理服务，日志: ${INFERENCE_LOG}"
"${PYTHON}" "${SCRIPT_DIR}/inference_server.py" \
  --model_dir "${MODEL_DIR}" \
  --inference_prefix hc_tj \
  --server_name holobrain \
  --port "${INFERENCE_PORT}" \
  >"${INFERENCE_LOG}" 2>&1 &
INFERENCE_PID=$!

echo "[local-launcher] 等待模型加载完成..."
deadline=$((SECONDS + 300))
while ! curl -s --max-time 1 -o /dev/null "http://127.0.0.1:${INFERENCE_PORT}/"; do
  if ! kill -0 "${INFERENCE_PID}" 2>/dev/null; then
    wait "${INFERENCE_PID}" || true
    echo "[local-launcher] 推理服务启动失败，日志末尾：" >&2
    tail -n 50 "${INFERENCE_LOG}" >&2 || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "[local-launcher] 模型加载超过 300 秒，日志: ${INFERENCE_LOG}" >&2
    exit 1
  fi
  sleep 1
done

echo "[local-launcher] 使用真实观测预热推理，预热动作不会下发..."
HC_TJ_WARMUP_ROBOT_URL="${ROBOT_URL}" \
HC_TJ_WARMUP_SERVER_URL="${SERVER_URL}" \
HC_TJ_WARMUP_INSTRUCTION="${INSTRUCTION}" \
PYTHONPATH="${REPOSITORY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
"${PYTHON}" - <<'PY'
import os

import requests

from projects.holobrain.scripts.hc_tj_async_client import _build_request
from projects.holobrain.scripts.hc_tj_http_async_client import (
    HcTjHttpAsyncClientConfig,
    _request_inference,
    _request_observation,
    packet_to_snapshot,
)

config = HcTjHttpAsyncClientConfig(
    robot_url=os.environ["HC_TJ_WARMUP_ROBOT_URL"],
    robot_token=os.environ["HC_TJ_GATEWAY_TOKEN"],
    server_url=os.environ["HC_TJ_WARMUP_SERVER_URL"],
    instruction=os.environ["HC_TJ_WARMUP_INSTRUCTION"],
    depth_units_per_meter=float(
        os.environ.get("HC_TJ_DEPTH_UNITS_PER_METER", "1000")
    ),
)
with requests.Session() as session:
    packet = _request_observation(session, config, 0)
    if packet is None:
        raise RuntimeError("预热失败：未获取到同步的机器人观测")
    files, form, _ = _build_request(
        packet_to_snapshot(packet), config, 0.0, 0.0, None, 0
    )
    _, elapsed = _request_inference(session, config, files, form)
print(f"[local-launcher] 推理预热完成 ({elapsed * 1000:.1f} ms)。")
PY

echo "[local-launcher] 推理服务已就绪，启动控制客户端。按 Ctrl+C 全部停止。"
"${PYTHON}" "${SCRIPT_DIR}/hc_tj_http_async_client.py" \
  --robot-url "${ROBOT_URL}" \
  --server-url "${SERVER_URL}" \
  --instruction "${INSTRUCTION}" \
  --depth-units-per-meter "${HC_TJ_DEPTH_UNITS_PER_METER:-1000}" \
  --first-point-jump-rad "${HC_TJ_FIRST_POINT_JUMP_RAD:-0.3}" \
  --inference-hz "${HC_TJ_INFERENCE_HZ:-2.0}" \
  --latency-ewma-alpha "${HC_TJ_LATENCY_EWMA_ALPHA:-0.25}" \
  --enable-control
