#!/usr/bin/env bash
# Start HoloBrain locally and exchange robot observations/actions through ROS 2 DDS.
set -Eeuo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
用法: start_hc_tj_ros_local.sh ['任务指令']
启动本机 HoloBrain 推理服务和 ROS 2 直连客户端。
观测直接订阅 /observation/*；动作通过 HC 中间件程序控制入口下发。
EOF
  exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
CONDA_BASE="${HC_TJ_CONDA_BASE:-${HOME}/miniforge3}"
if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  set +u
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${HC_TJ_CONDA_ENV:-holobrain}"
  set -u
fi
set +u
source /opt/ros/humble/setup.bash
set -u

PYTHON="${HC_TJ_PYTHON:-/home/hc-em-lap/miniforge3/envs/holobrain/bin/python}"
MODEL_DIR="${HC_TJ_MODEL_DIR:-/home/hc-em-lap/workspace/weights/holobrain/261027_export/model}"
URDF_PATH="${HC_TJ_URDF_PATH:-${REPOSITORY_ROOT}/projects/holobrain/assets/hc_tj_robot.urdf}"
INFERENCE_PORT="${HC_TJ_INFERENCE_PORT:-2000}"
INSTRUCTION="${*:-${HC_TJ_INSTRUCTION:-grasp red snack}}"
RUNTIME_DIR="${REPOSITORY_ROOT}/runtime/holobrain_ros"
INFERENCE_LOG="${RUNTIME_DIR}/inference_$(date +'%Y%m%d_%H%M%S').log"
CLIENT_LOG="${RUNTIME_DIR}/client_$(date +'%Y%m%d_%H%M%S').log"
TOKEN_FILE="${HC_TJ_CONTROL_TOKEN_FILE:-${HOME}/.config/holobrain/hc_tj_control_token}"
DDS_PROFILE="${HC_TJ_DDS_PROFILE:-${HOME}/fastdds_inference_profile.xml}"
INFERENCE_PID=""

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-13}"
export ROS_LOCALHOST_ONLY=0
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-${DDS_PROFILE}}"
export PYTHONPATH="${REPOSITORY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export no_proxy="${no_proxy:-},${NO_PROXY:-},localhost,127.0.0.1,::1"
export NO_PROXY="${no_proxy}"

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

[[ -x "${PYTHON}" ]] || { echo "[local-launcher] 找不到 Python: ${PYTHON}" >&2; exit 2; }
[[ -d "${MODEL_DIR}" ]] || { echo "[local-launcher] 找不到模型目录: ${MODEL_DIR}" >&2; exit 2; }
[[ -f "${URDF_PATH}" ]] || { echo "[local-launcher] 找不到机器人 URDF: ${URDF_PATH}" >&2; exit 2; }
[[ -f "${FASTRTPS_DEFAULT_PROFILES_FILE}" ]] || {
  echo "[local-launcher] 找不到 DDS 配置: ${FASTRTPS_DEFAULT_PROFILES_FILE}" >&2
  exit 2
}
if ss -ltnH "sport = :${INFERENCE_PORT}" | grep -q .; then
  echo "[local-launcher] 端口 ${INFERENCE_PORT} 已被占用，请先停止旧推理服务。" >&2
  exit 2
fi

if [[ ! -s "${TOKEN_FILE}" ]]; then
  mkdir -p -- "$(dirname -- "${TOKEN_FILE}")"
  umask 077
  "${PYTHON}" -c 'import secrets; print(secrets.token_urlsafe(32))' >"${TOKEN_FILE}"
fi
IFS= read -r HC_TJ_CONTROL_TOKEN <"${TOKEN_FILE}"
if (( ${#HC_TJ_CONTROL_TOKEN} < 16 )); then
  echo "[local-launcher] 控制令牌必须至少包含 16 个字符: ${TOKEN_FILE}" >&2
  exit 2
fi
chmod 600 "${TOKEN_FILE}"
mkdir -p -- "${RUNTIME_DIR}"
echo "[local-launcher] ROS_DOMAIN_ID=${ROS_DOMAIN_ID} DDS=${FASTRTPS_DEFAULT_PROFILES_FILE}"
echo "[local-launcher] 直接订阅 /observation/cameras/head/*、关节和手部反馈。"
echo "[local-launcher] 启动 GPU 推理服务，日志: ${INFERENCE_LOG}"
"${PYTHON}" "${SCRIPT_DIR}/inference_server.py" \
  --model_dir "${MODEL_DIR}" \
  --inference_prefix hc_tj \
  --server_name holobrain \
  --port "${INFERENCE_PORT}" \
  --require_control true \
  --control_token "${HC_TJ_CONTROL_TOKEN}" \
  >"${INFERENCE_LOG}" 2>&1 &
INFERENCE_PID=$!

echo "[local-launcher] 等待模型加载完成..."
deadline=$((SECONDS + 300))
while ! curl --noproxy '*' -s --max-time 1 -o /dev/null "http://127.0.0.1:${INFERENCE_PORT}/"; do
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

echo "[local-launcher] 推理服务已就绪，启动 ROS 2 直连客户端。按 Ctrl+C 全部停止。"
echo "[local-launcher] 模型持续推理；动作下发仍需 HC Dashboard 开启程序控制。"
echo "[local-launcher] 客户端频率、限速与抢占日志: ${CLIENT_LOG}"
"${PYTHON}" "${SCRIPT_DIR}/hc_tj_async_client.py" \
  --server-url "http://127.0.0.1:${INFERENCE_PORT}/holobrain/hc_tj_async/v2" \
  --instruction "${INSTRUCTION}" \
  --urdf-path "${URDF_PATH}" \
  --depth-units-per-meter "${HC_TJ_DEPTH_UNITS_PER_METER:-1000}" \
  --control-token "${HC_TJ_CONTROL_TOKEN}" \
  --inference-hz "${HC_TJ_INFERENCE_HZ:-8.0}" \
  --control-hz "${HC_TJ_CONTROL_HZ:-100.0}" \
  --color-topic /observation/cameras/head/color/compressed \
  --depth-topic /observation/cameras/head/depth/compressed \
  --calibration-topic /observation/cameras/head/calibration \
  --extrinsics-topic /observation/cameras/head/extrinsics \
  --joint-state-topic /observation/joint_states \
  --hand-state-topic /observation/hands/joint_states \
  --ownership-topic /status/control/ownership \
  --joint-command-topic /hc_teleop/joint_cmd \
  --right-hand-topic /hc_teleop/joint_cmd_finger_right \
  --left-hand-topic /hc_teleop/joint_cmd_finger_left \
  --first-point-jump-rad "${HC_TJ_FIRST_POINT_JUMP_RAD:-0.60}" \
  --state-timeout-s "${HC_TJ_STATE_TIMEOUT_S:-0.6}" \
  --enable-dagger \
  --enable-control 2>&1 | tee "${CLIENT_LOG}"
