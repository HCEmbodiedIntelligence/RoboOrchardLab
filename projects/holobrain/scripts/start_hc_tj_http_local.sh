#!/usr/bin/env bash
# Compatibility entry point. Robot observations and commands now use ROS 2 DDS.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/start_hc_tj_ros_local.sh" "$@"
