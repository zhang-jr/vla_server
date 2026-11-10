#!/usr/bin/env bash
# Start the D435i colour streams as ROS2 topics /cam/<name>/color/image_raw
# (depth / IMU off, so USB bandwidth and CPU go to what the policy consumes).
#
# Usage: WRIST_SERIAL=<serial> THIRD_SERIAL=<serial> bash benchmarks/closed_loop/start_cameras.sh
#   COLOR_PROFILE=640,480,30 by default; `rs-enumerate-devices -s` lists serials.
#   Ctrl-C stops both camera nodes.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/host_env.sh"
: "${WRIST_SERIAL:?set WRIST_SERIAL}"
: "${THIRD_SERIAL:?set THIRD_SERIAL}"
PROFILE=${COLOR_PROFILE:-640,480,30}

launch() {
    # serial_no needs the leading underscore so ROS keeps it a string
    ros2 launch realsense2_camera rs_launch.py \
        camera_namespace:=cam camera_name:="$1" serial_no:=_"$2" \
        rgb_camera.color_profile:="$PROFILE" enable_color:=true \
        enable_depth:=false enable_infra1:=false enable_infra2:=false \
        enable_gyro:=false enable_accel:=false
}

trap 'kill 0' INT TERM EXIT
launch wrist "$WRIST_SERIAL" &
launch third "$THIRD_SERIAL" &
wait
