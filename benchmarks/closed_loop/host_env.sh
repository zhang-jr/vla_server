# shellcheck shell=bash
# Robot-host environment for benchmarks/closed_loop: ROS2 Foxy + ARX X5 messages
# + realsense2_camera, system Python 3.8, repo on PYTHONPATH.
#
# Usage: source benchmarks/closed_loop/host_env.sh
# Same clean-up as ARX's arx_env.sh: conda / ROS1 leftovers break Foxy's Python 3.8.

ARX_WS=${ARX_WS:-/home/go2/ARX_X5/ROS2/X5_ws}
RS_WS=${RS_WS:-/home/go2/ros2_ws}
_vla_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

# ROS setup scripts reference unset variables
case $- in *u*) _vla_nounset=1; set +u ;; *) _vla_nounset=0 ;; esac

if [ -n "${CONDA_PREFIX:-}" ]; then conda deactivate >/dev/null 2>&1 || true; fi
PATH=$(printf '%s' "$PATH" | tr ':' '\n' | grep -v -i conda | paste -sd: -)
export PATH
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION ROS_MASTER_URI ROS_HOSTNAME ROS_IP
unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH LD_LIBRARY_PATH

source /opt/ros/foxy/setup.bash
source "$ARX_WS/install/local_setup.bash"
source "$RS_WS/install/local_setup.bash"
export PYTHONPATH="$_vla_repo${PYTHONPATH:+:$PYTHONPATH}"

[ "$_vla_nounset" = 1 ] && set -u
unset _vla_repo _vla_nounset
