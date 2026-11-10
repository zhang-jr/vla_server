# Environment for SSH sessions into the benchmark container.
#
# sshd starts sessions with a clean environment, so the image's ENV and the
# `docker run -e` values are lost there; set VLA_REMOTE_SETUP=". /etc/vla-env.sh"
# in the local .env so scripts/remote.py sources this before every command.

export PATH=/opt/venvs/lerobot/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export HF_HOME=${HF_HOME:-/data/hf}
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/data/openpi}
export VLA_BENCH_SCRATCH=${VLA_BENCH_SCRATCH:-/data/scratch}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export GIT_LFS_SKIP_SMUDGE=1

# Written by the entrypoint from `docker run -e ...` (HF_TOKEN, proxies, GPU selection).
if [ -f /etc/vla-runtime-env.sh ]; then
    . /etc/vla-runtime-env.sh
fi
