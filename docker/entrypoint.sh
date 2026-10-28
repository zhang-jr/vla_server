#!/usr/bin/env bash
# Container entrypoint: carry `docker run -e` values into SSH sessions, start sshd, run CMD.
set -euo pipefail

runtime_env=/etc/vla-runtime-env.sh
: > "$runtime_env"
chmod 600 "$runtime_env"   # may hold HF_TOKEN
for var in HF_TOKEN HF_ENDPOINT HF_HOME OPENPI_DATA_HOME CUDA_VISIBLE_DEVICES \
           NVIDIA_VISIBLE_DEVICES NVIDIA_DRIVER_CAPABILITIES \
           HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
    if [ -n "${!var:-}" ]; then
        printf 'export %s=%q\n' "$var" "${!var}" >> "$runtime_env"
    fi
done

mkdir -p /data/hf /data/openpi /data/scratch /root/.ssh
chmod 700 /root/.ssh
if [ -n "${SSH_PUBLIC_KEY:-}" ]; then
    grep -qxF "$SSH_PUBLIC_KEY" /root/.ssh/authorized_keys 2>/dev/null \
        || printf '%s\n' "$SSH_PUBLIC_KEY" >> /root/.ssh/authorized_keys
fi
if [ -s /root/.ssh/authorized_keys ]; then
    chmod 600 /root/.ssh/authorized_keys
    ssh-keygen -A >/dev/null
    /usr/sbin/sshd -p "${SSH_PORT:-22}"
    echo "sshd listening on ${SSH_PORT:-22} (key auth only)"
else
    echo "no SSH key given (SSH_PUBLIC_KEY or /root/.ssh/authorized_keys): sshd not started"
fi

exec "$@"
