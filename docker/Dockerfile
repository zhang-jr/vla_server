#
# Remote GPU host image for VLA Runtime benchmarks (ADR-020).
#
# One image, one Python env per model stack — they pin incompatible
# transformers versions, so they cannot share a venv:
#
#   /opt/venvs/openvla   OpenVLA-7B (py3.10, transformers 4.40.1, HF remote code)
#   /opt/venvs/oft       OpenVLA-OFT (py3.10, moojink/openvla-oft + its transformers fork)
#   /opt/venvs/openpi    openpi JAX reference (only with --build-arg WITH_OPENPI=1)
#   /opt/venvs/lerobot   LeRobot ≥ 0.6 (py3.12): π0 / π0.5 / SmolVLA, vla_runtime tests  ← default `python`
#
# benchmarks/latency/run.py picks /opt/venvs/<group>/bin/python per model.
# The repo itself is NOT baked in: scripts/remote.py syncs the working tree to
# /workspace/vla_server and runs from there.
#
# The LeRobot env is the last heavy layer: bumping LEROBOT_REF (checkpoints
# saved by a newer LeRobot don't load in an older one) rebuilds only that.
#
# Build (context = this directory):
#   docker build -f docker/Dockerfile -t vla-runtime:bench docker/
#   # PyPI defaults to the Aliyun mirror; to use upstream PyPI instead:
#   #   --build-arg PYPI_INDEX=https://pypi.org/simple
#   # torch wheels still come from TORCH_INDEX (must serve +cu128 wheels).
#   # Python 3.12 for the LeRobot env is fetched by uv from GitHub
#   # (python-build-standalone); PYTHON_INSTALL_MIRROR points it elsewhere.
#
# Run (SSH into the container is how scripts/remote.py reaches it):
#   docker run -d --name vla --gpus all --shm-size 16g -p 2222:22 \
#     -v /data/vla:/data -v /path/to/model_zoo:/models:ro \
#     -e SSH_PUBLIC_KEY="$(cat ~/.ssh/id_ed25519.pub)" \
#     -e HF_TOKEN=hf_xxx -e HF_ENDPOINT=https://hf-mirror.com \
#     vla-runtime:bench
#
# Then in the local .env:
#   VLA_REMOTE_HOST=<host>  VLA_REMOTE_PORT=2222  VLA_REMOTE_USER=root
#   VLA_REMOTE_WORKDIR=/workspace/vla_server  VLA_REMOTE_SETUP=. /etc/vla-env.sh
#
# Host driver: the CUDA 12.8 wheels run on driver ≥ 570 natively and on R550
# (CUDA 12.4) through CUDA minor-version compatibility — Ada (RTX 4090) only;
# RTX 5090 (Blackwell, sm_120) needs ≥ 570. The base image's forward-compat
# libcuda is removed below: on GeForce cards it fails with error 804, and
# NVIDIA_DISABLE_REQUIRE lets the container start on an older driver.

ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.8.4

FROM ${UV_IMAGE} AS uv

FROM ${CUDA_IMAGE}

ARG PYPI_INDEX=https://mirrors.aliyun.com/pypi/simple/
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG TORCH_VERSION=2.7.1
ARG TORCHVISION_VERSION=0.22.1
ARG TORCHAUDIO_VERSION=2.7.1
ARG TORCH_CUDA=cu128
ARG PYTHON_INSTALL_MIRROR=
# *_REF must be a tag or branch name (shallow clone); a bare commit hash won't work.
ARG LEROBOT_REF=v0.6.1
ARG OFT_REF=main
ARG OPENPI_REF=main
ARG WITH_OPENPI=0
ARG WITH_LIBERO=0

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_DISABLE_REQUIRE=1 \
    UV_DEFAULT_INDEX=${PYPI_INDEX} \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=600 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    GIT_LFS_SKIP_SMUDGE=1 \
    HF_HOME=/data/hf \
    OPENPI_DATA_HOME=/data/openpi \
    VLA_BENCH_SCRATCH=/data/scratch \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    PATH=/opt/venvs/lerobot/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev \
        git git-lfs curl ca-certificates build-essential cmake pkg-config \
        ffmpeg libgl1 libegl1 libglib2.0-0 libsm6 libxext6 \
        openssh-server tar rsync tmux htop \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /usr/local/cuda-*/compat \
    && ldconfig

COPY --from=uv /uv /uvx /usr/local/bin/

# Every venv gets the same CUDA 12.8 torch; later installs may not swap it for
# the PyPI default build (no sm_120 kernels), hence the overrides file and the
# check after each env. VENV_PYTHON picks the interpreter (default py3.10).
RUN mkdir -p /opt/vla /opt/src /data /workspace/vla_server \
    && printf 'torch==%s+%s\ntorchvision==%s+%s\ntorchaudio==%s+%s\n' \
        "${TORCH_VERSION}" "${TORCH_CUDA}" "${TORCHVISION_VERSION}" "${TORCH_CUDA}" \
        "${TORCHAUDIO_VERSION}" "${TORCH_CUDA}" > /opt/vla/torch-overrides.txt \
    && printf '%s\n' \
        '#!/usr/bin/env bash' \
        'set -euo pipefail' \
        'venv=$1; shift' \
        'uv venv --seed --python "${VENV_PYTHON:-/usr/bin/python3.10}" "$venv"' \
        'uv pip install --python "$venv/bin/python" --index-url "$TORCH_INDEX" \' \
        '    "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" "torchaudio==$TORCHAUDIO_VERSION"' \
        'uv pip install --python "$venv/bin/python" --override /opt/vla/torch-overrides.txt \' \
        '    --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match \' \
        '    pydantic msgpack nvidia-ml-py "$@"' \
        > /opt/vla/make-venv.sh \
    && printf '%s\n' \
        '#!/usr/bin/env bash' \
        'set -euo pipefail' \
        '"$1/bin/python" - <<EOF' \
        'import torch' \
        'cuda = torch.version.cuda' \
        'assert cuda and tuple(map(int, cuda.split(".")[:2])) >= (12, 8), f"$1: torch {torch.__version__} (CUDA {cuda}) has no sm_120 kernels"' \
        'print("$1: torch", torch.__version__, "CUDA", cuda)' \
        'EOF' \
        > /opt/vla/check-torch.sh \
    && chmod +x /opt/vla/*.sh
ENV TORCH_INDEX=${TORCH_INDEX} TORCH_VERSION=${TORCH_VERSION} \
    TORCHVISION_VERSION=${TORCHVISION_VERSION} TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION}

# ── OpenVLA-7B: its HF remote code was written against transformers 4.40.1 ───
RUN /opt/vla/make-venv.sh /opt/venvs/openvla \
        transformers==4.40.1 tokenizers==0.19.1 timm==0.9.10 \
        "accelerate>=0.25,<1.0" sentencepiece protobuf "numpy<2" pillow \
    && /opt/vla/check-torch.sh /opt/venvs/openvla

# ── OpenVLA-OFT: repo install (pins its transformers fork); torch overridden ─
# TF 2.15 pins protobuf 4.x; current tensorflow-metadata and wandb need
# protobuf 5 and break `import experiments.robot.openvla_utils`, so pin them back.
RUN git clone --depth 1 --branch ${OFT_REF} https://github.com/moojink/openvla-oft.git /opt/src/openvla-oft \
    && /opt/vla/make-venv.sh /opt/venvs/oft -e /opt/src/openvla-oft \
        tensorflow-metadata==1.14.0 wandb==0.16.6 \
    && echo /opt/src/openvla-oft > "$(/opt/venvs/oft/bin/python -c 'import site; print(site.getsitepackages()[0])')/openvla_oft_repo.pth" \
    && /opt/venvs/oft/bin/python -c "import experiments.robot.openvla_utils" \
    && /opt/vla/check-torch.sh /opt/venvs/oft

# ── openpi (JAX reference, optional: big, and JAX on sm_120 is unverified) ────
RUN if [ "${WITH_OPENPI}" = 1 ]; then \
        git clone --depth 1 --branch ${OPENPI_REF} --recurse-submodules --shallow-submodules \
            https://github.com/Physical-Intelligence/openpi.git /opt/src/openpi \
        && cd /opt/src/openpi \
        && UV_PROJECT_ENVIRONMENT=/opt/venvs/openpi uv sync --frozen \
        && uv pip install --python /opt/venvs/openpi/bin/python msgpack nvidia-ml-py; \
    fi

# ── LeRobot: π0 / π0.5 / SmolVLA + the project's own test / dev tools ────────
# LeRobot ≥ 0.6 needs Python ≥ 3.12 and transformers 5.x.
RUN if [ -n "${PYTHON_INSTALL_MIRROR}" ]; then export UV_PYTHON_INSTALL_MIRROR="${PYTHON_INSTALL_MIRROR}"; fi \
    && uv python install 3.12 \
    && git clone --depth 1 --branch ${LEROBOT_REF} https://github.com/huggingface/lerobot.git /opt/src/lerobot \
    && VENV_PYTHON=3.12 /opt/vla/make-venv.sh /opt/venvs/lerobot \
        -e "/opt/src/lerobot[pi,smolvla]" \
        pytest pytest-asyncio pytest-benchmark ruff mypy \
    && if [ "${WITH_LIBERO}" = 1 ]; then \
        uv pip install --python /opt/venvs/lerobot/bin/python \
            --override /opt/vla/torch-overrides.txt \
            --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match \
            -e "/opt/src/lerobot[libero]"; \
    fi \
    && /opt/venvs/lerobot/bin/python -c "from lerobot.policies.factory import get_policy_class; get_policy_class('pi05'); get_policy_class('smolvla')" \
    && /opt/vla/check-torch.sh /opt/venvs/lerobot

# Freeze what actually resolved, for the result files' provenance.
RUN for venv in /opt/venvs/*; do uv pip freeze --python "$venv/bin/python" > "$venv/requirements.lock"; done

# SSH: key-only root login (keys come from SSH_PUBLIC_KEY or a mounted authorized_keys).
RUN mkdir -p /run/sshd /root/.ssh && chmod 700 /root/.ssh \
    && sed -ri 's/^#?PermitRootLogin .*/PermitRootLogin prohibit-password/; s/^#?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config

COPY vla-env.sh /etc/vla-env.sh
COPY entrypoint.sh /usr/local/bin/vla-entrypoint
RUN chmod +x /usr/local/bin/vla-entrypoint

WORKDIR /workspace/vla_server
VOLUME ["/data"]
EXPOSE 22
ENTRYPOINT ["/usr/local/bin/vla-entrypoint"]
CMD ["sleep", "infinity"]
