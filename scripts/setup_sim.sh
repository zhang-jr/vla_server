#!/usr/bin/env bash
# Install the closed-loop benchmark stack on the remote GPU host (ADR-020).
#
#   python scripts/remote.py sync
#   python scripts/remote.py run -- bash scripts/setup_sim.sh            # libero + maniskill
#   python scripts/remote.py run -- bash scripts/setup_sim.sh libero     # one of them
#
# Needs, from the system (not installable here without root):
#   LIBERO    : EGL for headless MuJoCo rendering (libegl1, libgl1; NVIDIA driver's EGL)
#   ManiSkill : Vulkan for SAPIEN rendering (libvulkan1 + the NVIDIA ICD json)
set -euo pipefail

targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then
  targets=(libero maniskill)
fi

python -m pip install -e ".[dev]"

for target in "${targets[@]}"; do
  case "$target" in
    libero)
      python -m pip install -e ".[sim-libero]"
      export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
      # LIBERO asks where to keep its datasets on first import; take the default.
      printf 'N\n' | python -c "from libero.libero import benchmark; print('libero suites:', sorted(benchmark.get_benchmark_dict()))"
      ;;
    maniskill)
      python -m pip install -e ".[sim-maniskill]"
      python -c "import mani_skill, gymnasium; print('mani_skill', mani_skill.__version__)"
      ;;
    *)
      echo "unknown target: $target (expected libero or maniskill)" >&2
      exit 2
      ;;
  esac
done

python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count())
try:
    import lerobot
    print("lerobot", getattr(lerobot, "__version__", "?"))
except ImportError:
    print("lerobot not installed")
EOF
