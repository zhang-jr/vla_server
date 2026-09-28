"""Simulation environments behind one observation format.

Every environment returns observations as::

    {
        "images": {camera_name: FloatTensor[3, H, W] in [0, 1]},  # model order
        "robot_state": RobotState,       # structured; extra["policy_state"] is
                                         # the dataset-convention vector a model reads
        "instruction": str,
    }

and extends ``SimEnvironment`` (vla_runtime.quantization.validation), so the
same wrappers serve the quantization validator (ADR-008) and the runtime
benchmarks. ``step()`` reports success in ``info["success"]``.

Simulator packages are imported lazily: this module imports on a machine
without LIBERO or ManiSkill, and only the wrapper that is actually built needs
its simulator installed.
"""

from __future__ import annotations

import math
import os
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vla_runtime.action_head.base import RobotState
from vla_runtime.quantization.validation import SimEnvironment

Observation = dict[str, Any]


@dataclass(frozen=True)
class EnvSpec:
    """Static facts about an environment a policy is built against.

    Attributes:
        name: Environment / suite identifier for reports.
        action_dim: Width of the action the environment consumes.
        control_hz: Rate at which one ``step()`` advances the robot.
        camera_names: Views in ``obs["images"]``, in order.
        state_dim: ``RobotState.flat_dim`` of every observation.
        policy_state_dim: Width of ``robot_state.extra["policy_state"]``.
        action_is_delta: Actions are relative commands (a zero arm command
            holds position) rather than absolute targets.
        gripper_index: Index of the gripper channel in the action.
    """

    name: str
    action_dim: int
    control_hz: float
    camera_names: tuple[str, ...]
    state_dim: int
    policy_state_dim: int
    action_is_delta: bool
    gripper_index: int = -1


class BenchEnv(SimEnvironment):
    """SimEnvironment plus what the runners need to drive it in real time."""

    spec: EnvSpec

    def max_steps(self, task_name: str) -> int:
        """Episode step limit for a task."""
        raise NotImplementedError

    def hold_action(self, last_action: torch.Tensor | None) -> torch.Tensor:
        """Action sent when the chunk buffer is empty (an underrun).

        Delta control: zero arm motion, gripper kept where it was. Absolute
        control: repeat the last target.
        """
        if self.spec.action_is_delta:
            action = torch.zeros(self.spec.action_dim)
            if last_action is not None:
                action[self.spec.gripper_index] = last_action[self.spec.gripper_index]
            return action
        if last_action is not None:
            return last_action.clone()
        return torch.zeros(self.spec.action_dim)


def _image_to_tensor(image: Any, flip: bool = False) -> torch.Tensor:
    """HWC uint8 (numpy or torch) → CHW float in [0, 1]."""
    array = image.cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
    if flip:
        array = array[::-1, ::-1]
    array = np.ascontiguousarray(array)
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """(x, y, z, w) quaternion → axis-angle, as robosuite / openpi compute it."""
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = math.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


# ── Mock ──────────────────────────────────────────────────────────────────────

class MockBenchEnv(BenchEnv):
    """GPU-free stand-in with the real observation format.

    Integrates delta actions into a joint vector and succeeds with a
    seed-determined probability after ``steps_to_done`` steps. Exercises the
    runner plumbing; says nothing about policy quality.
    """

    def __init__(
        self,
        num_tasks: int = 3,
        steps_to_done: int = 40,
        success_probability: float = 0.5,
        image_size: int = 64,
        control_hz: float = 20.0,
        dof: int = 7,
    ) -> None:
        self._num_tasks = num_tasks
        self._steps_to_done = steps_to_done
        self._success_probability = success_probability
        self._image_size = image_size
        self._dof = dof
        self._rng = random.Random(0)
        self._task = ""
        self._step = 0
        self._joints = torch.zeros(dof)
        self._will_succeed = False
        self.spec = EnvSpec(
            name="mock",
            action_dim=dof,
            control_hz=control_hz,
            camera_names=("base", "wrist"),
            state_dim=2 * dof + 1 + dof,
            policy_state_dim=dof,
            action_is_delta=True,
        )

    def task_names(self) -> list[str]:
        return [f"mock_task_{i}" for i in range(self._num_tasks)]

    def max_steps(self, task_name: str) -> int:
        return self._steps_to_done * 2

    def reset(self, task_name: str, episode_idx: int, seed: int) -> Observation:
        self._rng = random.Random(seed)
        self._task = task_name
        self._step = 0
        self._joints = torch.zeros(self._dof)
        self._will_succeed = self._rng.random() < self._success_probability
        return self._obs()

    def step(self, action: torch.Tensor) -> tuple[Observation, float, bool, dict]:
        self._step += 1
        self._joints = self._joints + 0.01 * action.float()[: self._dof]
        success = self._will_succeed and self._step >= self._steps_to_done
        return self._obs(), float(success), success, {"success": success}

    def close(self) -> None:
        pass

    def _obs(self) -> Observation:
        size = self._image_size
        shade = (self._step % 255) / 255.0
        state = RobotState(
            joint_positions=self._joints.clone(),
            joint_velocities=torch.zeros(self._dof),
            gripper_state=torch.zeros(1),
            extra={"policy_state": self._joints.clone()},
        )
        return {
            "images": {
                "base": torch.full((3, size, size), shade),
                "wrist": torch.full((3, size, size), 1.0 - shade),
            },
            "robot_state": state,
            "instruction": f"mock instruction for {self._task}",
        }


# ── LIBERO ────────────────────────────────────────────────────────────────────

# Episode step limits used by openpi / LeRobot LIBERO evaluations.
LIBERO_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
# Gripper open (-1), no arm motion: what LIBERO evals send while objects settle.
_LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


class LiberoEnv(BenchEnv):
    """LIBERO task suite (robosuite / MuJoCo, rendered headless via EGL).

    Follows the openpi / LeRobot evaluation protocol: fixed per-episode initial
    states from the suite, ``num_steps_wait`` settle steps, both camera views
    rotated 180° (LIBERO renders upside down relative to the training data),
    and an 8-D policy state = eef position + axis-angle + gripper qpos.

    Args:
        suite: One of ``LIBERO_MAX_STEPS``.
        resolution: Rendered camera height / width.
        num_steps_wait: Settle steps after loading the initial state.
        flip_images: Rotate views 180°. Disable only for a model trained on
            unrotated frames.
        task_ids: Restrict to these task indices (default: the whole suite).
    """

    def __init__(
        self,
        suite: str = "libero_object",
        resolution: int = 256,
        num_steps_wait: int = 10,
        flip_images: bool = True,
        task_ids: Sequence[int] | None = None,
    ) -> None:
        if suite not in LIBERO_MAX_STEPS:
            raise ValueError(f"unknown LIBERO suite {suite!r}; expected one of {sorted(LIBERO_MAX_STEPS)}")
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        from libero.libero import benchmark, get_libero_path

        self._suite_name = suite
        self._suite = benchmark.get_benchmark_dict()[suite]()
        self._bddl_root = get_libero_path("bddl_files")
        self._resolution = resolution
        self._num_steps_wait = num_steps_wait
        self._flip = flip_images
        ids = list(task_ids) if task_ids is not None else list(range(self._suite.n_tasks))
        self._names = {f"{i:02d}_{self._suite.get_task(i).name}": i for i in ids}
        self._env: Any = None
        self._env_task: int | None = None
        self._instruction = ""
        self.spec = EnvSpec(
            name=suite,
            action_dim=7,
            control_hz=20.0,
            camera_names=("agentview", "wrist"),
            state_dim=7 + 7 + 2 + 7 + 8,
            policy_state_dim=8,
            action_is_delta=True,
        )

    def task_names(self) -> list[str]:
        return list(self._names)

    def max_steps(self, task_name: str) -> int:
        return LIBERO_MAX_STEPS[self._suite_name]

    def reset(self, task_name: str, episode_idx: int, seed: int) -> Observation:
        task_id = self._names[task_name]
        if self._env_task != task_id:
            self._open(task_id)
        self._env.seed(seed)
        self._env.reset()
        init_states = self._suite.get_task_init_states(task_id)
        raw = self._env.set_init_state(init_states[episode_idx % len(init_states)])
        for _ in range(self._num_steps_wait):
            raw, _, _, _ = self._env.step(_LIBERO_DUMMY_ACTION)
        return self._obs(raw)

    def step(self, action: torch.Tensor) -> tuple[Observation, float, bool, dict]:
        raw, reward, done, info = self._env.step(action.float().cpu().numpy().tolist())
        # LIBERO signals done only on task success.
        info = dict(info)
        info["success"] = bool(done)
        return self._obs(raw), float(reward), bool(done), info

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
            self._env_task = None

    def _open(self, task_id: int) -> None:
        from libero.libero.envs import OffScreenRenderEnv

        self.close()
        task = self._suite.get_task(task_id)
        bddl = os.path.join(self._bddl_root, task.problem_folder, task.bddl_file)
        self._env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=self._resolution,
            camera_widths=self._resolution,
        )
        self._env_task = task_id
        self._instruction = task.language

    def _obs(self, raw: dict[str, np.ndarray]) -> Observation:
        eef_pos = np.asarray(raw["robot0_eef_pos"], dtype=np.float64)
        eef_quat = np.asarray(raw["robot0_eef_quat"], dtype=np.float64)
        gripper = np.asarray(raw["robot0_gripper_qpos"], dtype=np.float64)
        if "robot0_joint_pos" in raw:
            joints = np.asarray(raw["robot0_joint_pos"], dtype=np.float64)
        else:
            joints = np.arctan2(raw["robot0_joint_pos_sin"], raw["robot0_joint_pos_cos"])
        policy_state = np.concatenate([eef_pos, quat2axisangle(eef_quat), gripper])
        state = RobotState(
            joint_positions=torch.as_tensor(joints, dtype=torch.float32),
            joint_velocities=torch.as_tensor(raw["robot0_joint_vel"], dtype=torch.float32),
            gripper_state=torch.as_tensor(gripper, dtype=torch.float32),
            ee_pose=torch.as_tensor(np.concatenate([eef_pos, eef_quat]), dtype=torch.float32),
            extra={"policy_state": torch.as_tensor(policy_state, dtype=torch.float32)},
        )
        return {
            "images": {
                "agentview": _image_to_tensor(raw["agentview_image"], flip=self._flip),
                "wrist": _image_to_tensor(raw["robot0_eye_in_hand_image"], flip=self._flip),
            },
            "robot_state": state,
            "instruction": self._instruction,
        }


# ── ManiSkill ─────────────────────────────────────────────────────────────────

# ManiSkill tasks carry no language; these are the instructions a policy sees.
MANISKILL_INSTRUCTIONS: dict[str, str] = {
    "PickCube-v1": "pick up the cube and move it to the goal position",
    "StackCube-v1": "stack the red cube on top of the green cube",
    "PushCube-v1": "push the cube to the goal region",
    "PullCube-v1": "pull the cube to the goal region",
    "PegInsertionSide-v1": "insert the peg into the hole",
    "PlugCharger-v1": "plug the charger into the receptacle",
    "LiftPegUpright-v1": "lift the peg upright",
}


class ManiSkillEnv(BenchEnv):
    """ManiSkill3 tasks, one environment instance (``num_envs=1``) per wrapper.

    Each task is an env id; episodes are distinguished by reset seed. One
    wrapper per robot keeps fleet benchmarks honest: every robot steps its own
    scene on its own thread.

    Args:
        env_ids: ManiSkill env ids, e.g. ``("PickCube-v1", "StackCube-v1")``.
        control_mode: Must match what the evaluated checkpoint was trained on.
        resolution: Sensor camera height / width.
        max_episode_steps: Override the env's default step limit.
        sim_backend: ``"auto"``, ``"physx_cpu"`` or ``"physx_cuda"``.
        robot_uids: e.g. ``"panda"`` or ``"panda_wristcam"`` (adds a hand camera).
        gripper_dofs: Trailing qpos entries that belong to the gripper.
        state_mode: ``policy_state`` layout — ``"qpos"`` or ``"qpos_qvel"``.
        instructions: Per env id instruction overrides.
    """

    def __init__(
        self,
        env_ids: Sequence[str] = ("PickCube-v1",),
        control_mode: str = "pd_ee_delta_pose",
        resolution: int = 256,
        max_episode_steps: int | None = None,
        sim_backend: str = "auto",
        robot_uids: str | None = None,
        gripper_dofs: int = 2,
        state_mode: str = "qpos",
        instructions: dict[str, str] | None = None,
    ) -> None:
        if state_mode not in ("qpos", "qpos_qvel"):
            raise ValueError(f"state_mode must be 'qpos' or 'qpos_qvel', got {state_mode!r}")
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401  (registers the env ids)

        self._gym = gym
        self._env_ids = list(env_ids)
        self._make_kwargs: dict[str, Any] = {
            "obs_mode": "rgb",
            "control_mode": control_mode,
            "num_envs": 1,
            "sim_backend": sim_backend,
            "sensor_configs": {"width": resolution, "height": resolution},
        }
        if robot_uids:
            self._make_kwargs["robot_uids"] = robot_uids
        if max_episode_steps:
            self._make_kwargs["max_episode_steps"] = max_episode_steps
        self._gripper_dofs = gripper_dofs
        self._state_mode = state_mode
        self._instructions = {**MANISKILL_INSTRUCTIONS, **(instructions or {})}
        self._env: Any = None
        self._env_id: str | None = None

        # Probe the first task: cameras and widths are only known from an obs.
        probe = self.reset(self._env_ids[0], 0, 0)
        state: RobotState = probe["robot_state"]
        self.spec = EnvSpec(
            name="maniskill:" + ",".join(self._env_ids),
            action_dim=int(self._env.action_space.shape[-1]),
            control_hz=float(self._env.unwrapped.control_freq),
            camera_names=tuple(probe["images"]),
            state_dim=state.flat_dim,
            policy_state_dim=int(state.extra["policy_state"].numel()),
            action_is_delta="delta" in control_mode,
        )

    def task_names(self) -> list[str]:
        return list(self._env_ids)

    def max_steps(self, task_name: str) -> int:
        if self._env_id != task_name:
            self._open(task_name)
        return int(self._env.spec.max_episode_steps)

    def reset(self, task_name: str, episode_idx: int, seed: int) -> Observation:
        if self._env_id != task_name:
            self._open(task_name)
        raw, _ = self._env.reset(seed=seed)
        return self._obs(raw)

    def step(self, action: torch.Tensor) -> tuple[Observation, float, bool, dict]:
        batched = action.float().cpu().numpy().reshape(1, -1)
        raw, reward, terminated, truncated, info = self._env.step(batched)
        success = bool(_first(info.get("success", False)))
        done = bool(_first(terminated)) or bool(_first(truncated))
        return self._obs(raw), float(_first(reward)), done, {"success": success}

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
            self._env_id = None

    def _open(self, env_id: str) -> None:
        self.close()
        self._env = self._gym.make(env_id, **self._make_kwargs)
        self._env_id = env_id

    def _obs(self, raw: dict[str, Any]) -> Observation:
        qpos = raw["agent"]["qpos"][0].float().cpu()
        qvel = raw["agent"]["qvel"][0].float().cpu()
        arm = qpos.shape[0] - self._gripper_dofs
        extra_obs = raw.get("extra", {})
        ee_pose = extra_obs["tcp_pose"][0].float().cpu() if "tcp_pose" in extra_obs else None
        policy_state = qpos if self._state_mode == "qpos" else torch.cat([qpos, qvel])
        state = RobotState(
            joint_positions=qpos[:arm],
            joint_velocities=qvel[:arm],
            gripper_state=qpos[arm:],
            ee_pose=ee_pose,
            extra={"policy_state": policy_state.clone()},
        )
        images = {
            name: _image_to_tensor(sensor["rgb"][0])
            for name, sensor in raw["sensor_data"].items()
            if "rgb" in sensor
        }
        return {
            "images": images,
            "robot_state": state,
            "instruction": self._instructions.get(self._env_id or "", self._env_id or ""),
        }


def _first(value: Any) -> Any:
    """First element of a batched (num_envs=1) tensor / array, or the scalar."""
    if isinstance(value, torch.Tensor):
        return value.reshape(-1)[0].item()
    if isinstance(value, np.ndarray):
        return value.reshape(-1)[0].item()
    return value


# ── Factory ───────────────────────────────────────────────────────────────────

def env_factory(name: str, options: dict[str, str]) -> Callable[[], BenchEnv]:
    """Return a zero-arg constructor, so each worker thread builds its own env.

    Rendering contexts (EGL) and simulator handles are bound to the thread that
    creates them; the runners construct and step every env on one dedicated
    thread.

    Args:
        name: ``mock``, ``libero`` or ``maniskill``.
        options: ``--env-opt key=value`` pairs, converted per environment.
    """
    opts = dict(options)
    if name == "mock":
        kwargs: dict[str, Any] = {
            k: _parse_number(v) for k, v in opts.items()
        }
        return lambda: MockBenchEnv(**kwargs)
    if name == "libero":
        kwargs = {}
        if "suite" in opts:
            kwargs["suite"] = opts.pop("suite")
        if "resolution" in opts:
            kwargs["resolution"] = int(opts.pop("resolution"))
        if "num_steps_wait" in opts:
            kwargs["num_steps_wait"] = int(opts.pop("num_steps_wait"))
        if "flip_images" in opts:
            kwargs["flip_images"] = _parse_bool(opts.pop("flip_images"))
        if "task_ids" in opts:
            kwargs["task_ids"] = [int(x) for x in opts.pop("task_ids").split(",") if x]
        _reject_unknown(name, opts)
        return lambda: LiberoEnv(**kwargs)
    if name == "maniskill":
        kwargs = {}
        if "env_ids" in opts:
            kwargs["env_ids"] = [x for x in opts.pop("env_ids").split(",") if x]
        for key in ("control_mode", "sim_backend", "robot_uids", "state_mode"):
            if key in opts:
                kwargs[key] = opts.pop(key)
        for key in ("resolution", "max_episode_steps", "gripper_dofs"):
            if key in opts:
                kwargs[key] = int(opts.pop(key))
        _reject_unknown(name, opts)
        return lambda: ManiSkillEnv(**kwargs)
    raise ValueError(f"unknown env {name!r}; expected mock, libero or maniskill")


def _parse_number(value: str) -> int | float:
    return float(value) if any(c in value for c in ".e") else int(value)


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _reject_unknown(env: str, leftover: dict[str, str]) -> None:
    if leftover:
        raise ValueError(f"unknown --env-opt for {env}: {sorted(leftover)}")
