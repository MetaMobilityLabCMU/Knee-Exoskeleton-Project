import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import json
import pickle
import sys
from argparse import ArgumentParser
from dataclasses import dataclass, field
from multiprocessing import freeze_support
from pathlib import Path
from typing import List, Optional

import gym
import numpy as np
import numpy.core as np_core
import numpy.core.multiarray as np_multiarray
import numpy.core.numeric as np_numeric
import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv

import env
from data.ReferenceMotion2D_H1018_nc import ReferenceMotion2D_H1018_nc as ReferenceMotion2D
from utils.sim_config import get_clip_indices, get_step_size, resolve_config_paths


sys.modules.setdefault("numpy._core", np_core)
sys.modules.setdefault("numpy._core.numeric", np_numeric)
sys.modules.setdefault("numpy._core.multiarray", np_multiarray)


BASE = Path(__file__).resolve().parent


def resolve_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return BASE / path


def get_env_id(config: dict) -> str:
    return config.get("env_id", "H1018-v0")


def build_ref_motion(config: dict) -> ReferenceMotion2D:
    return ReferenceMotion2D(
        ik_folder_path=config["ik_folder_path"],
        analyze_folder_path=config["analyze_folder_path"],
        simul_step_size=get_step_size(config),
        clip_indices=get_clip_indices(config),
    )


def make_env(config: dict, seed: int):
    env_id = get_env_id(config)
    evaluation_mode = int(config.get("evaluation_mode", 1))

    def _init():
        instance = gym.make(
            env_id,
            config=config,
            ref_motion=build_ref_motion(config),
            seed=seed,
            evaluation_mode=evaluation_mode,
        )
        instance.render_mode = getattr(instance, "render_mode", "rgb_array")
        return instance

    return _init


class _DummyRNG:
    def __init__(self, *args, **kwargs):
        self.state = None

    def __setstate__(self, state):
        self.state = state

    def __getstate__(self):
        return {}


def _dummy_rng_ctor(*args, **kwargs):
    return _DummyRNG()


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy.random._pickle" and name in {
            "__bit_generator_ctor",
            "__generator_ctor",
            "__randomstate_ctor",
        }:
            return _dummy_rng_ctor

        if module.startswith("numpy.random") and name in {
            "Generator",
            "RandomState",
            "MT19937",
            "PCG64",
            "PCG64DXSM",
            "Philox",
            "SFC64",
        }:
            return _DummyRNG

        return super().find_class(module, name)


def load_vecnormalize_compat(path: Path, venv):
    with open(path, "rb") as f:
        vec_norm = _CompatUnpickler(f).load()
    vec_norm.set_venv(venv)
    return vec_norm


@dataclass
class EpisodeBuffer:
    sensor_samples: List[np.ndarray] = field(default_factory=list)
    targets: List[np.float32] = field(default_factory=list)
    episode_steps: List[int] = field(default_factory=list)
    mc_idx: Optional[int] = None
    start_frame: Optional[int] = None

    def append(self, info: dict):
        sensor_sample = np.asarray(info["distill/sensor_sample"], dtype=np.float32)
        target = np.float32(info["distill/knee_joint_moment"])
        episode_step = int(info["distill/episode_step"])

        if self.mc_idx is None:
            self.mc_idx = int(info["distill/mc_idx"])
        if self.start_frame is None:
            self.start_frame = int(info["distill/start_frame"])

        self.sensor_samples.append(sensor_sample.copy())
        self.targets.append(target)
        self.episode_steps.append(episode_step)

    def to_episode_payload(self, episode_id: int, window_size: int):
        if len(self.sensor_samples) < window_size:
            return None

        sensor_samples = np.stack(self.sensor_samples, axis=0)
        targets = np.asarray(self.targets, dtype=np.float32)
        episode_steps = np.asarray(self.episode_steps, dtype=np.int32)

        windows = np.stack(
            [
                sensor_samples[end_idx - window_size + 1 : end_idx + 1].T
                for end_idx in range(window_size - 1, sensor_samples.shape[0])
            ],
            axis=0,
        ).astype(np.float32)

        return {
            "inputs": windows,
            "targets": targets[window_size - 1 :].reshape(-1, 1).astype(np.float32),
            "episode_ids": np.full((windows.shape[0],), episode_id, dtype=np.int64),
            "sample_steps": episode_steps[window_size - 1 :].astype(np.int32),
            "mc_idx": np.full((windows.shape[0],), int(self.mc_idx), dtype=np.int32),
            "start_frame": np.full((windows.shape[0],), int(self.start_frame), dtype=np.int32),
            "episode_length": np.full((windows.shape[0],), len(self.sensor_samples), dtype=np.int32),
        }


class RunningStats:
    def __init__(self, num_channels: int):
        self.input_sum = np.zeros((num_channels,), dtype=np.float64)
        self.input_sq_sum = np.zeros((num_channels,), dtype=np.float64)
        self.input_count = 0
        self.target_sum = 0.0
        self.target_sq_sum = 0.0
        self.target_count = 0

    def update(self, inputs: np.ndarray, targets: np.ndarray):
        self.input_sum += inputs.sum(axis=(0, 2), dtype=np.float64)
        self.input_sq_sum += np.square(inputs, dtype=np.float64).sum(axis=(0, 2), dtype=np.float64)
        self.input_count += int(inputs.shape[0] * inputs.shape[2])

        flat_targets = targets.reshape(-1)
        self.target_sum += float(flat_targets.sum(dtype=np.float64))
        self.target_sq_sum += float(np.square(flat_targets, dtype=np.float64).sum(dtype=np.float64))
        self.target_count += int(flat_targets.shape[0])

    def to_dict(self) -> dict:
        input_mean = self.input_sum / max(self.input_count, 1)
        input_var = np.maximum(self.input_sq_sum / max(self.input_count, 1) - input_mean**2, 0.0)
        target_mean = self.target_sum / max(self.target_count, 1)
        target_var = max(self.target_sq_sum / max(self.target_count, 1) - target_mean**2, 0.0)

        return {
            "input_mean": input_mean.astype(np.float32),
            "input_std": np.sqrt(input_var).astype(np.float32),
            "target_mean": np.asarray([target_mean], dtype=np.float32),
            "target_std": np.asarray([np.sqrt(target_var)], dtype=np.float32),
            "input_count": int(self.input_count),
            "target_count": int(self.target_count),
        }


class SensorStats:
    def __init__(self, num_channels: int):
        self.sensor_sum = np.zeros((num_channels,), dtype=np.float64)
        self.sensor_sq_sum = np.zeros((num_channels,), dtype=np.float64)
        self.sensor_count = 0

    def update(self, sensor_samples: List[np.ndarray]):
        if not sensor_samples:
            return

        samples = np.asarray(sensor_samples, dtype=np.float64)
        self.sensor_sum += samples.sum(axis=0, dtype=np.float64)
        self.sensor_sq_sum += np.square(samples, dtype=np.float64).sum(axis=0, dtype=np.float64)
        self.sensor_count += int(samples.shape[0])

    def to_meta(self, config: dict) -> dict:
        sensor_mean = self.sensor_sum / max(self.sensor_count, 1)
        sensor_var = np.maximum(
            self.sensor_sq_sum / max(self.sensor_count, 1) - sensor_mean**2,
            0.0,
        )
        sensor_std = np.sqrt(sensor_var)

        return {
            "include_knee_angle": bool(config.get("include_knee_angle", True)),
            "sensor_channels": int(sensor_mean.shape[0]),
            "sensor_window": int(config.get("sensor_window", 95)),
            "sensor_mean": [float(value) for value in sensor_mean.tolist()],
            "sensor_std": [float(value) for value in sensor_std.tolist()],
        }


class ShardWriter:
    def __init__(
        self,
        save_dir: Path,
        save_prefix: str,
        shard_size: int,
        input_names: List[str],
        target_name: str,
    ):
        self.save_dir = save_dir
        self.save_prefix = save_prefix
        self.shard_size = int(shard_size)
        self.input_names = list(input_names)
        self.target_name = target_name
        self.pending = {
            "inputs": [],
            "targets": [],
            "episode_ids": [],
            "sample_steps": [],
            "mc_idx": [],
            "start_frame": [],
            "episode_length": [],
        }
        self.pending_samples = 0
        self.shard_index = 0
        self.saved_shards = []
        self.saved_samples = 0
        self.stats = RunningStats(len(self.input_names))

    def add_episode(self, payload: dict):
        self.stats.update(payload["inputs"], payload["targets"])
        for key in self.pending:
            self.pending[key].append(payload[key])
        self.pending_samples += int(payload["inputs"].shape[0])

        while self.pending_samples >= self.shard_size:
            self._flush(self.shard_size)

    def close(self):
        if self.pending_samples > 0:
            self._flush(self.pending_samples)

    def _flush(self, count: int):
        merged = {
            key: np.concatenate(self.pending[key], axis=0) if self.pending[key] else np.empty((0,))
            for key in self.pending
        }
        chunk = {key: value[:count] for key, value in merged.items()}
        remainder = {key: value[count:] for key, value in merged.items()}

        for key in self.pending:
            if remainder[key].shape[0] == 0:
                self.pending[key] = []
            else:
                self.pending[key] = [remainder[key]]

        self.pending_samples -= count
        self._write_shard(chunk)

    def _write_shard(self, chunk: dict):
        shard_path = self.save_dir / f"{self.save_prefix}_{self.shard_index:05d}.pt"
        shard_payload = {
            "inputs": torch.from_numpy(np.ascontiguousarray(chunk["inputs"])),
            "targets": torch.from_numpy(np.ascontiguousarray(chunk["targets"])),
            "episode_ids": torch.from_numpy(np.ascontiguousarray(chunk["episode_ids"])),
            "sample_steps": torch.from_numpy(np.ascontiguousarray(chunk["sample_steps"])),
            "mc_idx": torch.from_numpy(np.ascontiguousarray(chunk["mc_idx"])),
            "start_frame": torch.from_numpy(np.ascontiguousarray(chunk["start_frame"])),
            "episode_length": torch.from_numpy(np.ascontiguousarray(chunk["episode_length"])),
            "input_names": self.input_names,
            "target_name": self.target_name,
            "num_channels": int(chunk["inputs"].shape[1]),
            "window_size": int(chunk["inputs"].shape[2]),
        }
        torch.save(shard_payload, shard_path)
        self.saved_shards.append(shard_path.name)
        self.saved_samples += int(chunk["inputs"].shape[0])
        self.shard_index += 1


def ensure_output_dir(save_dir: Path, save_prefix: str):
    save_dir.mkdir(parents=True, exist_ok=True)
    existing_shards = sorted(save_dir.glob(f"{save_prefix}_*.pt"))
    if existing_shards:
        raise FileExistsError(
            f"Output directory already has shard files for prefix '{save_prefix}': {save_dir}"
        )


def load_policy(config: dict, vec_env):
    model_path = resolve_path(config["policy_path"])
    custom_objects = {
        "observation_space": vec_env.observation_space,
        "action_space": vec_env.action_space,
    }
    return SAC.load(
        model_path,
        env=vec_env,
        device=config.get("device", "auto"),
        custom_objects=custom_objects,
    )


def get_input_channel_names(config: dict) -> List[str]:
    if "input_channel_names" in config:
        return list(config["input_channel_names"])

    channel_names = ["thigh_gyro_z", "shank_gyro_z"]
    if bool(config.get("include_knee_angle", True)):
        channel_names.append("knee_angle_r")
    return channel_names


def save_run_artifacts(
    config: dict,
    save_dir: Path,
    writer: ShardWriter,
    sensor_stats: SensorStats,
    accepted_episodes: int,
    rejected_episodes: int,
):
    torch.save(writer.stats.to_dict(), save_dir / "dataset_stats.pt")
    torch.save(
        {
            "accepted_episodes": int(accepted_episodes),
            "rejected_episodes": int(rejected_episodes),
            "saved_samples": int(writer.saved_samples),
            "saved_shards": writer.saved_shards,
            "input_names": writer.input_names,
            "target_name": writer.target_name,
            "policy_path": str(resolve_path(config["policy_path"])),
            "vecnormalize_path": str(resolve_path(config["vecnormalize_path"])),
        },
        save_dir / "dataset_metadata.pt",
    )

    with open(save_dir / "meta.json", "w") as f:
        json.dump(sensor_stats.to_meta(config), f, indent=2)

    with open(save_dir / "config_snapshot.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def run(config: dict):
    save_dir = resolve_path(config["save_dir"])
    save_prefix = str(config.get("save_prefix", "knee_moment"))
    ensure_output_dir(save_dir, save_prefix)

    n_env = int(config["n_env"])
    base_seed = int(config["seed"])
    window_size = int(config.get("sensor_window", 95))
    min_episode_steps = int(config.get("min_episode_steps", window_size))
    target_episodes = int(config["num_episodes"])
    deterministic = bool(config.get("deterministic", True))

    env_fns = [make_env(config, base_seed + idx) for idx in range(n_env)]
    vec_env = SubprocVecEnv(
        env_fns,
        start_method=str(config.get("subproc_start_method", "fork")),
    )
    vec_env = load_vecnormalize_compat(resolve_path(config["vecnormalize_path"]), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = load_policy(config, vec_env)
    input_channel_names = get_input_channel_names(config)
    writer = ShardWriter(
        save_dir=save_dir,
        save_prefix=save_prefix,
        shard_size=int(config["shard_size"]),
        input_names=input_channel_names,
        target_name=str(config.get("target_name", "knee_joint_moment_nm")),
    )
    sensor_stats = SensorStats(len(input_channel_names))

    accepted_episodes = 0
    rejected_episodes = 0
    episode_buffers = [EpisodeBuffer() for _ in range(n_env)]

    obs = vec_env.reset()
    try:
        while accepted_episodes < target_episodes:
            actions, _ = model.predict(obs, deterministic=deterministic)
            obs, _, dones, infos = vec_env.step(actions)

            for env_idx, info in enumerate(infos):
                episode_buffers[env_idx].append(info)

                if not dones[env_idx]:
                    continue

                payload = episode_buffers[env_idx].to_episode_payload(
                    episode_id=accepted_episodes,
                    window_size=window_size,
                )
                episode_length = len(episode_buffers[env_idx].sensor_samples)

                if episode_length < min_episode_steps or payload is None:
                    rejected_episodes += 1
                    print(
                        f"[reject] env={env_idx} episode_length={episode_length} "
                        f"(min={min_episode_steps})"
                    )
                else:
                    writer.add_episode(payload)
                    sensor_stats.update(episode_buffers[env_idx].sensor_samples)
                    accepted_episodes += 1
                    print(
                        f"[save] episode={accepted_episodes}/{target_episodes} "
                        f"env={env_idx} episode_length={episode_length} "
                        f"samples={payload['inputs'].shape[0]}"
                    )

                episode_buffers[env_idx] = EpisodeBuffer()
                if accepted_episodes >= target_episodes:
                    break
    finally:
        writer.close()
        save_run_artifacts(
            config=config,
            save_dir=save_dir,
            writer=writer,
            sensor_stats=sensor_stats,
            accepted_episodes=accepted_episodes,
            rejected_episodes=rejected_episodes,
        )
        vec_env.close()

    print(
        f"Finished data collection. accepted_episodes={accepted_episodes}, "
        f"rejected_episodes={rejected_episodes}, saved_samples={writer.saved_samples}, "
        f"saved_shards={len(writer.saved_shards)}"
    )


if __name__ == "__main__":
    freeze_support()

    parser = ArgumentParser()
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = resolve_config_paths(yaml.safe_load(f))

    run(config)
