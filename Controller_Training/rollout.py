import pickle
import sys
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import gym
import numpy as np
import numpy.core as np_core
import numpy.core.multiarray as np_multiarray
import numpy.core.numeric as np_numeric
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

import env
from data.ReferenceMotion2D_H1018_nc import ReferenceMotion2D_H1018_nc
from utils.sim_config import get_clip_indices, get_max_step_size, get_step_size, resolve_config_paths


sys.modules.setdefault("numpy._core", np_core)
sys.modules.setdefault("numpy._core.numeric", np_numeric)
sys.modules.setdefault("numpy._core.multiarray", np_multiarray)


BASE = Path(__file__).resolve().parent
RESULT_DIR = BASE / "results"


def get_env_id(config: dict) -> str:
    return config.get("env_id", "H0709-v0")


def build_ref_motion(config: dict) -> ReferenceMotion2D_H1018_nc:
    return ReferenceMotion2D_H1018_nc(
        ik_folder_path=config["ik_folder_path"],
        analyze_folder_path=config["analyze_folder_path"],
        simul_step_size=get_step_size(config),
        clip_indices=get_clip_indices(config),
    )


def make_env(config: dict, seed: int):
    config = deepcopy(config)
    env_id = get_env_id(config)
    if "rollout_output_dir" not in config:
        config["rollout_output_dir"] = "SAC_rollout"

    def _init():
        instance = gym.make(
            env_id,
            config=config,
            ref_motion=build_ref_motion(config),
            seed=seed,
            evaluation_mode=1,
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


def load_vecnormalize_compat(path, venv):
    with open(path, "rb") as f:
        vec_norm = _CompatUnpickler(f).load()
    vec_norm.set_venv(venv)
    return vec_norm


def enable_episode_storage(eval_env) -> None:
    base_env = eval_env.envs[0].unwrapped
    base_env.store_next_episode()


def run(config: dict, check_number: int):
    eval_env = DummyVecEnv([make_env(config, seed=int(config["seed"]))])

    result_dir = RESULT_DIR / config["trial_name"]
    vecnormalize_path = result_dir / "checkpoints" / f"sac_model_vecnormalize_{check_number}_steps.pkl"
    model_path = result_dir / "checkpoints" / f"sac_model_{check_number}_steps.zip"

    if not vecnormalize_path.exists():
        raise FileNotFoundError(
            f"VecNormalize checkpoint not found: {vecnormalize_path}. "
            "Note: with VecEnv training, save_freq is measured in env steps, "
            "so the saved timestep suffix can be n_env times larger."
        )
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    eval_env = load_vecnormalize_compat(vecnormalize_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    custom_objects = {
        "observation_space": eval_env.observation_space,
        "action_space": eval_env.action_space,
    }

    model = SAC.load(
        model_path,
        env=eval_env,
        device="auto",
        custom_objects=custom_objects,
    )

    rewards_store = []
    success_rate = []
    max_step_size = get_max_step_size(config)

    for ep in range(10):
        enable_episode_storage(eval_env)
        obs = eval_env.reset()
        ep_reward = 0.0
        ep_steps = 0
        done = [False]

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            # action = [eval_env.action_space.sample()]
            obs, reward, done, info = eval_env.step(action)
            ep_reward += reward[0]
            ep_steps += 1

        success_rate.append(1 if ep_steps == max_step_size else 0)
        rewards_store.append(ep_reward)
        print(f"Rollout ended; steps = {ep_steps}; total reward = {ep_reward:.3f}")

    eval_env.close()
    print("Mean reward: ", np.mean(rewards_store))
    print(f"Success rate: {np.mean(success_rate):.4f}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config_file", type=str)
    parser.add_argument("check_number", type=int)
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = resolve_config_paths(yaml.safe_load(f))

    run(config, args.check_number)
