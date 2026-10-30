import os
from argparse import ArgumentParser
from multiprocessing import freeze_support
from pathlib import Path

import gym
import torch.nn as nn
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

import env
from callback.SB3Callback import Callback
from data.ReferenceMotion2D_H1018_nc import ReferenceMotion2D_H1018_nc
from utils.sim_config import get_clip_indices, get_step_size, resolve_config_paths


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


def make_env(config: dict, seed: int, evaluation_mode: int = 0):
    env_id = get_env_id(config)

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


def get_envs(config: dict):
    base_seed = int(config["seed"])
    n_env = int(config["n_env"])

    train_env_fns = [make_env(config, base_seed + idx, evaluation_mode=0) for idx in range(n_env)]
    train_env = SubprocVecEnv(train_env_fns)
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)

    eval_env = DummyVecEnv([make_env(config, base_seed + n_env, evaluation_mode=1)])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    eval_env.training = False
    eval_env.norm_reward = False

    return train_env, eval_env


def get_callback(config: dict, eval_env):
    save_dir = RESULT_DIR / config["trial_name"]
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_freq = int(config.get("checkpoint_freq", 10000))
    eval_freq = int(config.get("eval_freq", 10000))
    eval_episodes = int(config.get("eval_episodes", 20))

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix="sac_model",
        save_vecnormalize=True,
        save_replay_buffer=False,
    )

    sb3_callback = Callback(
        eval_env=eval_env,
        eval_episodes=eval_episodes,
        eval_freq=eval_freq,
        save_path=os.path.join(save_dir, "mean_median_max"),
        verbose=1,
    )
    return CallbackList([checkpoint_callback, sb3_callback])


def get_model(config: dict, train_env, save_name: str):
    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=dict(pi=config["actor_net"], qf=config["critic_net"]),
    )

    return SAC(
        "MlpPolicy",
        train_env,
        seed=int(config["seed"]),
        learning_rate=config["lr"],
        buffer_size=int(config["buffer_size"]),
        batch_size=int(config["batch_size"]),
        tau=config["tau"],
        gamma=config["gamma"],
        train_freq=int(config["train_freq"]),
        gradient_steps=int(config["gradient_steps"]),
        target_update_interval=int(config["target_update_interval"]),
        policy_kwargs=policy_kwargs,
        tensorboard_log=os.path.join(RESULT_DIR / save_name, "tensorboard_log"),
        device=config.get("device", "auto"),
        verbose=1,
    )


if __name__ == "__main__":
    freeze_support()

    parser = ArgumentParser()
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = resolve_config_paths(yaml.safe_load(f))

    train_env, eval_env = get_envs(config)
    callback = get_callback(config, eval_env)
    model = get_model(config, train_env, config["trial_name"])

    model.learn(
        total_timesteps=int(config["total_timesteps"]),
        callback=callback,
        log_interval=500,
        progress_bar=True,
    )

    save_dir = RESULT_DIR / config["trial_name"]
    save_dir.mkdir(parents=True, exist_ok=True)

    model.save(os.path.join(save_dir, "final_model"))
    train_env.save(os.path.join(save_dir, "final_model_vecnormalize.pkl"))
    train_env.close()
    eval_env.close()
