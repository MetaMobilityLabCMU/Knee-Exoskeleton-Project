import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from collections import defaultdict
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization, VecNormalize

class Callback(BaseCallback):
    def __init__(
        self,
        eval_env,
        eval_episodes: int = 10,
        eval_freq: int = 10_000,
        save_path: str = ".",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_episodes = eval_episodes
        self.eval_freq = eval_freq
        self.save_path = save_path

    def _on_step(self) -> bool:

        if "infos" in self.locals:
            for info in self.locals["infos"]:
                if not info:
                    continue

                for key, value in info.items():
                    if key.startswith("reward/"):
                        self.logger.record(f"train/{key}", value)

        # only run evaluation every eval_freq calls
        if self.n_calls % self.eval_freq != 0:
            return True

        # — Sync VecNormalize statistics from training env to eval_env —
        # model.get_vec_normalize_env() returns the VecNormalize wrapper
        sync_envs_normalization(self.model.get_vec_normalize_env(), self.eval_env)


        episode_rewards = []
        episode_lengths = []
        # Use defaultdict to easily accumulate reward components from evaluation
        episode_reward_components = defaultdict(list)

        for _ in range(self.eval_episodes):
            obs = self.eval_env.reset()
            done = [False]
            current_episode_reward = 0.0
            current_episode_length = 0
            # Accumulator for components in the current episode
            current_episode_components = defaultdict(float)

            while not done[0]: # Access the boolean value from the list
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, infos = self.eval_env.step(action)
                
                # We need to index since VecEnvs return lists
                current_episode_reward += reward[0]
                current_episode_length += 1
                
                # Get info dict and accumulate component values
                info = infos[0] 
                for key, value in info.items():
                    if key.startswith("reward/"):
                        current_episode_components[key] += value
            
            # After an episode, store its total reward and length
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            # And store the summed components for this episode
            for key, value in current_episode_components.items():
                episode_reward_components[key].append(value)

        # compute statistics
        mean_r   = float(np.mean(episode_rewards))
        median_r = float(np.median(episode_rewards))
        max_r    = float(np.max(episode_rewards))
        mean_l   = float(np.mean(episode_lengths))
        median_l = float(np.median(episode_lengths))
        max_l    = float(np.max(episode_lengths))

        # log to TensorBoard
        self.logger.record('eval/mean_reward',   mean_r)
        self.logger.record('eval/median_reward', median_r)
        self.logger.record('eval/max_reward',    max_r)
        self.logger.record('eval/mean_length',   mean_l)
        self.logger.record('eval/median_length', median_l)
        self.logger.record('eval/max_length',    max_l)

        for key, values_list in episode_reward_components.items():
             self.logger.record(f"eval/mean_{key.replace('reward/','')}", np.mean(values_list))
        if self.verbose:
            print(
                f"Eval @ step={self.n_calls}: "
                f"mean_r={mean_r:.2f}, median_r={median_r:.2f}, max_r={max_r:.2f}, "
                f"mean_l={mean_l:.1f}, median_l={median_l:.1f}, max_l={max_l:.1f}"
            )

        return True