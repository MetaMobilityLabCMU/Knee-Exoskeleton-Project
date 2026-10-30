"""Exercise the supplied 100 Hz SAC policy and distilled TCN in SCONE."""

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from distillation.tcn_model.TCN_Header_Model import TCNModel
from data_collection import load_vecnormalize_compat
from rollout import make_env
from utils.sim_config import resolve_config_paths


ROOT = Path(__file__).resolve().parent
SAC_CHECKPOINT = ROOT / "results/sl_3_100hz/checkpoints/sac_model_248000000_steps.zip"
VEC_NORMALIZE = ROOT / "results/sl_3_100hz/checkpoints/sac_model_vecnormalize_248000000_steps.pkl"
TCN_CHECKPOINT = ROOT / "distillation/student_model/student_tcn_best.pt"


def main():
    with open(ROOT / "cfg/sac_100hz.yaml") as f:
        config = resolve_config_paths(yaml.safe_load(f))
    assert config["step_size"] == 0.01

    env = DummyVecEnv([make_env(config, seed=int(config["seed"]))])
    env = load_vecnormalize_compat(VEC_NORMALIZE, env)
    env.training = False
    env.norm_reward = False
    try:
        policy = SAC.load(
            SAC_CHECKPOINT,
            env=env,
            device="cpu",
            custom_objects={
                "observation_space": env.observation_space,
                "action_space": env.action_space,
            },
        )
        obs = env.reset()
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, done, infos = env.step(action)
        sensor_window = np.asarray(infos[0]["distill/sensor_window"], dtype=np.float32)
        assert obs.shape == (1, 149)
        assert action.shape == (1, 10)
        assert sensor_window.shape == (3, 95)
        assert np.isfinite(reward).all()
    finally:
        env.close()

    with open(ROOT / "cfg/tcn_100hz.yaml") as f:
        tcn_config = yaml.safe_load(f)
    with open(ROOT / "distillation/data/meta.json") as f:
        meta = json.load(f)
    model_config = dict(tcn_config)
    model_config.update(
        input_size=meta["sensor_channels"],
        window_size=meta["sensor_window"],
        output_size=1,
    )
    student = TCNModel(model_config)
    student.load_state_dict(torch.load(TCN_CHECKPOINT, map_location="cpu", weights_only=True))
    student.eval()
    mean = np.asarray(meta["sensor_mean"], dtype=np.float32)[:, None]
    std = np.asarray(meta["sensor_std"], dtype=np.float32)[:, None]
    model_input = torch.from_numpy(((sensor_window - mean) / np.maximum(std, 1e-6))[None])
    with torch.no_grad():
        normalized_moment = float(student(model_input).item())
    assert np.isfinite(normalized_moment)
    print(f"PASS: 100 Hz SCONE/SAC step, reward={float(reward[0]):.4f}")
    print(f"PASS: TCN inference, predicted moment={normalized_moment * 74.5:.4f} Nm")


if __name__ == "__main__":
    main()
