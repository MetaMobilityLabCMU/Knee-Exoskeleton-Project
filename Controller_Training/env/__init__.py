"""Register the SCONE squat environment used by the SAC teacher."""

from gym.envs.registration import register

register(id="H0709-v0", entry_point="env.scone_env:SconeEnv")
