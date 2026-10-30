import random
from collections import deque
from pathlib import Path
from typing import Optional

import gym
import numpy as np
from gym.spaces import Box

from sconegym_main.sconetools import sconepy
from utils.sim_config import get_max_step_size, get_step_size


REF_DOF_ORDER = [
    "pelvis_tilt",
    "pelvis_tx",
    "pelvis_ty",
    "hip_flexion_r",
    "knee_angle_r",
    "ankle_angle_r",
    "hip_flexion_l",
    "knee_angle_l",
    "ankle_angle_l",
    "lumbar_extension",
]


class SconeEnv(gym.Env):
    def __init__(
        self,
        config: dict,
        ref_motion: "ReferenceMotion2D",
        seed: int = None,
        evaluation_mode: int = 0,
    ):
        super().__init__()

        sconepy.set_log_level(3)

        self.ref_motion = ref_motion
        self.config = config
        self.seed = 0 if seed is None else int(seed)
        self.evaluation_mode = int(evaluation_mode)
        self.current_dir = Path(__file__).resolve().parent.parent
        self.rng = random.Random(self.seed)

        self.general_model_file = str(
            Path(config.get("model_file", self.current_dir / "models" / "H0709.scone"))
        )
        self.train_mc_candidates = tuple(config.get("train_mc_idx_candidates", [10]))
        self.rollout_output_dir = config.get("rollout_output_dir")
        self.time = 0.0
        self.steps = 0
        self.episode = 0
        self.total_reward = 0.0
        self.step_size = get_step_size(config)
        self.max_step_size = get_max_step_size(config)
        self.n_future_steps = 4
        self.start_frame = 0
        self.ref_time = 0.0
        self.pelvis_xy_terminate_dist = float(config.get("pelvis_xy_terminate_dist", 0.25))


        # Settings for distillation
        self.include_knee_angle = bool(config.get("include_knee_angle", True))
        self.sensor_window = int(config.get("sensor_window", 95))
        self.gyro_femur_r_z_hist = deque(maxlen=self.sensor_window)
        self.gyro_tibia_r_z_hist = deque(maxlen=self.sensor_window)
        self.knee_angle_r_hist = deque(maxlen=self.sensor_window)


        self.init_activations_mean = float(config["init_activations_mean"])
        self.init_activations_std = float(config["init_activations_std"])

        self.Kp_lumbar = float(config["Kp_lumbar"])
        self.Kd_lumbar = float(config["Kd_lumbar"])
        self.lumb_max_ext = float(config["lumb_max_ext"])
        self.lumb_min_ext = float(config["lumb_min_ext"])
        self.lumb_max_torque = float(config["lumb_max_torque"])
        self.filt_tau = 0.10
        self.rate_clip_threshold = float(config["rate_clip_threshold"])

        self.store_next = False
        self.mc_idx = self._sample_mc_idx()

        self.model = None
        self.model_file = None
        self.prev_excitations = np.zeros(0, dtype=np.float64)
        self.prev_torque = 0.0

        self._load_model(self.general_model_file)

        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_actuators,),
            dtype=np.float32,
        )
        dummy_obs = self._get_obs()
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=dummy_obs.shape,
            dtype=np.float32,
        )

    def _get_results_output_dir(self) -> str:
        if self.rollout_output_dir:
            return str(self.rollout_output_dir)
        return sconepy.replace_string_tags("DATE_TIME.")

    def _read_sensor_pair(self):
        g1 = float(self.body_femur_r.ang_vel().z)
        g2 = float(self.body_tibia_r.ang_vel().z)
        knee_angle_r = float(self.model.dof_position_array()[self.knee_dof_idx])
        return g1, g2, knee_angle_r
    
    def _init_sensor_buffers(self):
        g1, g2, knee_angle_r = self._read_sensor_pair()
        self.gyro_femur_r_z_hist.clear()
        self.gyro_tibia_r_z_hist.clear()
        self.knee_angle_r_hist.clear()
        for _ in range(self.sensor_window):
            self.gyro_femur_r_z_hist.append(g1)
            self.gyro_tibia_r_z_hist.append(g2)
            if self.include_knee_angle:
                self.knee_angle_r_hist.append(knee_angle_r)

    def _append_sensor_buffers(self):
        g1, g2, knee_angle_r = self._read_sensor_pair()
        self.gyro_femur_r_z_hist.append(g1)
        self.gyro_tibia_r_z_hist.append(g2)
        if self.include_knee_angle:
            self.knee_angle_r_hist.append(knee_angle_r)

    def _get_sensor_sample(self) -> np.ndarray:
        g1, g2, knee_angle_r = self._read_sensor_pair()
        sample = [g1, g2]
        if self.include_knee_angle:
            sample.append(knee_angle_r)
        return np.asarray(sample, dtype=np.float32)

    def _get_sensor_window(self) -> np.ndarray:
        channels = [
            np.asarray(self.gyro_femur_r_z_hist, dtype=np.float32),
            np.asarray(self.gyro_tibia_r_z_hist, dtype=np.float32),
        ]
        if self.include_knee_angle:
            channels.append(np.asarray(self.knee_angle_r_hist, dtype=np.float32))
        return np.stack(channels, axis=0)

    def _get_knee_joint_moment(self) -> float:
        return float(self.knee_dof.muscle_moment())

    def _get_distillation_info(self) -> dict:
        return {
            "distill/episode_step": int(self.steps),
            "distill/mc_idx": int(self.mc_idx),
            "distill/start_frame": int(self.start_frame),
            "distill/window_size": int(self.sensor_window),
            "distill/num_channels": int(3 if self.include_knee_angle else 2),
            "distill/window_ready": bool(self.steps >= self.sensor_window),
            "distill/sensor_sample": self._get_sensor_sample(),
            "distill/sensor_window": self._get_sensor_window().copy(),
            "distill/knee_joint_moment": np.asarray(
                self._get_knee_joint_moment()/74.5, dtype=np.float32
            ),
        }








    def _sample_mc_idx(self) -> int:
        if self.evaluation_mode == 0:
            return int(self.rng.choice(self.train_mc_candidates))
        if self.evaluation_mode == 1:
            return int(self.config["mc_idx"])
        raise NotImplementedError("WRONG MC IDX")

    def _load_model(self, model_file: str):
        self.model_file = str(model_file)
        self.model = sconepy.load_model(self.model_file)
        self.init_dof_pos = self.model.dof_position_array().copy()
        self.init_dof_vel = self.model.dof_velocity_array().copy()
        self._init_handles()
        self.prev_excitations = np.zeros(self.n_muscles, dtype=np.float64)
        self.prev_torque = 0.0

    def _init_handles(self):
        m = self.model

        self.bodies = {b.name(): b for b in m.bodies()}
        self.joints = {j.name(): j for j in m.joints()}
        self.legs = list(m.legs())
        self.dofs = {d.name(): d for d in m.dofs()}
        self.dof_index = {name: i for i, name in enumerate(self.dofs)}
        self.model_dof_names = list(self.dofs)

        self.body_torso = self.bodies["torso"]
        self.body_pelvis = self.bodies["pelvis"]
        self.body_femur_r = self.bodies["femur_r"]
        self.body_tibia_r = self.bodies["tibia_r"]
        self.body_calcn_r = self.bodies["calcn_r"]
        self.body_femur_l = self.bodies.get("femur_l")
        self.body_tibia_l = self.bodies.get("tibia_l")
        self.body_calcn_l = self.bodies.get("calcn_l")

        self.leg_l = None
        self.leg_r = None
        for leg in self.legs:
            leg_name = leg.name() if hasattr(leg, "name") else ""
            if leg_name.endswith("_l"):
                self.leg_l = leg
            elif leg_name.endswith("_r"):
                self.leg_r = leg

        if self.leg_r is None and len(self.legs) == 1:
            self.leg_r = self.legs[0]

        self.has_left_leg = all(
            body is not None
            for body in (self.body_femur_l, self.body_tibia_l, self.body_calcn_l)
        )

        self.observed_bodies = [
            self.body_torso,
            self.body_pelvis,
            self.body_femur_r,
            self.body_tibia_r,
            self.body_calcn_r,
        ]
        if self.has_left_leg:
            self.observed_bodies.extend(
                [
                    self.body_femur_l,
                    self.body_tibia_l,
                    self.body_calcn_l,
                ]
            )

        self.lumbar_dof_idx = self.dof_index["lumbar_extension"]
        self.pelvis_tx_idx = self.dof_index["pelvis_tx"]
        self.pelvis_ty_idx = self.dof_index["pelvis_ty"]
        self.knee_dof_idx = self.dof_index["knee_angle_r"]
        self.knee_dof = self.dofs["knee_angle_r"]

        self.ref_dof_indices = np.asarray(
            [REF_DOF_ORDER.index(name) for name in self.model_dof_names],
            dtype=np.int64,
        )
        if self.has_left_leg:
            self.ref_end_indices = np.asarray([0, 1, 2, 3], dtype=np.int64)
        else:
            self.ref_end_indices = np.asarray([0, 1], dtype=np.int64)

        self.n_dofs = len(self.model_dof_names)
        self.n_muscles = len(self.model.muscles())
        self.n_actuators = len(self.model.actuators())

        self.pose_w = np.full(self.n_dofs, 2.0, dtype=np.float64)
        self.vel_w = np.full(self.n_dofs, 0.05, dtype=np.float64)

    def _stable_pd(self, q, qd, q_des, qd_des):
        dt = self.step_size
        kp = self.Kp_lumbar
        kd = self.Kd_lumbar

        g = 1.0 / (1.0 + kd * dt + kp * dt * dt)
        ksg = kp * g
        kdg = (kd + kp * dt) * g

        tau = ksg * (q_des - q) + kdg * (qd_des - qd)
        tau = np.clip(tau, -self.lumb_max_torque, self.lumb_max_torque)
        return tau

    def _pelvis_xy_ref_error(self):
        ref_pos, _, _ = self._get_ref_target(self.steps)
        cur_pos = self.model.dof_position_array().copy()

        dx = np.abs(float(cur_pos[self.pelvis_tx_idx] - ref_pos[self.pelvis_tx_idx]))
        dy = np.abs(float(cur_pos[self.pelvis_ty_idx] - ref_pos[self.pelvis_ty_idx]))
        dist = float(np.sqrt(dx * dx + dy * dy))
        return dist, dx, dy


    def _get_current_end_points(self) -> np.ndarray:
        pelvis = self.body_pelvis.com_pos()
        calcn_r = self.body_calcn_r.com_pos()
        if not self.has_left_leg:
            return np.array(
                [
                    calcn_r.x - pelvis.x,
                    calcn_r.y - pelvis.y,
                ],
                dtype=np.float64,
            )

        calcn_l = self.body_calcn_l.com_pos()
        return np.array(
            [
                calcn_r.x - pelvis.x,
                calcn_r.y - pelvis.y,
                calcn_l.x - pelvis.x,
                calcn_l.y - pelvis.y,
            ],
            dtype=np.float64,
        )

    def _project_ref_state(self, ref_pos: np.ndarray, ref_vel: np.ndarray):
        pos = np.asarray(ref_pos, dtype=np.float64)[self.ref_dof_indices].copy()
        vel = np.asarray(ref_vel, dtype=np.float64)[self.ref_dof_indices].copy()
        return pos, vel

    def _project_ref_end_points(self, ref_end: np.ndarray):
        return np.asarray(ref_end, dtype=np.float64)[self.ref_end_indices].copy()

    def _get_ref_target(self, env_step=None):
        if env_step is None:
            env_step = self.steps
        ref_pos, ref_vel, ref_end = self.ref_motion.get_ref_by_env_step(
            self.mc_idx,
            self.start_frame,
            env_step,
        )
        pos, vel = self._project_ref_state(ref_pos, ref_vel)
        end_points = self._project_ref_end_points(ref_end)
        return pos, vel, end_points

    def _get_future_ref_target(self):
        future_pos, future_vel, future_ep = self.ref_motion.get_future_ref_data(
            self.mc_idx,
            self.start_frame,
            self.steps,
            n_future_steps=self.n_future_steps,
        )

        future_pos = np.asarray(future_pos, dtype=np.float64)[:, self.ref_dof_indices].copy()
        future_vel = np.asarray(future_vel, dtype=np.float64)[:, self.ref_dof_indices].copy()
        future_ep = np.asarray(future_ep, dtype=np.float64)[:, self.ref_end_indices].copy()

        return future_pos, future_vel, future_ep

    def _get_current_ref_time(self) -> float:
        current_ref_frame = self.start_frame + self.steps * self.ref_motion.frames_per_sim_step
        self.ref_time = float(current_ref_frame * self.ref_motion.ref_dt)
        return self.ref_time

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed = int(seed)
            self.rng.seed(self.seed)

        self.mc_idx = self._sample_mc_idx()
        self._load_model(self.general_model_file)

        self.model.set_store_data(True)
        self.model.reset()
        self._init_handles()
        self.model.set_store_data(self.store_next)

        self.total_reward = 0.0
        self.steps = 0
        self.time = 0.0
        self.prev_torque = 0.0
        self.prev_excitations = np.zeros(self.n_muscles, dtype=np.float64)

        self.start_frame = self.ref_motion.sample_start_frame(
            clip_idx=self.mc_idx,
            max_episode_steps=self.max_step_size + self.n_future_steps,
        )

        ref_pos, ref_vel, _ = self.ref_motion.get_reset_ref(self.mc_idx, self.start_frame)
        dof_pos, dof_vel = self._project_ref_state(ref_pos, ref_vel)

        self.model.set_dof_positions(np.asarray(dof_pos, dtype=np.float64))
        self.model.set_dof_velocities(np.asarray(dof_vel, dtype=np.float64))

        if self.init_activations_std != 0.0:
            muscle_activations = np.clip(
                np.random.normal(
                    self.init_activations_mean,
                    self.init_activations_std,
                    size=self.n_muscles,
                ),
                0.01,
                1.0,
            )
        else:
            muscle_activations = np.full(
                self.n_muscles,
                self.init_activations_mean,
                dtype=np.float64,
            )

        self.model.init_muscle_activations(muscle_activations)
        self.model.init_state_from_dofs()
        self.model.adjust_state_for_load(0.2)
        self._init_sensor_buffers()

        obs = self._get_obs()
        info = {
            "ref/start_frame": int(self.start_frame),
            "ref/start_time": float(self.start_frame * self.ref_motion.ref_dt),
        }
        info.update(self._get_distillation_info())
        return obs, info

    def step(self, action):
        self.steps += 1
        self.time += self.step_size

        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape[0] != self.n_actuators:
            raise ValueError(
                f"Expected action dimension {self.n_actuators}, got {action.shape[0]}."
            )
        action = np.clip(action, -1.0, 1.0)

        action_muscles = 0.5 * (action[: self.n_muscles] + 1.0)
        action_lumbar = float(action[self.n_muscles])

        q = float(self.model.dof_position_array()[self.lumbar_dof_idx])
        qd = float(self.model.dof_velocity_array()[self.lumbar_dof_idx])

        q_des = np.deg2rad(
            self.lumb_min_ext
            + 0.5 * (action_lumbar + 1.0) * (self.lumb_max_ext - self.lumb_min_ext)
        )
        tau = float(self._stable_pd(q, qd, q_des, 0.0))

        alpha = self.step_size / (self.filt_tau + self.step_size)
        tau_filtered = alpha * tau + (1.0 - alpha) * self.prev_torque
        tau_filtered = np.clip(
            tau_filtered - self.prev_torque,
            -self.rate_clip_threshold,
            self.rate_clip_threshold,
        ) + self.prev_torque

        actuator_input = np.concatenate(
            [
                action_muscles,
                np.array([tau_filtered], dtype=np.float64),
            ]
        )

        self.model.set_actuator_inputs(actuator_input)
        self.model.advance_simulation_to(self.time)
        self._get_current_ref_time()
        self._append_sensor_buffers()

        terminated = self._is_terminated()
        truncated = self._is_truncated()
        done = terminated or truncated

        reward, info = self._compute_reward()
        info.update(self._get_distillation_info())
        self.total_reward += reward
        obs = self._get_obs()

        self.prev_torque = float(self.model.actuator_input_array()[-1])
        self.prev_excitations = self.model.muscle_excitation_array().copy()

        if done:
            self.episode += 1
            if self.store_next:
                self.model.write_results(
                    self._get_results_output_dir(),
                    f"{self.total_reward:.3f}",
                )
                self.store_next = False

        return obs, reward, terminated, truncated, info

    def _body_contact_xy(self, body, attribute: str) -> np.ndarray:
        value = getattr(body, attribute)()
        return np.array([value.x, value.y], dtype=np.float64)

    def _get_obs(self) -> np.ndarray:
        return self._get_current_obs()

    def _get_current_obs(self) -> np.ndarray:
        m_fibl = self.model.muscle_fiber_length_array().copy()
        m_fibv = self.model.muscle_fiber_velocity_array().copy()
        m_force = self.model.muscle_force_array().copy()
        m_exc = self.model.muscle_excitation_array().copy()
        acts = self.model.muscle_activation_array().copy()

        if self.has_left_leg and self.leg_l is not None and self.leg_r is not None:
            grf = np.array(
                [
                    self.leg_l.contact_force().x,
                    self.leg_l.contact_force().y,
                    self.leg_r.contact_force().x,
                    self.leg_r.contact_force().y,
                ],
                dtype=np.float64,
            )
        elif self.has_left_leg:
            grf = np.concatenate(
                [
                    self._body_contact_xy(self.body_calcn_l, "contact_force"),
                    self._body_contact_xy(self.body_calcn_r, "contact_force"),
                ]
            )
        else:
            if self.leg_r is not None:
                grf = np.array(
                    [
                        self.leg_r.contact_force().x,
                        self.leg_r.contact_force().y,
                    ],
                    dtype=np.float64,
                )
            else:
                grf = self._body_contact_xy(self.body_calcn_r, "contact_force")
        grf /= self.model.mass() * 9.81

        dof_values = self.model.dof_position_array().copy()
        dof_vels = self.model.dof_velocity_array().copy()

        body_contact_forces = np.concatenate(
            [self._body_contact_xy(body, "contact_force") for body in self.observed_bodies]
        ) / (self.model.mass() * 9.81)

        body_contact_points = np.concatenate(
            [self._body_contact_xy(body, "contact_point") for body in self.observed_bodies]
        )

        future_pos, future_vel, future_ep = self._get_future_ref_target()

        lumb_torque_norm = np.array(
            [self.model.actuator_input_array()[-1] / self.lumb_max_torque],
            dtype=np.float64,
        )
        mc_idx = np.array([float(self.mc_idx)], dtype=np.float64)
        com_vel = np.array(
            [
                self.model.com_vel().x,
                self.model.com_vel().y,
            ],
            dtype=np.float64,
        )

        raw_obs = np.concatenate(
            [
                m_fibl,
                m_fibv,
                m_force,
                m_exc,
                dof_values,
                dof_vels,
                acts,
                grf,
                body_contact_forces,
                body_contact_points,
                future_pos.reshape(-1),
                future_vel.reshape(-1),
                future_ep.reshape(-1),
                lumb_torque_norm,
                mc_idx,
                com_vel,
            ]
        )

        return raw_obs.astype(np.float32)

    def _compute_reward(self):
        return self._compute_deep_mimic_reward(self.config)

    def _compute_deep_mimic_reward(self, config):
        ref_pos, ref_vel, ref_end_points = self._get_ref_target(self.steps)

        cur_pos = self.model.dof_position_array().copy()
        cur_vel = self.model.dof_velocity_array().copy()
        cur_end_points = self._get_current_end_points()

        pose_err = float(np.mean(self.pose_w * (cur_pos - ref_pos) ** 2))
        vel_err = float(np.mean(self.vel_w * (cur_vel - ref_vel) ** 2))
        end_err = float(np.mean((cur_end_points - ref_end_points) ** 2))

        r_pose = float(np.exp(-pose_err))
        r_vel = float(np.exp(-vel_err))
        r_end = float(np.exp(-80.0 * end_err))

        w_pose = config["w"]["pose"]
        w_vel = config["w"]["vel"]
        w_end = config["w"]["end"]
        w_effort = config["w"]["effort"]
        w_knee = config["w"]["knee"]
        w_effort_change = config["w"]["effort_change"]

        imitation_reward = w_pose * r_pose + w_vel * r_vel + w_end * r_end

        effort_cost = self._effort_cost()
        knee_load_cost = self._knee_load_cost()
        effort_change_cost = self._effort_change_cost()

        reward = (
            imitation_reward
            + w_effort * effort_cost
            + w_knee * knee_load_cost
            + w_effort_change * effort_change_cost
        )

        info = {
            "reward/imitation_total": np.asarray(imitation_reward),
            "reward/pose": np.asarray(w_pose * r_pose),
            "reward/vel": np.asarray(w_vel * r_vel),
            "reward/end": np.asarray(w_end * r_end),
            "reward/effort_penalty": np.asarray(w_effort * effort_cost),
            "reward/knee_penalty": np.asarray(w_knee * knee_load_cost),
            "reward/effort_change_penalty": np.asarray(w_effort_change * effort_change_cost),
        }
        return float(reward), info

    def _effort_cost(self):
        activations = self.model.muscle_activation_array()
        return float(np.mean(activations ** 2))

    def _effort_change_cost(self):
        curr_excitations = self.model.muscle_excitation_array()
        curr_norm_lumb_torque = np.array(
            [self.model.actuator_input_array()[-1] / self.lumb_max_torque],
            dtype=np.float64,
        )
        prev_norm_lumb_torque = np.array(
            [self.prev_torque / self.lumb_max_torque],
            dtype=np.float64,
        )
        curr_input = np.concatenate([curr_excitations, curr_norm_lumb_torque])
        prev_input = np.concatenate([self.prev_excitations, prev_norm_lumb_torque])
        return float(np.mean((curr_input - prev_input) ** 2))

    def _knee_load_cost(self):
        knee_load = max(float(self.joints["knee_r"].load()) - 3.0, 0.0)
        viol = np.clip(knee_load ** 2, 0.0, 100.0)
        return float(viol)

    def _is_terminated(self) -> bool:
        dof_values = self.model.dof_position_array()
        dof_vels = self.model.dof_velocity_array()
        if not np.all(np.isfinite(dof_values)) or not np.all(np.isfinite(dof_vels)):
            return True

        if self.steps > 4:
            dist, _, _ = self._pelvis_xy_ref_error()
            if dist > self.pelvis_xy_terminate_dist:
                return True

        return False

    def _is_truncated(self) -> bool:
        return self.steps >= self.max_step_size

    def store_next_episode(self):
        self.store_next = True
        self.reset()

    def write_now(self):
        if self.store_next:
            self.model.write_results(
                self._get_results_output_dir(),
                f"{self.episode:05d}_{self.total_reward:.3f}",
            )
            self.store_next = False
