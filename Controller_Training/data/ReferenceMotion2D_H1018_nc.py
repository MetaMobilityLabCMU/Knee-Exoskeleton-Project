import pandas as pd
import numpy as np
from typing import Optional, Tuple, Sequence
from scipy.signal import butter, filtfilt
from pathlib import Path

from utils.sim_config import get_frame_skip


class ReferenceMotion2D_H1018_nc:
    DOF_ORDER = [
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

    END_POINT_ORDER = [
        "pelvis_X",
        "pelvis_Y",
        "calcn_r_X",
        "calcn_r_Y",
        "calcn_l_X",
        "calcn_l_Y",
    ]

    def __init__(
        self,
        ik_folder_path: str,
        analyze_folder_path: str,
        simul_step_size: float = 0.025,
        num_clips: int = 22,
        clip_indices: Optional[Sequence[int]] = None,
    ):
        self.rng = np.random.default_rng(None)
        self.simul_step_size = float(simul_step_size)
        self.ref_dt = 0.005

        self.num_clips = int(num_clips)
        self.clip_indices = tuple(range(self.num_clips)) if clip_indices is None else tuple(sorted(set(map(int, clip_indices))))
        if not self.clip_indices or any(idx < 0 or idx >= self.num_clips for idx in self.clip_indices):
            raise ValueError("clip_indices must contain valid reference motion indices")


        self.cutoff_hz = float(4)
        self.filt_order = int(6)

        self.ik_folder_path = Path(ik_folder_path)
        self.analyze_folder_path = Path(analyze_folder_path)

        self.ik_paths = {t: self.ik_folder_path / f"mc_{t}.mot" for t in self.clip_indices}
        self.analyze_paths = {
            t: self.analyze_folder_path / f"mc_{t}_BodyKinematics_pos_global.sto"
            for t in self.clip_indices
        }

        self.ik_dict = {f"mc_{t}": {"time": None, "data": None, "header": None,
                                    "pos": None, "vel": None, "dt": None, "num_frames": None}
                        for t in self.clip_indices}
        self.analyze_dict = {f"mc_{t}": {"time": None, "data": None, "header": None,
                                         "endpoints": None, "num_frames": None}
                             for t in self.clip_indices}

        self._load_all_trials()
        self._build_arrays()

    # -------------------------
    # IO
    # -------------------------
    @staticmethod
    def _read_file(path: Path) -> Tuple[list, np.ndarray, np.ndarray]:
        skiprows = None
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if "endheader" in line.lower():
                    skiprows = i + 1
                    break
        if skiprows is None:
            raise ValueError(f"'endheader' not found in file: {path}")

        cols = pd.read_csv(path, sep="\t", skiprows=skiprows, nrows=0).columns.tolist()
        df = pd.read_csv(path, sep="\t", skiprows=skiprows)

        time = df.values[:, 0].astype(np.float64)
        data = df.values[:, 1:].astype(np.float64)
        column_names = cols[1:]
        return column_names, time, data

    def _load_all_trials(self):
        for idx in self.clip_indices:
            ik_cols, ik_time, ik_data = self._read_file(self.ik_paths[idx])
            an_cols, an_time, an_data = self._read_file(self.analyze_paths[idx])

            key = f"mc_{idx}"
            self.ik_dict[key]["header"] = ik_cols
            self.ik_dict[key]["time"] = ik_time
            self.ik_dict[key]["data"] = ik_data

            self.analyze_dict[key]["header"] = an_cols
            self.analyze_dict[key]["time"] = an_time
            self.analyze_dict[key]["data"] = an_data

    # -------------------------
    # Filtering
    # -------------------------

    def _butter_lowpass_zero_lag(self, data: np.ndarray, dt: float) -> np.ndarray:
        fs = 1.0 / dt
        nyq = 0.5 * fs
        wn = self.cutoff_hz / nyq
        wn = min(max(wn, 1e-6), 0.999999)

        b, a = butter(self.filt_order, wn, btype="low")

        padlen = 3 * max(len(a), len(b))
        if data.shape[0] <= padlen:
            return data
        return filtfilt(b, a, data, axis=0)

    def _build_arrays(self):
        dt = self.ref_dt
        for clip_idx in self.clip_indices:
            key = f"mc_{clip_idx}"

            ik_cols = self.ik_dict[key]["header"]
            ik_time = self.ik_dict[key]["time"]
            ik_out = self.ik_dict[key]["data"]

            self.ik_dict[key]["dt"] = dt
            print(f"Clip {clip_idx}: dt = {dt:.6f} sec")

            pos, vel = self._compute_dof_from_output(ik_cols, ik_time, ik_out, dt)
            self.ik_dict[key]["pos"] = pos
            self.ik_dict[key]["vel"] = vel

            an_cols = self.analyze_dict[key]["header"]
            an_out = self.analyze_dict[key]["data"]

            dt_an = 0.005
            endpoints = self._compute_endpoint_from_output(an_cols, an_out, dt_an)
            self.analyze_dict[key]["endpoints"] = endpoints

            self.ik_dict[key]["num_frames"] = pos.shape[0]

        self.frames_per_sim_step = get_frame_skip(
            self.simul_step_size,
            dt,
            param_name="simul_step_size",
        )

    def _compute_dof_from_output(self, column_names, time, output, dt: float):
        num_frames = output.shape[0]
        num_dofs = len(self.DOF_ORDER)
        pos = np.zeros((num_frames, num_dofs), dtype=np.float64)

        for j, dof_name in enumerate(self.DOF_ORDER):
            col_idx = column_names.index(dof_name)
            pos[:, j] = output[:, col_idx]

        # deg → rad
        angle_indices = [0, 3, 4, 5, 6, 7, 8, 9]
        pos[:, angle_indices] = np.deg2rad(pos[:, angle_indices])

        # filtering
        pos = self._butter_lowpass_zero_lag(pos, dt=dt)

        # vel
        vel = np.gradient(pos, time, axis=0)
        return pos.astype(np.float32), vel.astype(np.float32)

    def _compute_endpoint_from_output(self, column_names, output, dt: float):
        num_frames = output.shape[0]
        num_pts = len(self.END_POINT_ORDER)
        endpoints = np.zeros((num_frames, num_pts), dtype=np.float64)

        for j, name in enumerate(self.END_POINT_ORDER):
            col_idx = column_names.index(name)
            endpoints[:, j] = output[:, col_idx]

        endpoints = self._butter_lowpass_zero_lag(endpoints, dt=dt)
        return endpoints.astype(np.float32)

    # -------------------------
    # Internal helpers
    # -------------------------
    def _clip_frame_idx(self, clip_idx: int, frame_idx: int) -> int:
        num_frames = self.ik_dict[f"mc_{clip_idx}"]["num_frames"]
        return int(np.clip(frame_idx, 0, num_frames - 1))

    def _time_to_frame(self, clip_idx: int, t: float) -> int:
        frame_idx = int(round(t / self.ref_dt))
        return self._clip_frame_idx(clip_idx, frame_idx)

    def frame_to_time(self, clip_idx: int, frame_idx: int) -> float:
        frame_idx = self._clip_frame_idx(clip_idx, frame_idx)
        return float(frame_idx * self.ref_dt)

    def _env_step_to_frame(self, clip_idx: int, start_frame: int, env_step: int) -> int:
        frame_idx = int(start_frame) + int(env_step) * self.frames_per_sim_step
        return self._clip_frame_idx(clip_idx, frame_idx)
        # -------------------------
    # Public API
    # -------------------------
    def get_ref_data(self, clip_idx: int, frame_idx: int):
        frame_idx = self._clip_frame_idx(clip_idx, frame_idx)

        key = f"mc_{clip_idx}"
        p = self.ik_dict[key]["pos"][frame_idx]
        v = self.ik_dict[key]["vel"][frame_idx]
        ed = self.analyze_dict[key]["endpoints"][frame_idx]

        i_tilt, i_tx, i_ty = 0, 1, 2
        i_hip_r, i_knee_r, i_ankle_r = 3, 4, 5
        i_hip_l, i_knee_l, i_ankle_l = 6, 7, 8
        i_lumbar = 9

        i_pelvis_X, i_pelvis_Y = 0, 1
        i_calcn_r_X, i_calcn_r_Y = 2, 3
        i_calcn_l_X, i_calcn_l_Y = 4, 5

        pos = np.array([
            p[i_tilt], p[i_tx], p[i_ty],
            p[i_hip_r], p[i_knee_r], p[i_ankle_r],
            p[i_hip_l], p[i_knee_l], p[i_ankle_l],
            p[i_lumbar]
        ])

        vel = np.array([
            v[i_tilt], v[i_tx], v[i_ty],
            v[i_hip_r], v[i_knee_r], v[i_ankle_r],
            v[i_hip_l], v[i_knee_l], v[i_ankle_l],
            v[i_lumbar]
        ])

        end_points = np.array([
            ed[i_calcn_r_X] - ed[i_pelvis_X],
            ed[i_calcn_r_Y] - ed[i_pelvis_Y],
            ed[i_calcn_l_X] - ed[i_pelvis_X],
            ed[i_calcn_l_Y] - ed[i_pelvis_Y],
        ])

        return pos, vel, end_points

    def get_reset_ref(self, clip_idx: int,  start_frame: int):
        return self.get_ref_data(clip_idx, start_frame)
    
    def sample_start_frame(self, clip_idx: int, max_episode_steps: int) -> int:
        max_episode_steps = int(max_episode_steps)

        num_frames = self.ik_dict[f"mc_{clip_idx}"]["num_frames"]

        max_offset = (max_episode_steps - 1) * self.frames_per_sim_step
        last_valid_start = num_frames - 1 - max_offset

        if last_valid_start <= 0:
            return 0

        valid_starts = np.arange(0, last_valid_start + 1, self.frames_per_sim_step, dtype=np.int32)
        return int(self.rng.choice(valid_starts))

    def sample_start_time(self, clip_idx: int, max_episode_steps: int) -> float:
        start_frame = self.sample_start_frame(clip_idx, max_episode_steps)
        return self.frame_to_time(clip_idx, start_frame)

    def get_ref_by_env_step(self, clip_idx: int, start_frame: int, env_step: int):
        frame_idx = self._env_step_to_frame(clip_idx, start_frame, env_step)
        return self.get_ref_data(clip_idx, frame_idx)
    
    def get_future_ref_data(self,clip_idx:int, start_frame: int, env_step: int, n_future_steps: int = 4):
        """
        Returns:
            future_pos:        (n_future_steps, 10)
            future_vel:        (n_future_steps, 10)
            future_end_points: (n_future_steps, 4)

        For env_step=t, this returns targets for:
            t+1, t+2, ..., t+n_future_steps
        """
        future_pos = []
        future_vel = []
        future_end_points = []

        for k in range(1, n_future_steps + 1):
            p, v, e = self.get_ref_by_env_step(clip_idx, start_frame, env_step + k)
            future_pos.append(p)
            future_vel.append(v)
            future_end_points.append(e)

        future_pos = np.stack(future_pos, axis=0)
        future_vel = np.stack(future_vel, axis=0)
        future_end_points = np.stack(future_end_points, axis=0)

        return future_pos, future_vel, future_end_points

    def get_future_ref_obs(self,clip_idx:int, start_frame: int, env_step: int, n_future_steps: int = 4) -> np.ndarray:
        """
        Flattened future reference vector for observation.

        Shape:
            n_future_steps * (10 + 10 + 4)

        Order:
            [future_pos(flat), future_vel(flat), future_end_points(flat)]
        """
        future_pos, future_vel, future_end_points = self.get_future_ref_data(
            clip_idx=clip_idx,
            start_frame=start_frame,
            env_step=env_step,
            n_future_steps=n_future_steps,
        )

        future_obs = np.concatenate([
            future_pos.reshape(-1),
            future_vel.reshape(-1),
            future_end_points.reshape(-1),
        ])

        return future_obs
