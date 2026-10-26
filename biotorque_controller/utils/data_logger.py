import numpy as np

from controllers.base import CtrlResult
from utils.sensors import SensorFrame


LOG_FIELDS = (
    "time",
    "knee_angle_r",
    "knee_angle_l",
    "knee_angle_r_u",
    "knee_angle_l_u",
    "gyro_thigh_r_x",
    "gyro_thigh_r_y",
    "gyro_thigh_r_z",
    "gyro_shank_r_x",
    "gyro_shank_r_y",
    "gyro_shank_r_z",
    "gyro_thigh_l_x",
    "gyro_thigh_l_y",
    "gyro_thigh_l_z",
    "gyro_shank_l_x",
    "gyro_shank_l_y",
    "gyro_shank_l_z",
    "imu_thigh_r_x",
    "imu_thigh_r_y",
    "imu_thigh_r_z",
    "imu_shank_r_x",
    "imu_shank_r_y",
    "imu_shank_r_z",
    "imu_thigh_l_x",
    "imu_thigh_l_y",
    "imu_thigh_l_z",
    "imu_shank_l_x",
    "imu_shank_l_y",
    "imu_shank_l_z",
    "cmd_L",
    "cmd_R",
    "biotorque_raw_r",
    "biotorque_raw_l",
    "biotorque_delayed_r",
    "biotorque_delayed_l",
    "biotorque_filtered_r",
    "biotorque_filtered_l",
    "assist_gate_r",
    "assist_gate_l",
    "GPIO",
)


class DataLogger:
    def __init__(self, duration_sec: float, sample_rate_hz: float):
        capacity = int(duration_sec * sample_rate_hz)
        self.data = {field: np.zeros(capacity) for field in LOG_FIELDS}
        self.sample_count = 0

    def record(
        self,
        elapsed: float,
        frame: SensorFrame,
        result: CtrlResult,
        cmd_l: float,
        cmd_r: float,
        gpio_state: int,
    ) -> None:
        if self.sample_count < len(self.data["time"]):
            index = self.sample_count
            self.data["time"][index] = elapsed
            self.data["knee_angle_r"][index] = frame.pos_r
            self.data["knee_angle_l"][index] = -frame.pos_l
            self.data["knee_angle_r_u"][index] = result.extra.get("knee_angle_r_u", 0.0)
            self.data["knee_angle_l_u"][index] = result.extra.get("knee_angle_l_u", 0.0)

            for side, thigh_imu, shank_imu, sign in (
                ("r", frame.imu_r1, frame.imu_r2, 1.0),
                ("l", frame.imu_l1, frame.imu_l2, -1.0),
            ):
                for segment, imu in (("thigh", thigh_imu), ("shank", shank_imu)):
                    for index_axis, axis in enumerate(("x", "y", "z")):
                        self.data[f"imu_{segment}_{side}_{axis}"][index] = imu[index_axis]
                        self.data[f"gyro_{segment}_{side}_{axis}"][index] = (
                            sign * imu[index_axis + 3]
                        )

            self.data["cmd_L"][index] = cmd_l
            self.data["cmd_R"][index] = cmd_r
            for name in (
                "biotorque_raw_r",
                "biotorque_raw_l",
                "biotorque_delayed_r",
                "biotorque_delayed_l",
                "biotorque_filtered_r",
                "biotorque_filtered_l",
                "assist_gate_r",
                "assist_gate_l",
            ):
                self.data[name][index] = result.extra.get(name, 0.0)
            self.data["GPIO"][index] = gpio_state

        self.sample_count += 1

    def save(self, trial_name: str) -> None:
        data = {
            key: values[: self.sample_count]
            for key, values in self.data.items()
        }
        np.savez(trial_name, **data)
