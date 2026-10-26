from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np

from controllers.base import Sensors
from utils.Jetson_Teensy import Device, JetsonCanInterface


class Side(Enum):
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"


@dataclass
class SensorFrame:
    sensors: Sensors
    pos_l: float
    pos_r: float
    imu_l1: np.ndarray
    imu_l2: np.ndarray
    imu_r1: np.ndarray
    imu_r2: np.ndarray


def _imu6_zeros() -> np.ndarray:
    return np.zeros((6,), dtype=np.float32)


def _pack_imu6(device: Device, which: int) -> np.ndarray:
    if which == 1:
        accel = device.IMU1_accel_data[-1] if device.IMU1_accel_data else (0.0, 0.0, 0.0)
        gyro = device.IMU1_gyro_data[-1] if device.IMU1_gyro_data else (0.0, 0.0, 0.0)
    else:
        accel = device.IMU2_accel_data[-1] if device.IMU2_accel_data else (0.0, 0.0, 0.0)
        gyro = device.IMU2_gyro_data[-1] if device.IMU2_gyro_data else (0.0, 0.0, 0.0)
    return np.asarray(
        [accel[0], accel[1], accel[2], gyro[0], gyro[1], gyro[2]],
        dtype=np.float32,
    )


def _latest_pos_vel(device: Device) -> Tuple[float, float]:
    pos = float(device.Motor_pos_data[-1]) if device.Motor_pos_data else 0.0
    vel = float(device.Motor_vel_data[-1]) if device.Motor_vel_data else 0.0
    return pos, vel


def read_sensor_frame(
    side: Side,
    left_device: Device,
    right_device: Device,
    left_exo: JetsonCanInterface,
    right_exo: JetsonCanInterface,
) -> SensorFrame:
    use_left = side in (Side.LEFT, Side.BOTH)
    use_right = side in (Side.RIGHT, Side.BOTH)

    pos_l = vel_l = pos_r = vel_r = 0.0
    imu_l1 = imu_l2 = imu_r1 = imu_r2 = _imu6_zeros()

    if use_left:
        left_exo.getParameters()
        pos_l, vel_l = _latest_pos_vel(left_device)
        imu_l1 = _pack_imu6(left_device, which=1)
        imu_l2 = _pack_imu6(left_device, which=2)

    if use_right:
        right_exo.getParameters()
        pos_r, vel_r = _latest_pos_vel(right_device)
        imu_r1 = _pack_imu6(right_device, which=1)
        imu_r2 = _pack_imu6(right_device, which=2)

    sensors = Sensors(
        imu_L1=imu_l1,
        imu_L2=imu_l2,
        imu_R1=imu_r1,
        imu_R2=imu_r2,
        pos_L=pos_l,
        pos_R=pos_r,
        vel_L=vel_l,
        vel_R=vel_r,
    )
    return SensorFrame(sensors, pos_l, pos_r, imu_l1, imu_l2, imu_r1, imu_r2)
