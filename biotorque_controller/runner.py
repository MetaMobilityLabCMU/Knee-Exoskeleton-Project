import time
from typing import Optional

from controllers import build_controller
from controllers.base import CtrlResult
from utils.Jetson_Teensy import Device, JetsonCanInterface, configure_can_interface
from utils.data_logger import DataLogger
from utils.Header_Mocap_trigger import Mocap_trigger
from utils.gpio_control import TrialGPIO
from utils.sensors import Side, read_sensor_frame
from utils.teleplot import Teleplot
from utils.rate_keeper import RateKeeper


class DualKneeRunner:
    def __init__(self, config: dict):
        self.config = config
        self.logger = DataLogger(config["exp_time_sec"], config["fs"])
        self.gpio = TrialGPIO(config["GPIO_OUTPUT_PIN"])
        self.left_device: Optional[Device] = None
        self.right_device: Optional[Device] = None
        self.left_exo: Optional[JetsonCanInterface] = None
        self.right_exo: Optional[JetsonCanInterface] = None
        self.controller = None
        self.teleplot: Optional[Teleplot] = None
        self.mocap_trigger: Optional[Mocap_trigger] = None

    def setup(self) -> None:
        if self.config["trigger_type"] == "mocap":
            try:
                self.mocap_trigger = Mocap_trigger(
                    server_ip="172.24.44.177",
                    port_number=10,
                )
                self.mocap_trigger.start_client()
            except Exception as error:
                print(f"[Mocap] start_client error: {error}")

        configure_can_interface(channel=self.config["can_channel"])
        self.teleplot = Teleplot(
            self.config["teleplot_ip"],
            self.config["teleplot_port"],
        )

        self.left_device = Device()
        self.right_device = Device()
        self.left_exo = JetsonCanInterface(
            device_storage=self.left_device,
            channel=self.config["can_channel"],
            teensy_id=self.config["teensy_id_left"],
        )
        self.right_exo = JetsonCanInterface(
            device_storage=self.right_device,
            channel=self.config["can_channel"],
            teensy_id=self.config["teensy_id_right"],
        )

        print(f"Initializing Left Exoskeleton (ID {hex(self.config['teensy_id_left'])})...")
        self.left_exo.connect()
        print(f"Initializing Right Exoskeleton (ID {hex(self.config['teensy_id_right'])})...")
        self.right_exo.connect()

        self.controller = build_controller(
            self.config["controller_name"],
            config=self.config,
        )
        self.controller.start()

        print("\n--- Dual Knee Exo Control Loop Started ---")
        print(f"Teleplot: {self.config['teleplot_ip']}:{self.config['teleplot_port']}")
        print("Press Ctrl+C to stop.")

    def shutdown(self) -> None:
        try:
            if self.left_exo:
                self.left_exo.setTorque(0.0)
            if self.right_exo:
                self.right_exo.setTorque(0.0)
        except Exception as error:
            print(f"[Shutdown] setTorque(0) failed: {error}")

        time.sleep(0.1)

        for label, exo in (
            ("left", self.left_exo),
            ("right", self.right_exo),
        ):
            try:
                if exo:
                    exo.close()
            except Exception as error:
                print(f"[Shutdown] {label}_exo.close failed: {error}")

        try:
            if self.controller:
                self.controller.close()
        except Exception as error:
            print(f"[Shutdown] controller.close failed: {error}")

        try:
            self.gpio.pulse_end()
            self.gpio.close()
        except Exception as error:
            print(f"[Exit] GPIO cleanup error: {error}")

        try:
            if self.teleplot:
                self.teleplot.close()
        except Exception as error:
            print(f"[Shutdown] Teleplot close failed: {error}")

        print("System Shutdown Complete.")

    def _wait_for_start_trigger(self) -> None:
        if self.config["trigger_type"] == "typing":
            input("Press Enter to start...\n")
        elif self.config["trigger_type"] == "mocap":
            if self.mocap_trigger is not None:
                self.mocap_trigger.wait_for_trigger()
            else:
                print("[WARN] mocap_trigger is None. Starting immediately.")
        else:
            raise NotImplementedError(
                f"Unknown trigger_type: {self.config['trigger_type']}"
            )

    def _limit_commands(self, result: CtrlResult, side: Side) -> tuple[float, float]:
        use_left = side in (Side.LEFT, Side.BOTH)
        use_right = side in (Side.RIGHT, Side.BOTH)
        cmd_l = float(result.applied_L) if use_left else 0.0
        cmd_r = float(result.applied_R) if use_right else 0.0

        if not self.config["exo_on"]:
            cmd_l = 0.0
            cmd_r = 0.0

        scale = self.config["scale"]
        limit = self.config["torque_limit"]
        cmd_l = max(-limit, min(limit, cmd_l * scale))
        cmd_r = max(-limit, min(limit, cmd_r * scale))
        return cmd_l, cmd_r

    def _apply_commands(self, side: Side, cmd_l: float, cmd_r: float) -> None:
        if side in (Side.LEFT, Side.BOTH):
            if self.left_exo is None:
                raise RuntimeError("Left exoskeleton interface is not initialized")
            self.left_exo.setTorque(-cmd_l)
        if side in (Side.RIGHT, Side.BOTH):
            if self.right_exo is None:
                raise RuntimeError("Right exoskeleton interface is not initialized")
            self.right_exo.setTorque(cmd_r)

    def _send_telemetry(
        self,
        result: CtrlResult,
        cmd_l: float,
        cmd_r: float,
        loop_time: float,
    ) -> None:
        teleplot = self.teleplot
        if teleplot is None:
            raise RuntimeError("Teleplot is not initialized")

        values = {
            "cmd_R": cmd_r,
            "cmd_L": cmd_l,
            "knee_angle_r": result.extra.get("knee_angle_r", 0.0),
            "knee_angle_l": result.extra.get("knee_angle_l", 0.0),
            "assist_gate_r": result.extra.get("assist_gate_r", 0.0),
            "assist_gate_l": result.extra.get("assist_gate_l", 0.0),
            "biotorque_raw_r": result.extra.get("biotorque_raw_r", 0.0),
            "biotorque_raw_l": result.extra.get("biotorque_raw_l", 0.0),
            "GPIO": self.gpio.state(),
            "loop_time": loop_time,
        }
        for name, value in values.items():
            teleplot.sendValue(name, value)

    def run(self) -> None:
        if (
            self.left_device is None
            or self.right_device is None
            or self.left_exo is None
            or self.right_exo is None
            or self.controller is None
        ):
            raise RuntimeError("Runner setup must complete before starting the control loop")

        left_device = self.left_device
        right_device = self.right_device
        left_exo = self.left_exo
        right_exo = self.right_exo
        controller = self.controller

        self._wait_for_start_trigger()
        side = Side(self.config["side"])
        rate_keeper = RateKeeper(self.config["fs"])
        rate_keeper.start()
        trial_start = time.perf_counter()

        while True:
            rate_keeper.wait()
            step_start = time.perf_counter()

            frame = read_sensor_frame(
                side,
                left_device,
                right_device,
                left_exo,
                right_exo,
            )
            result = controller.step(frame.sensors)
            cmd_l, cmd_r = self._limit_commands(result, side)
            self._apply_commands(side, cmd_l, cmd_r)

            elapsed = step_start - trial_start
            self.gpio.update_trial_pulses(
                elapsed=elapsed,
                start_delay=self.config["GPIO_START_DELAY_SEC"],
                trial_duration=self.config["exp_time_sec"],
                pulse_width=self.config["PULSE_WIDTH_SEC"],
            )

            step_end = time.perf_counter()
            self._send_telemetry(
                result,
                cmd_l,
                cmd_r,
                step_end - step_start,
            )
            self.logger.record(
                elapsed,
                frame,
                result,
                cmd_l,
                cmd_r,
                self.gpio.state(),
            )

    def save_log(self) -> None:
        self.logger.save(self.config["trial_name"])
        print(f"=== Saving data for trial: {self.config['trial_name']} ===")


def run_controller(config: dict) -> None:
    runner = DualKneeRunner(config)
    try:
        runner.setup()
        runner.run()
    finally:
        runner.shutdown()
        runner.save_log()
