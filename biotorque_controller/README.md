# BioTorque Knee Controller Setup Guide

This folder contains the files needed to run the BioTorque knee exoskeleton controller on a Jetson, using the local `main_knee.py` runner and the CAN driver in `utils/Jetson_Teensy.py`.

Important: this is a runtime folder for a specific Jetson setup. Before running it in torque mode, verify the hardware and software are compatible with the target Jetson and the connected exoskeleton.

## Code layout

- `main_knee.py` is the command-line entry point.
- `runner.py` coordinates startup, the control loop, shutdown, and saving the log.
- `controllers/` contains the BioTorque controller and TensorRT inference worker.
- `utils/config.py` loads and displays the YAML settings.
- `utils/sensors.py` converts CAN device readings into controller sensor inputs.
- `utils/gpio_control.py` handles GPIO output and trial pulses.
- `utils/data_logger.py` records and saves the `.npz` trial data.
- `utils/Jetson_Teensy.py` contains CAN communication with the Teensy devices.
- `utils/rate_keeper.py` keeps the control loop on its configured sample interval.
- `utils/teleplot.py` and `utils/Header_Mocap_trigger.py` provide telemetry and mocap trigger support.

## 1. Copy the folder to the Jetson

Copy the entire `biotorque_controller` directory to the target Jetson, for example:

```bash
scp -r /path/to/biotorque_controller user@jetson:/home/user/
```

Then log in to the Jetson:

```bash
ssh user@jetson
```

## 2. Go to the project directory

```bash
cd /home/user/biotorque_controller
```

## 3. Check the Jetson environment

Before installing anything, confirm the Jetson OS and Python are what you expect:

```bash
cat /etc/nv_tegra_release
python3 --version
```

This controller depends on JetPack-provided CUDA, TensorRT, and PyTorch. Do not replace these with generic pip wheels from PyPI unless you know the Jetson package is compatible.

## 4. Install required Python packages

From the project folder, install the Python requirements used by this controller:

```bash
python3 -m pip install numpy PyYAML python-can Jetson.GPIO
```

If a package is already available on the Jetson, installation may say it is already satisfied.

## 5. Verify CAN and GPIO access

This system uses the Linux CAN interface and GPIO pins. Confirm the Jetson can configure the CAN interface and that the user has permission to run the setup commands.

Check the interface name in the config file:

```bash
cat cfg/biotorque/biotorque.yaml
```

The config includes:
- `can_channel: "can0"`
- `GPIO_OUTPUT_PIN: 7`
- `teensy_id_left: 256`
- `teensy_id_right: 512`

If your hardware uses a different CAN interface or pin setup, update the config before running.

## 6. Confirm the TensorRT model is valid

The project expects the model in:

```text
./tcn_model/s2s_controller_042326.trt
```

This file must match the Jetson’s actual CUDA/TensorRT environment. If it was created on a different Jetson or with a different TensorRT version, it may fail to load or may need to be rebuilt.

## 7. Check the config before running

Open `cfg/biotorque/biotorque.yaml` and verify:

- `exo_on` is set correctly
- `trigger_type` matches the test method
- `mass` is correct for the subject
- `biotorque_gain`, `delay`, `torque_limit` are valid
- `teensy_id_left` and `teensy_id_right` match the connected Teensy units
- `can_channel` matches the Jetson CAN interface
- `GPIO_OUTPUT_PIN` matches the real pin used for the trigger output
- `trt_engine_path` points to the correct model file

Important: the supplied config has `exo_on: false`, which keeps the controller from applying torque. That is the safe default for initial setup and verification.

## 8. Safe startup sequence

Start with the controller in non-torque mode, then verify the data stream and hardware before enabling force.

Run:

```bash
python3 main_knee.py cfg/biotorque/biotorque.yaml
```

With `trigger_type: "mocap"`, the runner expects the motion-capture trigger server at `172.24.44.177:10`. Make sure that server is available before starting. The current trigger code does not provide robust recovery if the connection or trigger stream fails.

## 9. What the controller does when started

The main runner will:

- configure the CAN interface
- create the left and right exoskeleton interfaces
- open a Teleplot output
- connect to the left/right Teensy CAN devices
- start the BioTorque controller
- begin reading IMU and motor data
- send commanded torque to the exoskeleton if `exo_on` is true
- save a `.npz` data log using the config’s `trial_name`

The main loop continues until stopped with `Ctrl+C`; `exp_time_sec` does not automatically stop it. It sets the log capacity and schedules the GPIO end pulse.

## 10. Before enabling torque

Only after the system is stable should you do the following:

1. Confirm the CAN bus is receiving data from both Teensies.
2. Confirm encoder position and velocity values are reasonable.
3. Confirm IMU signals look valid.
4. Confirm the controller is producing commanded torque values.
5. Set `exo_on: true` only when the hardware and safety checks are complete.

## 11. Typical commands

To run with the supplied config:

```bash
cd /home/user/biotorque_controller
python3 main_knee.py cfg/biotorque/biotorque.yaml
```

To inspect the config:

```bash
nano cfg/biotorque/biotorque.yaml
```

## 12. Common issues

### TensorRT engine load fails

- Check the Jetson architecture and TensorRT version.
- Rebuild the model if needed.
- Verify the `trt_engine_path` in the config.

### CAN connection fails

- Check the CAN interface is up (`can0` or the configured interface).
- Confirm the hardware and Teensy IDs match the config.
- Ensure the Jetson user has permission to run the CAN setup commands.

### GPIO / trigger issues

- Check the output pin number and physical wiring.
- Make sure the GPIO library is installed and the Jetson pin mapping is correct.
- Confirm the trigger type matches the experiment setup.

## 13. Safety

- Keep `exo_on: false` until you are certain the system is correct.
- Verify torque limits and command scaling before any active testing.
- Do not run the controller without checking the hardware and data stream first.

## 14. If you want to change the controller

This folder is set up to run the BioTorque controller only. If you want a different controller, you would need to edit the controller registry and re-create the needed files.

## 15. Quick summary

1. Copy the `biotorque_controller` folder to the Jetson.
2. Install the Python dependencies.
3. Ensure JetPack/CUDA/TensorRT/PyTorch are compatible.
4. Confirm CAN, GPIO, and trigger settings.
5. Run in non-torque mode first.
6. Verify sensors and data.
7. Enable torque only after a safe check.

This is the safest known startup path for this setup.
