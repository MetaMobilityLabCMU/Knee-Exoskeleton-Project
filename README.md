# Knee-Exoskeleton-Project
Full Exoskeleton Repo for MetaMobility Lab Open-Source Knee Exoskeleton

## Related Paper:
https://rajiv-j.github.io/Open-Source-Knee-Exo/

## What this repository contains:
1. CAD Files + STL Files
1. Controller Training Code
1. Low-Level MCU Code
1. Low-Level PCB Gerber Files
1. High-Level Jetson Code
1. Bill Of Materials

## Updates on the way:
1. High-Level PCB Gerber Files
1. Jetson/Battery Belt Holder
1. Documentation for building and assembly

## How to use:

### How the system fits together
Each leg has its own **Teensy 4.1** (low-level) that reads two IMUs (thigh and shank) and drives one **CubeMars AK80-9** knee motor. A single **NVIDIA Jetson Orin Nano** (high-level) talks to both Teensys over a shared CAN FD bus, runs the learned knee-torque controller, and sends a torque command back to each leg.

```
 Jetson Orin Nano  (biotorque_controller/, 100 Hz)
   │  reads IMU + encoder data, runs the TCN model (TensorRT), sends torque
   │
   └── CAN FD bus (1 Mbps / 2 Mbps data) ─┬─ Left  Teensy 4.1  (ID 0x100)
                                          └─ Right Teensy 4.1  (ID 0x200)
                                                │  (Low_Level_Code/, 1 kHz)
                                                ├─ Thigh IMU + Shank IMU (SPI, ICM-20948)
                                                └─ AK80-9 motor (CAN, 1 Mbps)
```

The controller itself was trained in simulation (`Controller_Training/`): a reinforcement-learning "teacher" policy learns squat assistance in a SCONE/Hyfydy musculoskeletal model, and it is then distilled into a small TCN "student" network that only needs the sensors the exoskeleton has (thigh gyro, shank gyro, knee angle).

### What is in each folder
| Folder / File | What it is | When you need it |
| --- | --- | --- |
| `Knee_Exo_BOM_Interactive.html`, `Knee_Exo_BOM_PDF.pdf` | Bill of materials with prices, quantities and purchase links (Mechanical, Electrical, Low-Level PCB, High-Level PCB). Quantities are for **one leg**. | Ordering parts |
| `Hardware/STL_Bambu/` | Print-ready STL files and Bambu Studio `.3mf` plates for the bars, cuffs and PCB cover (printed in PAHT-CF). | Printing the frame |
| `Hardware/Solidworks/` | Full SolidWorks assembly (`KneeExo_V1_Assembly.SLDASM`), individual parts, fasteners, the motor STEP and the PCB layout STEP. | Modifying the design or checking how it goes together |
| `Low_Level_Code/` | PlatformIO firmware for the Teensy 4.1: motor control, IMU reading, CAN/CAN FD comms, and a torque watchdog. | Flashing each leg's Teensy |
| `biotorque_controller/` | Python runtime for the Jetson: connects to both Teensys, runs the TensorRT controller, logs data. Has its own detailed `README.md`. | Running the exoskeleton |
| `Controller_Training/` | Simulation training code (SAC teacher + TCN distillation) and the trained checkpoints. Has its own `README.md`. | Only if you want to retrain or change the controller |

### Getting started (high level)
1. **Order the parts.** Open `Knee_Exo_BOM_Interactive.html` in a browser (or use the PDF). Everything is listed per leg; for a bilateral (two-leg) build add +1 motor, +2 IMUs and +1 Teensy. The PCB Gerber files are still on the way (see *Updates on the way*).
2. **Print and assemble the frame.** Print the parts in `Hardware/STL_Bambu/` (the `.3mf` files are pre-arranged plates for Bambu printers). The cuff and shank-bar files are for the right leg; use the SolidWorks files in `Hardware/Solidworks/` as the reference for how everything fits together. Assembly documentation is coming.
3. **Flash the low-level firmware (one Teensy per leg).** Open `Low_Level_Code/` in VS Code with the PlatformIO extension and upload to the Teensy 4.1 (the needed Adafruit libraries install automatically from `platformio.ini`). Before uploading, check the settings at the top of `src/main.cpp`:
   - `Teensy_ID`: `0x100` for the left leg, `0x200` for the right leg (these must match `teensy_id_left` / `teensy_id_right` in the Jetson config).
   - `motor_1_CAN_ID` and `motor_1_type`: must match your motor (default `0x04`, `"AK80-9"`).
   - The firmware runs at 1 kHz and sets torque to zero if it doesn't receive a command from the Jetson for 100 ms.
4. **Set up the Jetson and run the controller.** Copy `biotorque_controller/` to the Jetson and follow `biotorque_controller/README.md`. In short:
   ```bash
   python3 -m pip install numpy PyYAML python-can Jetson.GPIO
   python3 main_knee.py cfg/biotorque/biotorque.yaml
   ```
   All run settings (subject mass, assistance gain, torque limit, which leg(s), CAN channel, start trigger) live in `cfg/biotorque/biotorque.yaml`. Each run saves a `.npz` data log named after `trial_name`. The included TensorRT engine (`tcn_model/*.trt`) must match your Jetson's TensorRT version; if it fails to load, it needs to be rebuilt on your Jetson from the trained TCN (`Controller_Training/distillation/student_model/student_tcn_best.pt`).
5. **(Optional) Retrain or modify the controller.** See `Controller_Training/README.md`. This needs Linux, Python 3.9 and a separate SCONE/Hyfydy install. Run `python smoke_test.py` first to confirm your setup, then train the SAC teacher, collect data, and train the TCN. The trained checkpoints are already included, so this step isn't needed just to run the exoskeleton.

### Safety
- The config ships with `exo_on: false`, so **no torque is applied**. Run like this first and confirm the IMU, encoder and torque-command values look right for both legs before setting `exo_on: true`.
- Start with a low `biotorque_gain` and check `torque_limit` (default 18 Nm, the AK80-9 peak) before testing on a person.
- Keep the emergency stop (E-STOP in the BOM) wired and within reach during every test.


## For assistance/help:
Email: Rajiv Joshi - rvj@andrew.cmu.edu or rajivj2002@gmail.com

Email Header: "Open-Source Knee Exoskeleton: [what email pertains to]"