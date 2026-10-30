# PEAK squat controller: code supplement

This archive contains the **100 Hz SAC teacher** and **TCN distillation** code used for the simulated squat controller. The manuscript is *“A Holistic Design Framework for a Modular Knee Exoskeleton System.”*

## Quick check

Requires Linux, Python 3.9, and a separate SCONE/Hyfydy installation with `sconepy`. From this directory:

```bash
conda activate s2s
pip install -r requirements.txt
python smoke_test.py
```

The test runs one SCONE/SAC step and one TCN prediction. Both should print `PASS`.

## Reproduce the pipeline

```bash
python train.py cfg/sac_100hz.yaml                 # train SAC (250M steps)
python rollout.py cfg/sac_100hz.yaml 248000000    # evaluate supplied SAC checkpoint
python data_collection.py cfg/tcn_100hz.yaml      # collect teacher data
python -m distillation.train_tcn cfg/tcn_100hz.yaml  # train TCN
```

The supplied files include the **248M-step SAC checkpoint**, its observation-normalization state, the **best TCN checkpoint**, and the squat reference motion. The full generated TCN training dataset is omitted; `data_collection.py` recreates it in `distillation/data/generated/`.

## Key files

| File | Purpose |
| --- | --- |
| `cfg/sac_100hz.yaml`, `cfg/tcn_100hz.yaml` | Training and collection settings (`step_size: 0.01 s`). |
| `env/`, `models/`, `data/motion_clips/` | SCONE environment, model, and squat reference. |
| `data_collection.py` | Generate TCN training windows from SAC rollouts. |
| `distillation/train_tcn.py`, `distillation/tcn_model/` | TCN training and model code. |
| `results/sl_3_100hz/checkpoints/` | SAC checkpoint and normalization state. |
| `distillation/student_model/student_tcn_best.pt` | Trained TCN weights. |
| `distillation/data/meta.json` | TCN input normalization statistics. |

The TCN uses 95 samples of thigh angular velocity, shank angular velocity, and knee angle. Its output is knee muscle moment divided by **74.5 Nm**; multiply by 74.5 to obtain Nm.

Verified in the `s2s` environment: checkpoint loading, a SCONE/SAC step, ten SAC evaluation episodes, teacher-data collection, and a short TCN training run. Full SAC retraining and hardware operation were not tested. SCONE/Hyfydy binaries are not included.
