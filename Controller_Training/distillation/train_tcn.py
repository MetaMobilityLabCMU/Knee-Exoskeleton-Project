import json
import math
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

try:
    from tcn_model.TCN_Header_Model import TCNModel
except ModuleNotFoundError:
    from distillation.tcn_model.TCN_Header_Model import TCNModel


class ShardIterableDataset(IterableDataset):
    def __init__(
        self,
        shard_paths,
        sensor_mean,
        sensor_std,
        normalize_x=True,
        shuffle_shards=True,
        shuffle_within_shard=True,
        seed=123,
    ):
        super().__init__()
        self.shard_paths = [Path(path) for path in shard_paths]
        self.normalize_x = bool(normalize_x)
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_within_shard = bool(shuffle_within_shard)
        self.seed = int(seed)
        self.sensor_mean = torch.as_tensor(sensor_mean, dtype=torch.float32).view(1, -1, 1)
        self.sensor_std = torch.as_tensor(sensor_std, dtype=torch.float32).view(1, -1, 1)

    def _shard_list_for_this_worker(self):
        info = get_worker_info()
        if info is None:
            return list(self.shard_paths)

        num_shards = len(self.shard_paths)
        per_worker = int(math.ceil(num_shards / info.num_workers))
        start = info.id * per_worker
        end = min(start + per_worker, num_shards)
        return list(self.shard_paths[start:end])

    def __iter__(self):
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        shard_list = self._shard_list_for_this_worker()
        rng = random.Random(self.seed + worker_id)

        if self.shuffle_shards:
            rng.shuffle(shard_list)

        for shard_idx, shard_path in enumerate(shard_list):
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
            inputs = payload["inputs"].float()
            targets = payload["targets"].float()

            if targets.ndim == 1:
                targets = targets.unsqueeze(-1)

            num_samples = int(inputs.shape[0])
            if self.shuffle_within_shard and num_samples > 1:
                perm_gen = torch.Generator()
                perm_gen.manual_seed(self.seed + worker_id * 100_000 + shard_idx)
                permutation = torch.randperm(num_samples, generator=perm_gen)
                inputs = inputs[permutation]
                targets = targets[permutation]

            if self.normalize_x:
                inputs = (inputs - self.sensor_mean) / torch.clamp(self.sensor_std, min=1e-6)

            for sample_idx in range(num_samples):
                yield inputs[sample_idx], targets[sample_idx]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_torch_device(device_value: str) -> torch.device:
    value = str(device_value).lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(value)


def load_config(config_file: str) -> dict:
    with open(config_file, "r") as f:
        from utils.sim_config import resolve_config_paths
        return resolve_config_paths(yaml.safe_load(f))


def get_dataset_dir(config: dict) -> Path:
    return Path(config.get("dataset_dir", config["save_dir"]))


def get_student_run_name(config: dict) -> str:
    return str(config.get("student_run_name", config["trial_name"]))


def get_log_dir(config: dict) -> Path:
    base_dir = Path(__file__).resolve().parent
    return Path(config.get("tensorboard_dir", base_dir / "runs" / get_student_run_name(config)))


def get_model_dir(config: dict) -> Path:
    base_dir = Path(__file__).resolve().parent
    return Path(config.get("student_output_dir", base_dir / "student_model" / get_student_run_name(config)))


def list_shards(dataset_dir: Path, save_prefix: str):
    return sorted(dataset_dir.glob(f"{save_prefix}_*.pt"))


def split_train_val(shards, val_ratio=0.1, seed=123):
    shards = list(shards)
    if not shards:
        raise FileNotFoundError("No shard files were found for supervised learning.")

    if len(shards) == 1:
        return shards, shards

    rng = random.Random(seed)
    rng.shuffle(shards)

    num_shards = len(shards)
    num_val = max(1, int(round(num_shards * float(val_ratio))))
    num_val = min(num_val, num_shards - 1)

    val_shards = shards[:num_val]
    train_shards = shards[num_val:]
    return train_shards, val_shards


def _rand_uniform(shape, low, high, device, generator=None):
    return low + (high - low) * torch.rand(shape, device=device, generator=generator)


def _rand_log_uniform(shape, low, high, device, generator=None):
    low = max(low, 1e-12)
    log_low, log_high = math.log(low), math.log(high)
    u = torch.rand(shape, device=device, generator=generator)
    return torch.exp(log_low + (log_high - log_low) * u)


def add_gyro_gaussian_noise(
    xb,
    prob=0.7,
    big_prob=0.15,
    small_min=0.01,
    small_max=0.03,
    big_min=0.05,
    big_max=0.10,
    mode="iid",
    log_uniform=False,
    gyro_channels=2,
    generator=None,
):
    batch_size, num_channels, window_size = xb.shape
    device = xb.device
    gyro_channels = min(int(gyro_channels), int(num_channels))

    if gyro_channels <= 0:
        return xb, torch.zeros((batch_size,), device=device)

    apply = torch.rand((batch_size,), device=device, generator=generator) < prob
    if not apply.any():
        return xb, torch.zeros((batch_size,), device=device)

    big = torch.zeros((batch_size,), dtype=torch.bool, device=device)
    apply_idx = apply.nonzero(as_tuple=False).squeeze(1)
    if apply_idx.numel() > 0:
        big[apply_idx] = torch.rand(
            (apply_idx.numel(),),
            device=device,
            generator=generator,
        ) < big_prob

    sigma = torch.zeros((batch_size,), device=device)
    sampler = _rand_log_uniform if log_uniform else _rand_uniform

    small_idx = (apply & ~big).nonzero(as_tuple=False).squeeze(1)
    if small_idx.numel() > 0:
        sigma[small_idx] = sampler(
            (small_idx.numel(),),
            small_min,
            small_max,
            device,
            generator,
        )

    big_idx = (apply & big).nonzero(as_tuple=False).squeeze(1)
    if big_idx.numel() > 0:
        sigma[big_idx] = sampler(
            (big_idx.numel(),),
            big_min,
            big_max,
            device,
            generator,
        )

    if mode == "iid":
        eps = torch.randn((batch_size, gyro_channels, window_size), device=device, generator=generator)
        noise = eps * sigma.view(batch_size, 1, 1)
    elif mode == "channel_const":
        eps = torch.randn((batch_size, gyro_channels, 1), device=device, generator=generator)
        noise = eps * sigma.view(batch_size, 1, 1)
    elif mode == "time_const":
        eps = torch.randn((batch_size, 1, 1), device=device, generator=generator)
        noise = eps * sigma.view(batch_size, 1, 1)
    else:
        raise ValueError(f"Unknown gyro noise mode: {mode}")

    xb_out = xb.clone()
    xb_out[:, :gyro_channels, :] = xb_out[:, :gyro_channels, :] + noise
    return xb_out, sigma


def add_knee_bias(
    xb,
    sensor_std,
    normalize_x=True,
    prob=0.4,
    bias_deg_min=-3.0,
    bias_deg_max=3.0,
    knee_channel_idx=2,
    generator=None,
):
    batch_size, num_channels, _ = xb.shape
    device = xb.device
    knee_channel_idx = int(knee_channel_idx)

    if knee_channel_idx < 0 or knee_channel_idx >= num_channels:
        return xb, torch.zeros((batch_size,), device=device)

    apply = torch.rand((batch_size,), device=device, generator=generator) < prob
    if not apply.any():
        return xb, torch.zeros((batch_size,), device=device)

    bias_deg = torch.zeros((batch_size,), device=device)
    idx = apply.nonzero(as_tuple=False).squeeze(1)
    bias_deg[idx] = _rand_uniform(
        (idx.numel(),),
        bias_deg_min,
        bias_deg_max,
        device,
        generator,
    )

    bias_rad = bias_deg * (math.pi / 180.0)
    if normalize_x:
        bias_value = bias_rad / torch.clamp(sensor_std[knee_channel_idx], min=1e-6)
    else:
        bias_value = bias_rad

    xb_out = xb.clone()
    xb_out[:, knee_channel_idx : knee_channel_idx + 1, :] = (
        xb_out[:, knee_channel_idx : knee_channel_idx + 1, :] + bias_value.view(batch_size, 1, 1)
    )
    return xb_out, bias_deg


def compute_zero_output_batch_size(real_batch_size, target_ratio, device, generator=None):
    target_ratio = float(target_ratio)
    if target_ratio <= 0.0:
        return 0
    if target_ratio >= 1.0:
        raise ValueError("zero_output_data_ratio must be in [0, 1).")

    expected = (float(real_batch_size) * target_ratio) / max(1.0 - target_ratio, 1e-6)
    extra = int(math.floor(expected))
    remainder = expected - extra
    if remainder > 0.0:
        if float(torch.rand((), device=device, generator=generator).item()) < remainder:
            extra += 1
    return extra


def build_zero_output_batch(
    batch_size,
    sensor_mean,
    sensor_std,
    sensor_window,
    output_size,
    normalize_x=True,
    gyro_noise_enable=True,
    gyro_noise_prob=1.0,
    gyro_noise_big_prob=0.15,
    gyro_noise_small_min=0.01,
    gyro_noise_small_max=0.03,
    gyro_noise_big_min=0.05,
    gyro_noise_big_max=0.10,
    gyro_noise_mode="iid",
    gyro_noise_log_uniform=False,
    gyro_channels=2,
    knee_bias_enable=True,
    knee_bias_prob=1.0,
    knee_bias_deg_min=-3.0,
    knee_bias_deg_max=3.0,
    knee_channel_idx=2,
    generator=None,
):
    num_channels = int(sensor_mean.numel())
    device = sensor_mean.device
    batch_size = int(batch_size)

    xb = torch.zeros((batch_size, num_channels, sensor_window), dtype=torch.float32, device=device)
    yb = torch.zeros((batch_size, output_size), dtype=torch.float32, device=device)

    if normalize_x:
        mean = sensor_mean.view(1, -1, 1)
        std = torch.clamp(sensor_std.view(1, -1, 1), min=1e-6)
        xb = (xb - mean) / std

    if gyro_noise_enable:
        xb, _ = add_gyro_gaussian_noise(
            xb,
            prob=float(gyro_noise_prob),
            big_prob=float(gyro_noise_big_prob),
            small_min=float(gyro_noise_small_min),
            small_max=float(gyro_noise_small_max),
            big_min=float(gyro_noise_big_min),
            big_max=float(gyro_noise_big_max),
            mode=str(gyro_noise_mode),
            log_uniform=bool(gyro_noise_log_uniform),
            gyro_channels=int(gyro_channels),
            generator=generator,
        )

    if knee_bias_enable:
        xb, _ = add_knee_bias(
            xb,
            sensor_std=sensor_std,
            normalize_x=normalize_x,
            prob=float(knee_bias_prob),
            bias_deg_min=float(knee_bias_deg_min),
            bias_deg_max=float(knee_bias_deg_max),
            knee_channel_idx=int(knee_channel_idx),
            generator=generator,
        )

    return xb, yb


def save_training_snapshot(config: dict, meta: dict, model_dir: Path, log_dir: Path):
    snapshot = dict(config)
    snapshot["resolved_dataset_dir"] = str(get_dataset_dir(config))
    snapshot["resolved_model_dir"] = str(model_dir)
    snapshot["resolved_tensorboard_dir"] = str(log_dir)
    snapshot["resolved_input_size"] = int(meta["sensor_channels"])
    snapshot["resolved_window_size"] = int(meta["sensor_window"])
    snapshot["resolved_output_size"] = int(config.get("output_size", 1))

    with open(model_dir / "config_snapshot.yaml", "w") as f:
        yaml.safe_dump(snapshot, f, sort_keys=False)


def run(config: dict):
    device = resolve_torch_device(config.get("train_device", config.get("device", "auto")))
    print("device:", device)

    seed = int(config.get("seed", 123))
    set_seed(seed)

    dataset_dir = get_dataset_dir(config)
    meta_path = dataset_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    sensor_mean = meta["sensor_mean"]
    sensor_std = meta["sensor_std"]
    sensor_channels = int(meta.get("sensor_channels", len(sensor_mean)))
    sensor_window = int(meta.get("sensor_window", config.get("sensor_window", 95)))
    normalize_x = bool(config.get("normalize_x", True))
    sensor_mean_t = torch.as_tensor(sensor_mean, dtype=torch.float32, device=device)
    sensor_std_t = torch.as_tensor(sensor_std, dtype=torch.float32, device=device)

    model_config = dict(config)
    model_config["input_size"] = sensor_channels
    model_config["window_size"] = sensor_window
    model_config["output_size"] = int(config.get("output_size", 1))

    all_shards = list_shards(dataset_dir, str(config["save_prefix"]))
    train_shards, val_shards = split_train_val(
        all_shards,
        val_ratio=float(config.get("val_ratio", 0.1)),
        seed=seed,
    )
    print(f"num shards: train={len(train_shards)}, val={len(val_shards)}")

    train_ds = ShardIterableDataset(
        train_shards,
        sensor_mean=sensor_mean,
        sensor_std=sensor_std,
        normalize_x=normalize_x,
        shuffle_shards=True,
        shuffle_within_shard=True,
        seed=seed,
    )
    val_ds = ShardIterableDataset(
        val_shards,
        sensor_mean=sensor_mean,
        sensor_std=sensor_std,
        normalize_x=normalize_x,
        shuffle_shards=False,
        shuffle_within_shard=False,
        seed=seed + 999,
    )

    batch_size = int(config["batch_size"])
    num_workers = int(config.get("num_workers", 0))
    pin_memory = device.type == "cuda"

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    model = TCNModel(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    loss_fn = nn.MSELoss()

    run_name = get_student_run_name(config)
    log_dir = get_log_dir(config)
    model_dir = get_model_dir(config)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    save_training_snapshot(config, meta, model_dir, log_dir)

    epochs = int(config["epochs"])
    train_batches = int(config.get("train_batches", 0))
    val_batches = int(config.get("val_batches", 0))
    noise_enable = bool(config.get("noise_enable", False))
    zero_output_data_enable = bool(config.get("zero_output_data_enable", False))
    zero_output_data_ratio = float(config.get("zero_output_data_ratio", 0.0))
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        pbar = tqdm(
            enumerate(train_dl, start=1),
            total=(train_batches if train_batches > 0 else None),
            desc=f"Epoch {epoch}/{epochs} [train]",
            ncols=110,
            dynamic_ncols=True,
            colour="#22C55E",
        )

        for iteration, (xb, yb) in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            batch_gen = torch.Generator(device=device.type)
            batch_gen.manual_seed(seed + epoch * 100_000 + iteration)

            if noise_enable:
                xb, _ = add_gyro_gaussian_noise(
                    xb,
                    prob=float(config.get("gyro_noise_prob", 0.7)),
                    big_prob=float(config.get("gyro_noise_big_prob", 0.15)),
                    small_min=float(config.get("gyro_noise_small_min", 0.01)),
                    small_max=float(config.get("gyro_noise_small_max", 0.03)),
                    big_min=float(config.get("gyro_noise_big_min", 0.05)),
                    big_max=float(config.get("gyro_noise_big_max", 0.10)),
                    mode=str(config.get("gyro_noise_mode", "iid")),
                    log_uniform=bool(config.get("gyro_noise_log_uniform", False)),
                    gyro_channels=int(config.get("gyro_channels", min(2, sensor_channels))),
                    generator=batch_gen,
                )

                if bool(config.get("knee_bias_enable", True)):
                    xb, _ = add_knee_bias(
                        xb,
                        sensor_std=sensor_std_t,
                        normalize_x=normalize_x,
                        prob=float(config.get("knee_bias_prob", 0.4)),
                        bias_deg_min=float(config.get("knee_bias_deg_min", -3.0)),
                        bias_deg_max=float(config.get("knee_bias_deg_max", 3.0)),
                        knee_channel_idx=int(config.get("knee_channel_idx", 2)),
                        generator=batch_gen,
                    )

            if zero_output_data_enable and zero_output_data_ratio > 0.0:
                zero_batch_size = compute_zero_output_batch_size(
                    real_batch_size=xb.shape[0],
                    target_ratio=zero_output_data_ratio,
                    device=device,
                    generator=batch_gen,
                )
                if zero_batch_size > 0:
                    zero_xb, zero_yb = build_zero_output_batch(
                        batch_size=zero_batch_size,
                        sensor_mean=sensor_mean_t,
                        sensor_std=sensor_std_t,
                        sensor_window=sensor_window,
                        output_size=int(model_config["output_size"]),
                        normalize_x=normalize_x,
                        gyro_noise_enable=bool(config.get("zero_output_gyro_noise_enable", True)),
                        gyro_noise_prob=float(config.get("zero_output_gyro_noise_prob", 1.0)),
                        gyro_noise_big_prob=float(config.get("gyro_noise_big_prob", 0.15)),
                        gyro_noise_small_min=float(config.get("gyro_noise_small_min", 0.01)),
                        gyro_noise_small_max=float(config.get("gyro_noise_small_max", 0.03)),
                        gyro_noise_big_min=float(config.get("gyro_noise_big_min", 0.05)),
                        gyro_noise_big_max=float(config.get("gyro_noise_big_max", 0.10)),
                        gyro_noise_mode=str(config.get("gyro_noise_mode", "iid")),
                        gyro_noise_log_uniform=bool(config.get("gyro_noise_log_uniform", False)),
                        gyro_channels=int(config.get("gyro_channels", min(2, sensor_channels))),
                        knee_bias_enable=bool(config.get("zero_output_knee_bias_enable", True)),
                        knee_bias_prob=float(config.get("zero_output_knee_bias_prob", 1.0)),
                        knee_bias_deg_min=float(config.get("knee_bias_deg_min", -3.0)),
                        knee_bias_deg_max=float(config.get("knee_bias_deg_max", 3.0)),
                        knee_channel_idx=int(config.get("knee_channel_idx", 2)),
                        generator=batch_gen,
                    )
                    xb = torch.cat((xb, zero_xb), dim=0)
                    yb = torch.cat((yb, zero_yb), dim=0)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            if pred.ndim == 1:
                pred = pred.unsqueeze(-1)

            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))
            pbar.set_postfix(
                loss=f"{np.mean(train_losses):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

            if train_batches > 0 and iteration >= train_batches:
                break

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        model.eval()
        val_losses = []

        pbar_val = tqdm(
            enumerate(val_dl, start=1),
            total=(val_batches if val_batches > 0 else None),
            desc=f"Epoch {epoch}/{epochs} [val]",
            ncols=110,
            dynamic_ncols=True,
            colour="#60A5FA",
        )

        with torch.no_grad():
            for iteration, (xb, yb) in pbar_val:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb)
                if pred.ndim == 1:
                    pred = pred.unsqueeze(-1)

                val_loss_item = float(loss_fn(pred, yb).item())
                val_losses.append(val_loss_item)
                pbar_val.set_postfix(loss=f"{np.mean(val_losses):.4f}")

                if val_batches > 0 and iteration >= val_batches:
                    break

        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        last_path = model_dir / f"student_tcn_epoch_{epoch:03d}.pt"
        torch.save(model.state_dict(), last_path)

        if val_loss < best_val:
            best_val = val_loss
            best_path = model_dir / "student_tcn_best.pt"
            torch.save(model.state_dict(), best_path)

        print(
            f"{run_name} | epoch={epoch:03d} "
            f"train={train_loss:.6f} val={val_loss:.6f} best={best_val:.6f}"
        )

    writer.close()
    print(f"Done. Logs: {log_dir}")
    print(f"Best ckpt: {model_dir / 'student_tcn_best.pt'}")


def main():
    parser = ArgumentParser()
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()
    config = load_config(args.config_file)
    run(config)


if __name__ == "__main__":
    main()
