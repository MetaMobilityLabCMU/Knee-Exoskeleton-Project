import math
from pathlib import Path


DEFAULT_STEP_SIZE = 0.025

REPO_ROOT = Path(__file__).resolve().parent.parent
PATH_KEYS = (
    "model_file", "ik_folder_path", "analyze_folder_path",
    "policy_path", "vecnormalize_path", "save_dir", "dataset_dir",
    "student_output_dir", "tensorboard_dir",
)


def resolve_config_paths(config: dict) -> dict:
    """Resolve paths in a YAML config relative to the repository root."""
    resolved = dict(config)
    for key in PATH_KEYS:
        if key in resolved:
            path = Path(resolved[key]).expanduser()
            resolved[key] = str(path if path.is_absolute() else REPO_ROOT / path)
    return resolved


def get_clip_indices(config: dict) -> list:
    """Load only the reference clips used by training or evaluation."""
    return sorted({int(config.get("mc_idx", 10)), *map(int, config.get("train_mc_idx_candidates", [10]))})


def get_step_size(config: dict) -> float:
    step_size = float(config.get("step_size", DEFAULT_STEP_SIZE))
    if step_size <= 0.0:
        raise ValueError(f"step_size must be positive, got {step_size}.")
    return step_size


def get_max_step_size(config: dict) -> int:
    step_size = get_step_size(config)

    if "episode_duration_sec" in config:
        episode_duration_sec = float(config["episode_duration_sec"])
        if episode_duration_sec <= 0.0:
            raise ValueError(
                f"episode_duration_sec must be positive, got {episode_duration_sec}."
            )

        max_step_size = int(round(episode_duration_sec / step_size))
        if max_step_size <= 0:
            raise ValueError(
                "Resolved max_step_size must be positive. "
                f"episode_duration_sec={episode_duration_sec}, step_size={step_size}."
            )
        return max_step_size

    if "max_step_size" not in config:
        raise KeyError("Config must define either 'max_step_size' or 'episode_duration_sec'.")

    max_step_size = int(config["max_step_size"])
    if max_step_size <= 0:
        raise ValueError(f"max_step_size must be positive, got {max_step_size}.")
    return max_step_size


def get_frame_skip(step_size: float, ref_dt: float, *, param_name: str = "step_size") -> int:
    ratio = float(step_size) / float(ref_dt)
    rounded = int(round(ratio))

    if rounded <= 0 or not math.isclose(ratio, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"{param_name}={step_size} is incompatible with ref_dt={ref_dt}. "
            "The ratio must be a positive integer."
        )

    return rounded
