import argparse
import yaml


HIGHLIGHT_KEYS = {
    "exo_on",
    "scale",
    "controller_name",
    "mass",
    "biotorque_gain",
    "torque_limit",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dual knee exoskeleton control loop.")
    parser.add_argument("config", help="Path to YAML config file.")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(config).__name__}")
    return config


def print_config(config: dict) -> None:
    print("=== CONFIG ===")
    for key, value in config.items():
        if key in HIGHLIGHT_KEYS:
            print(f"\033[92m{key}: {value}\033[0m")
        else:
            print(f"{key}: {value}")
    input("\n====Check Config====\nHit Enter to continue...")
