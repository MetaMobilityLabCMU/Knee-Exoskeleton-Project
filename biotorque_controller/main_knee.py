import signal
import sys

from runner import run_controller
from utils.config import load_config, parse_args, print_config


def _handle_signal(sig, _frame):
    print(f"\nSignal {sig} received. Shutting down...")
    sys.exit(0)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    print_config(config)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    run_controller(config)


if __name__ == "__main__":
    main()
