"""Command-line entry point for the original, corrected four-token baseline."""

import importlib.util
from pathlib import Path


def load_baseline():
    """Import the selected baseline without running an experiment."""
    path = Path(__file__).with_name("pie_usage_probe.py")
    spec = importlib.util.spec_from_file_location("pie_baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    return load_baseline().main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
