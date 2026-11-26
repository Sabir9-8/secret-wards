#!/usr/bin/env python3
"""Secret Wards privacy firewall entrypoint.

Usage: python main.py --input <input dir> --output <empty output dir>
Pure Python standard library (3.10+), deterministic, offline, CPU-only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rakshak.pipeline import run_pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description="Secret Wards privacy firewall")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    repo_root = Path(__file__).resolve().parent
    return run_pipeline(Path(args.input), Path(args.output), repo_root)


if __name__ == "__main__":
    sys.exit(main())
