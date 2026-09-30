#!/usr/bin/env python3
"""cuda-translator launcher for Windows (works from a checkout, no install)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cuda_translator.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
