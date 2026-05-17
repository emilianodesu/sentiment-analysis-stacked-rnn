"""Root conftest.py — adds src/ to sys.path for test imports."""

import os
import sys

# Add src/ directory to Python path so tests can import modules directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
