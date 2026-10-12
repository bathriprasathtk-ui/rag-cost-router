"""Vercel serverless entry point.

Vercel's Python runtime bundles this function and mounts the exported ASGI
app. The project root is added to sys.path so `app.*` imports resolve.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app  # noqa: E402

__all__ = ["app"]
