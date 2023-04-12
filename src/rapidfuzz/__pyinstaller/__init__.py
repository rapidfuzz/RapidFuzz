from __future__ import annotations

from pathlib import Path


def get_hook_dirs():
    return [str(Path(__file__).parent)]


def get_PyInstaller_tests():
    return [str(Path(__file__).parent)]
