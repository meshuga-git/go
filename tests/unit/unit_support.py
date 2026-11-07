"""Shared helpers for the unit tests.

Requires Python 3.10 or newer, with requirements.txt and pytest
installed.

Importing this module puts the package on ``sys.path``. The native
tests build with the compiler of the golden toolchain that
``TAVNIT_TOOLCHAIN`` names, or of this machine's default one.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
REPO = TESTS.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The golden harness's compiler for each toolchain.
COMPILERS = {
    "linux-gcc": "g++",
    "linux-clang": "clang++",
    "macos-clang": "clang++",
    "windows-mingw": "g++",
    "windows-msvc": "cl",
}
DEFAULT_TOOLCHAINS = {"linux": "linux-gcc", "darwin": "macos-clang",
                      "win32": "windows-mingw"}


def compiler_name() -> str:
    """Return the compiler of the toolchain under test.

    Returns:
        A command name such as ``g++``.
    """
    toolchain = os.environ.get(
        "TAVNIT_TOOLCHAIN", DEFAULT_TOOLCHAINS.get(sys.platform, "linux-gcc"))
    return COMPILERS[toolchain]


def make_tree(folder: Path, samples: int, seed: int) -> Path:
    """Write a random clustering result with the golden tree generator.

    Args:
        folder: Folder to create the ``clustering_results`` folder in.
        samples: Number of samples.
        seed: Seed of the generator.

    Returns:
        The ``clustering_results`` folder.
    """
    results = folder / "clustering_results"
    subprocess.run(
        [sys.executable, str(TESTS / "data" / "make_trees.py"), str(REPO),
         str(results), str(samples), str(seed)],
        check=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
    )
    return results
