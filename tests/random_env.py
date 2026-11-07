"""Install randomly chosen newer releases of the required libraries.

Requires Python 3.10 or newer, the ``packaging`` library and access to
PyPI.

For each library in requirements.txt, picks uniformly at random among
the releases newer than its stated minimum that offer a wheel for this
interpreter and platform, installs the picks with pip, and checks that
they load and work together. Libraries built against NumPy 1 do not
load with NumPy 2, so a NumPy 2 pick limits pandas to 2.2.2 or newer
and SciPy to 1.13.0 or newer. A combination that still fails to
install or load is replaced by a new draw, up to five times.

Usage:
    python tests/random_env.py SEED
"""

from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
from packaging.version import InvalidVersion, Version

REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements.txt"
NUMPY2_FLOORS = {"pandas": Version("2.2.2"), "scipy": Version("1.13.0")}
ATTEMPTS = 5
SMOKE = (
    "import io, numpy, pandas, scipy.cluster.hierarchy\n"
    "import plotly.graph_objects as go\n"
    "frame = pandas.DataFrame({'a': numpy.arange(3.0)})\n"
    "frame.to_csv(io.StringIO(), lineterminator='\\n')\n"
    "go.Figure(go.Scatter(x=[0, 1], y=[1, 0])).to_html()\n"
)


def minimums() -> dict[str, Version]:
    """Read the stated minimum of each library.

    Returns:
        Minimum version by library name, in file order.
    """
    found = {}
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*([A-Za-z0-9_.-]+)\s*>=\s*([^\s;#]+)", line)
        if match:
            found[match.group(1).lower()] = Version(match.group(2))
    return found


def installable(name: str, floor: Version) -> list[Version]:
    """List releases newer than ``floor`` with a wheel for here.

    Args:
        name: Library name on PyPI.
        floor: Stated minimum version.

    Returns:
        Matching versions in ascending order; ``[floor]`` when none is
        newer.
    """
    url = f"https://pypi.org/pypi/{name}/json"
    with urllib.request.urlopen(url, timeout=60) as response:
        releases = json.load(response)["releases"]
    supported = set(sys_tags())
    found = []
    for text, files in releases.items():
        try:
            version = Version(text)
        except InvalidVersion:
            continue
        if version.is_prerelease or version <= floor:
            continue
        for item in files:
            if item.get("yanked") or not item["filename"].endswith(".whl"):
                continue
            tags = parse_wheel_filename(item["filename"])[3]
            if supported & set(tags):
                found.append(version)
                break
    return sorted(found) or [floor]


def draw(rng: random.Random, choices: dict[str, list[Version]]
         ) -> dict[str, Version]:
    """Pick one version per library, respecting the NumPy 2 floors.

    Args:
        rng: Random source.
        choices: Candidate versions by library.

    Returns:
        The picked version by library.
    """
    picked = {"numpy": rng.choice(choices["numpy"])}
    for name, versions in choices.items():
        if name == "numpy":
            continue
        floor = NUMPY2_FLOORS.get(name)
        if floor is not None and picked["numpy"].major >= 2:
            versions = [v for v in versions if v >= floor] or versions
        picked[name] = rng.choice(versions)
    return picked


def main() -> int:
    """Draw, install and check versions until a combination works.

    Returns:
        Process exit code.
    """
    seed = int(sys.argv[1])
    choices = {name: installable(name, floor)
               for name, floor in minimums().items()}
    for attempt in range(ATTEMPTS):
        picked = draw(random.Random(f"{seed}-{attempt}"), choices)
        pins = [f"{name}=={version}" for name, version in picked.items()]
        print(f"Attempt {attempt + 1}: {' '.join(pins)}", flush=True)
        installed = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "--only-binary", ":all:", *pins])
        if installed.returncode != 0:
            continue
        if subprocess.run([sys.executable, "-c", SMOKE]).returncode == 0:
            print(f"Installed {' '.join(pins)}")
            return 0
    print("No working combination found.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
