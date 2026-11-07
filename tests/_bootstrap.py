"""Run one pipeline script with test-only hooks.

Requires Python 3.10 or newer.

Usage:
    python _bootstrap.py [--seed N] [--hashlog PATH] SCRIPT [ARGS...]

``--seed`` gives NumPy's unseeded ``default_rng()`` a fixed seed, so
Tjala is reproducible under the test harness while the program itself
stays unseeded. ``--hashlog`` counts calls to the shared SHA-256 helper
in ``pipeline_io.py`` per file and writes the counts to PATH as
JSON when the script exits. The script otherwise runs as it would with
``python SCRIPT ARGS``.
"""

import atexit
import collections
import json
import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    """Install the requested hooks, then run the script as main."""
    args = sys.argv[1:]
    seed = None
    hashlog = None
    while args and args[0] in ("--seed", "--hashlog"):
        if args[0] == "--seed":
            seed = int(args[1])
        else:
            hashlog = Path(args[1])
        args = args[2:]
    target = os.path.abspath(args[0])
    sys.path.insert(0, os.path.dirname(target))

    if seed is not None:
        import numpy as np  # noqa: PLC0415 - only needed for this hook

        unseeded = np.random.default_rng

        def seeded(seed_arg=None, *rest, **options):
            """Substitute the harness seed for a missing one."""
            chosen = seed if seed_arg is None else seed_arg
            return unseeded(chosen, *rest, **options)

        np.random.default_rng = seeded

    shared = None
    if hashlog is not None:
        import compute_variances as shared  # noqa: PLC0415
        import pipeline_io  # noqa: PLC0415

        counts: collections.Counter[str] = collections.Counter()
        real_hash = pipeline_io.sha256_file

        def counting(path, *rest, **options):
            """Count one hash of ``path`` and delegate."""
            counts[str(Path(path).resolve())] += 1
            return real_hash(path, *rest, **options)

        def dump() -> None:
            """Write the counts collected so far."""
            hashlog.write_text(
                json.dumps(dict(sorted(counts.items())), indent=1),
                encoding="utf-8",
            )

        # Patch the helper where it lives and where it is re-exported.
        pipeline_io.sha256_file = counting
        shared.sha256_file = counting
        atexit.register(dump)

    sys.argv = [target, *args[1:]]
    if shared is not None and Path(target).name == "compute_variances.py":
        # Running the file as __main__ would create a second module
        # object that the hook does not see; call its entry point on
        # the hooked module instead, as its own __main__ block does.
        sys.exit(shared.main(args[1:]))
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
