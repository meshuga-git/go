"""Run one command and record its wall time and peak memory.

Requires Python 3.10 or newer.

Usage:
    python _measure.py OUT_JSON COMMAND [ARGS...]

The command inherits stdin, stdout and stderr. Peak memory is the
largest resident set of any process in the command's tree, measured
through ``resource`` on POSIX systems; on Windows it is reported as
null.
"""

import json
import subprocess
import sys
import time


def main() -> None:
    """Run the command, record the measurements, pass on its code."""
    out_path, command = sys.argv[1], sys.argv[2:]
    start = time.perf_counter()
    code = subprocess.call(command)
    seconds = time.perf_counter() - start
    peak_kib = None
    try:
        import resource  # noqa: PLC0415 - POSIX only

        peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        peak_kib = peak // 1024 if sys.platform == "darwin" else peak
    except ImportError:
        pass
    with open(out_path, "w", encoding="utf-8") as stream:
        json.dump(
            {"seconds": round(seconds, 3), "peak_kib": peak_kib,
             "exit": code},
            stream,
        )
    sys.exit(code)


if __name__ == "__main__":
    main()
