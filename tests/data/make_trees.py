"""Generate a random clustering result for the history builder.

Requires Python 3.10 or newer.

The result has the layout ``run_clustering`` writes: ``filenames.txt``,
``tree_splits.txt``, one ``iter<k>.txt`` per split, and a
``clustering.provenance`` record made with the package's own writer.
Sample paths contain commas, double quotes and non-ASCII characters;
the files they name need not exist. Every number comes from
``random.Random.random``, so a seed gives the same tree on every
Python version.

Usage:
    python make_trees.py PACKAGE_DIR OUT_DIR SAMPLES SEED

PACKAGE_DIR holds ``clustering.py`` and ``compute_variances.py``;
OUT_DIR becomes the ``clustering_results`` folder.
"""

import random
import sys
from pathlib import Path


def main() -> None:
    """Write one random tree and its provenance record."""
    package, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    n, seed = int(sys.argv[3]), int(sys.argv[4])
    sys.path.insert(0, str(package))
    import compute_variances  # noqa: PLC0415 - the package under test

    rng = random.Random(seed)

    def below(k: int) -> int:
        return min(int(rng.random() * k), k - 1)

    out.mkdir(parents=True, exist_ok=True)
    paths = [
        f'/data/run,{seed}/grp"{i % 3}"/échantillon_{i:05d}_μ.csv'
        for i in range(n)
    ]
    (out / "filenames.txt").write_bytes(
        "".join(p + "\n" for p in paths).encode("utf-8")
    )
    assign = [0] * n
    members = {0: list(range(n))}
    active = [0]
    splits = ["root 0"]
    outputs = [out / "filenames.txt", out / "tree_splits.txt"]
    for k in range(1, n):
        candidates = [c for c in active if len(members[c]) > 1]
        parent = candidates[below(len(candidates))]
        group = members.pop(parent)
        active.remove(parent)
        for i in range(len(group) - 1, 0, -1):
            j = below(i + 1)
            group[i], group[j] = group[j], group[i]
        cut = 1 + below(len(group) - 1)
        for child, part in ((2 * k - 1, group[:cut]), (2 * k, group[cut:])):
            members[child] = part
            active.append(child)
            for s in part:
                assign[s] = child
        splits.append(f"split {k} {parent} {2 * k - 1} {2 * k}")
        path = out / f"iter{k}.txt"
        path.write_bytes((" ".join(map(str, assign)) + "\n").encode())
        outputs.append(path)
    (out / "tree_splits.txt").write_bytes(
        ("\n".join(splits) + "\n").encode()
    )
    compute_variances.write_provenance(
        out / "clustering.provenance", stage="clustering", outputs=outputs,
        producer=package / "clustering.py",
    )


if __name__ == "__main__":
    main()
