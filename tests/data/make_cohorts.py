"""Generate a synthetic two-cohort expression dataset.

Requires Python 3.10 or newer.

The generator draws every number from ``random.Random.random``, the
one method whose output Python guarantees for a given seed across
versions, so the same arguments always produce the same bytes.

The dataset exercises the input cases the pipeline handles:

* Three genes appear only in the cancer matrix and three only in the
  noncancer matrix, so cross-sample alignment zero-fills them.
* One gene is zero in every sample, so preparation drops it.
* The last cancer sample repeats the previous one exactly, so the
  clustering engine meets two identical samples.

Usage:
    python make_cohorts.py OUT_DIR [--cancer N] [--noncancer N]
        [--genes N] [--seed N]

The defaults create the 22-sample golden set: 12 cancer and 10
noncancer samples over 400 shared genes, seed 7.
"""

import argparse
import math
import random
from pathlib import Path

ONLY_CANCER = 3
ONLY_NONCANCER = 3


class Draws:
    """Derive the needed distributions from ``random()`` alone."""

    def __init__(self, seed: int) -> None:
        """Seed the underlying generator.

        Args:
            seed: Integer seed.
        """
        self._rng = random.Random(seed)

    def uniform(self, low: float, high: float) -> float:
        """Return a float drawn uniformly from ``[low, high)``.

        Args:
            low: Lower bound.
            high: Upper bound.

        Returns:
            The drawn value.
        """
        return low + (high - low) * self._rng.random()

    def below(self, n: int) -> int:
        """Return an integer drawn uniformly from ``0 .. n - 1``.

        Args:
            n: Number of possible values.

        Returns:
            The drawn value.
        """
        return min(int(self._rng.random() * n), n - 1)

    def gauss(self, mean: float, sigma: float) -> float:
        """Return a normal draw through the Box-Muller transform.

        Args:
            mean: Mean of the distribution.
            sigma: Standard deviation.

        Returns:
            The drawn value.
        """
        u1 = 1.0 - self._rng.random()
        u2 = self._rng.random()
        radius = math.sqrt(-2.0 * math.log(u1))
        return mean + sigma * radius * math.cos(2.0 * math.pi * u2)


def gene_rows(draws: Draws, n_genes: int) -> list[tuple[str, float]]:
    """Create gene identifiers and their baseline expression levels.

    Args:
        draws: Random source.
        n_genes: Number of genes to create.

    Returns:
        ``(gene_id, base_level)`` pairs.
    """
    return [
        (f"ENSG{g:011d}.{1 + draws.below(9)}", draws.uniform(0.0, 300.0))
        for g in range(n_genes)
    ]


def write_matrix(
    path: Path,
    prefix: str,
    n_samples: int,
    genes: list[tuple[int, str, float]],
    shift: float,
    draws: Draws,
    repeat_last: bool,
) -> None:
    """Write one genes-by-samples CSV matrix.

    Args:
        path: Destination file.
        prefix: Sample-name prefix, such as ``T``.
        n_samples: Number of samples.
        genes: ``(index, gene_id, base_level)`` triples; a negative
            base level marks a gene that is zero everywhere.
        shift: Expression shift added to every seventh gene.
        draws: Random source.
        repeat_last: Whether the last sample copies the one before.
    """
    width = max(2, len(str(n_samples)))
    names = [f"{prefix}{i:0{width}d}" for i in range(1, n_samples + 1)]
    lines = ["gene," + ",".join(names)]
    for index, gene, base in genes:
        values = []
        for s in range(n_samples):
            if base < 0.0:
                value = 0.0
            elif repeat_last and s == n_samples - 1:
                value = values[-1]
            else:
                bump = shift if index % 7 == 0 else 0.0
                sub = 40.0 if (s % 3 == 0 and index % 5 == 0) else 0.0
                value = max(0.0, draws.gauss(base + bump + sub, 25.0))
            values.append(value)
        lines.append(gene + "," + ",".join(f"{v:.3f}" for v in values))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def main() -> None:
    """Parse arguments and write both cohort matrices."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--cancer", type=int, default=12)
    parser.add_argument("--noncancer", type=int, default=10)
    parser.add_argument("--genes", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    draws = Draws(args.seed)
    extra = ONLY_CANCER + ONLY_NONCANCER + 1
    rows = gene_rows(draws, args.genes + extra)
    indexed = [(i, gene, base) for i, (gene, base) in enumerate(rows)]
    shared = indexed[: args.genes]
    only_cancer = indexed[args.genes : args.genes + ONLY_CANCER]
    only_noncancer = indexed[args.genes + ONLY_CANCER : -1]
    index, gene, _ = indexed[-1]
    all_zero = [(index, gene, -1.0)]

    write_matrix(
        args.out_dir / "cancer" / "tumor.csv", "T", args.cancer,
        shared + only_cancer + all_zero, 120.0, draws, repeat_last=True,
    )
    write_matrix(
        args.out_dir / "noncancer" / "normal.csv", "N", args.noncancer,
        shared + only_noncancer + all_zero, 0.0, draws, repeat_last=False,
    )


if __name__ == "__main__":
    main()
