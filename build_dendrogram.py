"""Interactive visualization of immutable clustering trees.

The script consumes ``total_data.csv`` from the immutable
``reconstruct_hierarchy.cpp`` and ``tree_splits.txt`` from the immutable
``run_clustering.cpp``.  The CSV has one filename/path column plus N
hierarchy-state columns for N samples: explicit root state 0 followed by
the state after each of the N-1 binary splits.  Repeated values down a
state column are run-length compressed as blank fields and are
forward-filled before validation.

``tree_splits.txt`` is the authoritative topology.  Root node 0 is
permanent; split k retires one active parent and creates two fresh
immutable child IDs ``2k-1`` and ``2k``.  A full N-leaf binary tree
therefore contains ``2N-1`` immutable nodes with IDs ``0..2N-2``.  The
script cross-validates every CSV state against that topology, maps
immutable nodes explicitly into SciPy's separate linkage-node ID space,
and renders a responsive interactive Plotly dendrogram.

Requires Python 3.10 or later. Inputs and outputs follow the documented
working-directory contract; generated inputs require valid provenance.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy.typing import NDArray
from scipy.cluster.hierarchy import dendrogram, is_valid_linkage

from compute_variances import (
    CANNOT_LINKS_FILE,
    DENDROGRAM_FILE,
    MUST_LINKS_FILE,
    NO_LINKS_FILE,
    PARSED_DATA_PROVENANCE_FILE,
    TREE_STATES_FILE,
    TREE_TOPOLOGY_FILE,
    TreeSplit,
    TreeTopology,
    provenance_path,
    read_tree_states,
    read_tree_topology,
    read_utf8_lines,
    validate_tree_states,
    verify_artifacts,
    verify_provenance,
    write_provenance,
    write_text_atomically,
)

# SciPy's default leaf spacing convention (leaf i sits at LEAF_OFFSET +
# i * LEAF_SPACING along the x-axis, left to right, for
# orientation='top').
LEAF_OFFSET = 5.0
LEAF_SPACING = 10.0

# Color shown for a leaf that never matched any configured folder. Using
# a single fixed constant (rather than inventing a random color on
# demand) keeps "no folder matched" visually distinct and consistent
# everywhere it appears, instead of looking like a real, meaningful
# color.
UNMATCHED_COLOR = "#000000"


class DendrogramResult(TypedDict):
    """Subset of scipy.cluster.hierarchy.dendrogram's return value.

    Attributes:
        icoord: Four x coordinates for each SciPy dendrogram link.
        dcoord: Four y coordinates for each SciPy dendrogram link.
        ivl: SciPy leaf-display labels in rendered order.
        leaves: SciPy leaf indices in rendered order, not immutable node
            IDs.
        color_list: Colors assigned to the corresponding dendrogram
            links.
    """

    icoord: list[list[float]]
    dcoord: list[list[float]]
    ivl: list[str]
    leaves: list[int]
    color_list: list[str]


# ======================================================================
# Block 1: Colors and Source Categories
# ======================================================================


def _relative_luminance(hex_color: str) -> float:
    """Return WCAG relative luminance for a ``#RRGGBB`` color.

    Args:
        hex_color: Six-digit hexadecimal color string.

    Returns:
        Relative luminance in the range 0..1.
    """
    value = hex_color.lstrip("#")
    channels = [int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]

    def linearize(channel: float) -> float:
        """Convert one sRGB channel to linear intensity."""
        if channel <= 0.04045:
            return channel / 12.92
        return ((channel + 0.055) / 1.055) ** 2.4

    red, green, blue = (linearize(channel) for channel in channels)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast_ratio_with_white(hex_color: str) -> float:
    """Return the WCAG contrast ratio between a color and white."""
    return 1.05 / (_relative_luminance(hex_color) + 0.05)


def generate_random_color(rng: random.Random) -> str:
    """Generate a color that is legible on a white background.

    Colors are accepted only when their WCAG contrast ratio against
    white is at least 4.5:1. This keeps branches, markers, and legend
    colors visible even when categories are assigned randomly.

    Args:
        rng: Seeded ``random.Random`` instance.

    Returns:
        A contrast-safe hexadecimal color string.
    """
    for _ in range(1000):
        color = "#{:02X}{:02X}{:02X}".format(
            rng.randint(0, 180),
            rng.randint(0, 180),
            rng.randint(0, 180),
        )
        if _contrast_ratio_with_white(color) >= 4.5:
            return color
    return "#1F2937"

def load_folder_categories(config_paths: list[Path]) -> list[Path]:
    """Read source categories with distinct directory identities.

    Args:
        config_paths: Optional constraint-configuration paths.

    Returns:
        Distinct resolved source folders and visible immediate
        subfolders.

    Raises:
        ValueError: If a present configuration has invalid text
            encoding.
        OSError: If a present source folder cannot be inspected.
    """
    folders: list[Path] = []
    for config in config_paths:
        if not config.is_file():
            print((
                f'Warning: optional configuration missing: '
                f'{config}'
            ), file=sys.stderr)
            continue
        for raw in read_utf8_lines(config, allow_blank=True, allow_empty=True):
            if not raw or raw.isspace():
                continue
            folder = Path(raw).resolve()
            if not folder.is_dir():
                continue
            folders.append(folder)
            folders.extend(
                item.resolve() for item in sorted(folder.iterdir(),
                    key=lambda item: item.name)
                if item.is_dir() and not item.name.startswith(".")
            )
    return list(dict.fromkeys(folders))


def _folder_name_matches_segment(folder_name: str, path_segment: str) -> bool:
    """Test whether a path segment identifies a source folder.

    Matches an exact segment (segment == folder_name), or a segment that
    starts with the folder name immediately followed by a
    non-alphanumeric separator (e.g. folder "cancer" matches segments
    "cancer_parsed_data" and "cancer_TCGA-...csv", since both start with
    "cancer" + "_").

    This lets a config file use short, legend-friendly folder names
    (e.g. "cancer") even when the actual data's directory/file names
    carry a longer, suffixed form (e.g. "cancer_parsed_data"), without
    resorting to a plain substring search -- which would be unsafe here,
    since one category's name can be a substring of another's (e.g.
    "cancer" is a substring of "noncancer_parsed_data"). Requiring the
    match to start at position 0 and be followed by a separator (not
    another letter/digit) means "noncancer_parsed_data" can never match
    folder name "cancer": it doesn't *start* with "cancer" at all,
    regardless of what it contains.

    Args:
        folder_name: The configured folder's name (dict key in
            ``folder_name_to_color``).
        path_segment: One path segment from a sample's filename.

    Returns:
        True if the segment should be attributed to this folder.
    """
    if path_segment == folder_name:
        return True
    if path_segment.startswith(folder_name):
        next_char = path_segment[len(folder_name) : len(folder_name) + 1]
        return next_char != "" and not next_char.isalnum()
    return False


def map_clusters_to_colors(
    filenames: pd.Series, leaf_labels: pd.Series,
    folders: list[Path], rng: random.Random,
) -> tuple[dict[int, str], dict[str, str]]:
    """Assign colors to distinct source-directory identities.

    Full directory identity determines category membership. A duplicate
    basename receives a full-path legend label. Ingestion provenance
    identifies immediate-subfolder sources after parsed files are
    flattened.

    Args:
        filenames: Sample paths in hierarchy row order.
        leaf_labels: Immutable terminal IDs in the same order.
        folders: Configured full source-directory paths.
        rng: Generator used for category colors.

    Returns:
        Leaf colors and human-readable legend-label-to-color mapping.

    Raises:
        ValueError: If lengths, source metadata, or leaf identities
            conflict.
        OSError: If available source provenance cannot be verified.
    """
    if len(filenames) != len(leaf_labels):
        raise ValueError("Sample paths and leaf IDs have different lengths")
    folders = list(dict.fromkeys(folder.resolve() for folder in folders))
    counts = Counter(folder.name for folder in folders)
    labels = {
        folder: folder.name if counts[folder.name] == 1 else str(folder)
        for folder in folders
    }
    colors = {folder: generate_random_color(rng) for folder in folders}
    legend = {labels[folder]: color for folder, color in colors.items()}
    source_maps: dict[Path, dict[str, str]] = {}
    leaves: dict[int, str] = {}
    unmatched: list[str] = []
    for index, raw in enumerate(filenames):
        sample = Path(str(raw))
        source_manifest = sample.with_name(PARSED_DATA_PROVENANCE_FILE)
        matched: Path | None = None
        if source_manifest.is_file():
            if source_manifest not in source_maps:
                record = verify_provenance(source_manifest)
                source_maps[source_manifest] = record.categories
            source = source_maps[source_manifest].get(sample.name)
            if source is not None and Path(source).resolve() in colors:
                matched = Path(source).resolve()
        if matched is None:
            candidates = []
            for folder in folders:
                parsed = folder.with_name(folder.name + "_parsed_data")
                if sample.parent.resolve() in {folder, parsed}:
                    candidates.append(folder)
            if len(candidates) == 1:
                matched = candidates[0]
        if matched is None:
            segments = [part for part in re.split(r"[\\/]+", str(raw)) if part]
            candidates = [
                folder for folder in folders if counts[folder.name] == 1
                and any(_folder_name_matches_segment(folder.name, part)
                    for part in segments)
            ]
            if len(candidates) == 1:
                matched = candidates[0]
        leaf = int(leaf_labels.iloc[index])
        color = colors[matched] if matched is not None else UNMATCHED_COLOR
        if leaf in leaves and leaves[leaf] != color:
            raise ValueError(f"Cluster #{leaf} has conflicting source colors")
        leaves[leaf] = color
        if matched is None:
            unmatched.append(str(raw))
    if unmatched:
        print((
            f'Warning: {len(unmatched)} samples have no '
            f'unambiguous source category'
        ), file=sys.stderr)
    return leaves, legend


# ======================================================================
# Block 2: Immutable Tree and Linkage
# ======================================================================


def load_tree_topology(path: Path, sample_count: int) -> TreeTopology:
    """Read the shared immutable topology for dendrogram generation.

    Args:
        path: Authoritative tree-topology path.
        sample_count: Number of terminal sample leaves.

    Returns:
        Common validated topology with explicit immutable node IDs.

    Raises:
        ValueError: If fewer than two samples or invalid splits are
            supplied.
        OSError: If the topology cannot be read.
    """
    if sample_count < 2:
        raise ValueError("A dendrogram requires at least two samples")
    return read_tree_topology(path, sample_count)


def _decode_tree_states(
    compressed_matrix: pd.DataFrame, topology: TreeTopology
) -> pd.DataFrame:
    """Cross-check decoded states against the topology.

    Args:
        compressed_matrix: State matrix decoded by ``read_tree_states``.
        topology: Validated topology for the same samples.

    Returns:
        Validated integer matrix in original history row order.

    Raises:
        ValueError: If state transitions or dimensions disagree.
    """
    history = compressed_matrix.copy()
    data = history.copy()
    data.insert(0, "0", [str(i) for i in range(len(data))])
    validate_tree_states(data, topology)
    return history


def build_linkage_matrix(
    topology: TreeTopology,
    leaf_colors: dict[int, str],
    rng: random.Random,
) -> tuple[NDArray[np.float64], dict[int, str], dict[int, int]]:
    """Build a SciPy linkage matrix directly from immutable topology.

    Immutable node IDs and SciPy linkage IDs are distinct namespaces.
    Terminal immutable leaves are mapped to SciPy leaves ``0..N-1``;
    replaying splits in reverse creates SciPy internal nodes
    ``N..2N-2``.  The returned mapping preserves every immutable node ID
    for plotting and inspection.

    Args:
        topology: Validated immutable topology.
        leaf_colors: Colors keyed by immutable terminal leaf ID.
        rng: Seeded random generator used only as a defensive color
            fallback.

    Returns:
        ``(Z, linkage_colors, tree_to_linkage)``.

    Raises:
        ValueError: If the topology cannot be represented as a valid
            linkage.
    """
    sample_count = len(topology.leaves)
    if topology.node_count != 2 * sample_count - 1:
        raise ValueError("Immutable topology does not have 2N-1 nodes.")

    tree_to_linkage: dict[int, int] = {
        leaf_id: linkage_leaf_id
        for linkage_leaf_id, leaf_id in enumerate(topology.leaves)
    }
    linkage_colors: dict[int, str] = {
        linkage_leaf_id: leaf_colors.get(leaf_id, UNMATCHED_COLOR)
        for leaf_id, linkage_leaf_id in tree_to_linkage.items()
    }
    node_sizes: dict[int, int] = {leaf: 1 for leaf in range(sample_count)}
    z_rows: list[list[float]] = []

    def ensure_linkage_color(linkage_id: int) -> str:
        """Return the color assigned to a plotted linkage node."""
        if linkage_id not in linkage_colors:
            linkage_colors[linkage_id] = generate_random_color(rng)
        return linkage_colors[linkage_id]

    for split in reversed(topology.splits):
        try:
            left_node = tree_to_linkage[split.retained_child]
            right_node = tree_to_linkage[split.extracted_child]
        except KeyError as exc:
            raise ValueError(
                f"Cannot reconstruct immutable parent {split.parent}: child "
                f"{int(exc.args[0])} has not been resolved."
            ) from exc

        left_size = node_sizes[left_node]
        right_size = node_sizes[right_node]
        merged_size = left_size + right_size
        row_index = len(z_rows)
        new_linkage_id = sample_count + row_index
        merge_height = float(sample_count - split.iteration)
        z_rows.append(
            [
                float(left_node),
                float(right_node),
                merge_height,
                float(merged_size),
            ]
        )

        winner = left_node if left_size >= right_size else right_node
        linkage_colors[new_linkage_id] = ensure_linkage_color(winner)
        node_sizes[new_linkage_id] = merged_size
        tree_to_linkage[split.parent] = new_linkage_id

    expected_immutable = set(range(topology.node_count))
    if set(tree_to_linkage) != expected_immutable:
        missing = sorted(expected_immutable - set(tree_to_linkage))
        raise ValueError(
            f"Not every immutable node was mapped to SciPy; missing {missing}."
        )
    expected_scipy = set(range(topology.node_count))
    if set(tree_to_linkage.values()) != expected_scipy:
        raise ValueError(
            "Immutable-to-SciPy node mapping is not one-to-one and complete."
        )
    if tree_to_linkage[topology.root] != topology.node_count - 1:
        raise ValueError((
            'Immutable root did not map to the final SciPy '
            'root node.'
        ))

    z_matrix = np.asarray(z_rows, dtype=np.float64)
    if z_matrix.shape != (sample_count - 1, 4):
        raise ValueError(
            f"Incomplete linkage: expected {sample_count - 1} merges, built "
            f"{z_matrix.shape[0]}."
        )
    is_valid_linkage(z_matrix, throw=True, name="z_matrix")
    return z_matrix, linkage_colors, tree_to_linkage


def _linkage_tree_depth(
    z_matrix: NDArray[np.float64], sample_count: int
) -> int:
    """Return the maximum root-to-leaf depth of a linkage matrix."""
    depths = np.zeros(2 * sample_count - 1, dtype=np.int64)
    for row_index, row in enumerate(z_matrix):
        left = int(row[0])
        right = int(row[1])
        depths[sample_count + row_index] = max(depths[left], depths[right]) + 1
    return int(depths[-1])

def _linkage_node_coordinates(
    z_matrix: NDArray[np.float64],
    dendro_result: DendrogramResult,
    sample_count: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute coordinates for leaves and internal linkage nodes.

    Args:
        z_matrix: SciPy linkage matrix.
        dendro_result: Geometry returned by SciPy's ``dendrogram``
            function.
        sample_count: Number of leaf samples.

    Returns:
        Two arrays of length ``2 * sample_count - 1`` containing x and y
        coordinates for every SciPy node id.

    Raises:
        ValueError: If the leaf order or linkage references are
            inconsistent.
    """
    leaves = [int(leaf) for leaf in dendro_result["leaves"]]
    if len(leaves) != sample_count or set(leaves) != set(range(sample_count)):
        raise ValueError((
            'Dendrogram leaves do not match the linkage leaf '
            'ids.'
        ))

    node_count = 2 * sample_count - 1
    x_coords = np.full(node_count, np.nan, dtype=np.float64)
    y_coords = np.full(node_count, np.nan, dtype=np.float64)

    for position, leaf_id in enumerate(leaves):
        x_coords[leaf_id] = LEAF_OFFSET + LEAF_SPACING * position
        y_coords[leaf_id] = 0.0

    for row_index, row in enumerate(z_matrix):
        left = int(row[0])
        right = int(row[1])
        linkage_id = sample_count + row_index
        if np.isnan(x_coords[left]) or np.isnan(x_coords[right]):
            raise ValueError(
                f"Cannot position internal node {linkage_id}: "
                "child coordinates "
                "have not been resolved."
            )
        x_coords[linkage_id] = (x_coords[left] + x_coords[right]) / 2.0
        y_coords[linkage_id] = float(row[2])

    return x_coords, y_coords


def _linkage_sample_count(
    linkage_id: int,
    z_matrix: NDArray[np.float64],
    sample_count: int,
) -> int:
    """Return the number of leaves represented by one SciPy node id."""
    if linkage_id < sample_count:
        return 1
    return int(z_matrix[linkage_id - sample_count, 3])


# ======================================================================
# Block 3: Interactive HTML Output
# ======================================================================


def write_dendrogram_html(
    dendro_result: DendrogramResult,
    z_matrix: NDArray[np.float64],
    tree_to_linkage: dict[int, int],
    linkage_colors: dict[int, str],
    topology: TreeTopology,
    output_path: Path,
    tree_depth: int,
    source_name: str,
    folder_name_to_color: dict[str, str] | None = None,
) -> None:
    """Render and publish an interactive immutable-tree dendrogram.

    Args:
        dendro_result: Geometry returned by SciPy's ``dendrogram``
            function.
        z_matrix: Valid SciPy linkage matrix.
        tree_to_linkage: Immutable-node-ID to SciPy-node-ID mapping.
        linkage_colors: SciPy-node-ID to branch/marker color mapping.
        topology: Validated immutable topology and node metadata.
        output_path: HTML output path.
        tree_depth: Maximum root-to-leaf depth.
        source_name: Input CSV name shown in the title.
        folder_name_to_color: Optional source-folder legend mapping.

    Raises:
        ValueError: If the immutable/SciPy node mapping is inconsistent.
    """
    sample_count = len(topology.leaves)
    node_count = topology.node_count
    if node_count != 2 * sample_count - 1:
        raise ValueError("Topology node count must equal 2N-1.")
    if set(tree_to_linkage) != set(range(node_count)):
        raise ValueError(
            (
                f'Cluster-node mapping must contain immutable IDs '
                f'0..{node_count - 1}.'
            )
        )
    if set(tree_to_linkage.values()) != set(range(node_count)):
        raise ValueError((
            'Cluster-node mapping must cover every SciPy node '
            'exactly once.'
        ))

    fig = go.Figure()

    def lookup_linkage_color(linkage_id: int) -> str:
        """Return the color assigned to a plotted linkage node."""
        return linkage_colors.get(linkage_id, UNMATCHED_COLOR)

    color_groups_x: dict[str, list[float | None]] = {}
    color_groups_y: dict[str, list[float | None]] = {}
    for xs, ys, link_color in zip(
        dendro_result["icoord"],
        dendro_result["dcoord"],
        dendro_result["color_list"],
        strict=True,
    ):
        color_groups_x.setdefault(link_color, []).extend([*xs, None])
        color_groups_y.setdefault(link_color, []).extend([*ys, None])

    for color, x_values in color_groups_x.items():
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=color_groups_y[color],
                mode="lines",
                line={"color": color, "width": 1.15},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    x_coords, y_coords = _linkage_node_coordinates(
        z_matrix, dendro_result, sample_count
    )
    tree_ids = list(range(node_count))
    linkage_ids = [tree_to_linkage[tree_id] for tree_id in tree_ids]
    node_xs = [float(x_coords[linkage_id]) for linkage_id in linkage_ids]
    node_ys = [float(y_coords[linkage_id]) for linkage_id in linkage_ids]
    sample_counts = [
        _linkage_sample_count(linkage_id, z_matrix, sample_count)
            for linkage_id in linkage_ids
    ]
    parents = [
        "—" if tree_id == topology.root
        else str(topology.parent_by_node[tree_id])
        for tree_id in tree_ids
    ]
    created = [
        "Root (state 0)" if tree_id == topology.root
        else f"Created by split {topology.creation_split[tree_id]}"
        for tree_id in tree_ids
    ]
    node_colors = [
        lookup_linkage_color(linkage_id) for linkage_id in linkage_ids
    ]

    overview_label_limit = min(18, node_count)
    largest_clusters = sorted(
        tree_ids,
        key=lambda tree_id: (sample_counts[tree_id], -tree_id),
        reverse=True,
    )[:overview_label_limit]
    overview_labels = set(largest_clusters)
    overview_labels.add(0)
    initial_text = [
        f"C{tree_id}" if tree_id in overview_labels else ""
        for tree_id in tree_ids
    ]
    customdata = [
        [tree_id, sample_counts[tree_id], parents[tree_id],
            created[tree_id]]
        for tree_id in tree_ids
    ]

    cluster_trace_index = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=node_xs,
            y=node_ys,
            mode="markers+text",
            name="Cluster identifiers",
            text=initial_text,
            textposition="top center",
            textfont={"size": 10, "color": "#111827"},
            marker={
                "size": 7,
                "color": node_colors,
                "line": {"color": "#111827", "width": 0.6},
            },
            customdata=customdata,
            hovertemplate=(
                "<b>Cluster %{customdata[0]}</b><br>"
                "Samples in node: %{customdata[1]}<br>"
                "Parent node: %{customdata[2]}<br>"
                "%{customdata[3]}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    if folder_name_to_color:
        for folder_name, folder_color in folder_name_to_color.items():
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker={
                        "color": folder_color,
                        "size": 9,
                        "line": {"color": "#111827", "width": 0.5},
                    },
                    name=folder_name,
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

    title_text = (
        (
            f'Divisive Hierarchical Clustering '
            f'Dendrogram<br><sup>{sample_count:,} samples · '
            f'immutable node IDs 0–{node_count - 1:,} · tree '
            f'depth {tree_depth:,} · source: '
            f'{source_name}</sup>'
        )
    )

    fig.update_layout(
        title={
            "text": title_text,
            "x": 0.5,
            "xanchor": "center",
            # Keep the complete two-line title comfortably inside the
            # viewport. The enlarged top margin below reserves space for
            # both title lines.
            "y": 0.965,
            "yanchor": "top",
            "font": {"size": 20, "color": "#111827"},
            "pad": {"t": 0, "b": 10},
        },
        autosize=True,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        hovermode="closest",
        hoverdistance=18,
        dragmode="pan",
        # Reserve top space for the two-line title and right space for
        # the legend, so neither can overlap the dendrogram itself.
        margin={"l": 55, "r": 190, "t": 125, "b": 45},
        xaxis={
            "title": None,
            "showticklabels": False,
            "showgrid": False,
            "zeroline": False,
            "showline": True,
            "mirror": True,
            "linecolor": "#374151",
            "linewidth": 1,
            "fixedrange": False,
        },
        yaxis={
            "title": None,
            "showticklabels": False,
            "showgrid": False,
            "zeroline": False,
            "showline": True,
            "mirror": True,
            "linecolor": "#374151",
            "linewidth": 1,
            "fixedrange": False,
        },
        legend={
            "bgcolor": "rgba(255,255,255,0.98)",
            "bordercolor": "#9CA3AF",
            "borderwidth": 1,
            "font": {"color": "#111827"},
            # Place the legend outside the plotting rectangle in the
            # reserved right margin so it can never cover dendrogram
            # branches.
            "x": 1.02,
            "xanchor": "left",
            "y": 1.0,
            "yanchor": "top",
        },
        font={"color": "#111827"},
        uirevision="divisive-dendrogram",
    )

    plot_config = {
        "responsive": True,
        "displaylogo": False,
        "displayModeBar": True,
        "scrollZoom": True,
        "doubleClick": "reset+autosize",
        "modeBarButtonsToRemove": [
            "select2d",
            "lasso2d",
            "toggleSpikelines",
            "hoverClosestCartesian",
            "hoverCompareCartesian",
        ],
        "toImageButtonOptions": {
            "format": "png",
            "filename": "divisive_dendrogram",
            "scale": 2,
        },
    }

    post_script = (
        '(function () {\n'
        "    const gd = document.getElementById('{plot_id"
        "}');\n"
        '    if (!gd) return;\n'
        '\n'
        '    const CLUSTER_TRACE = __TRACE_INDEX__;\n'
        '    const ADAPTIVE_LABEL_LIMIT = 120;\n'
        '    const OVERVIEW_LABEL_LIMIT = 18;\n'
        '    const LENS_SIZE = 230;\n'
        '    const LENS_SCALE = 2.4;\n'
        '\n'
        '    const host = gd.parentElement;\n'
        "    if (host) host.style.position = 'relative';\n"
        "    document.body.style.margin = '0';\n"
        "    document.body.style.background = '#ffffff';\n"
        '    document.documentElement.style.background = '
        "'#ffffff';\n"
        '\n'
        '    const buttonStyle = [\n'
        "        'font:12px Arial,sans-serif', 'padding:6"
        "px 9px',\n"
        "        'border:1px solid #6B7280', 'border-radi"
        "us:5px',\n"
        "        'background:rgba(255,255,255,0.96)', 'co"
        "lor:#111827',\n"
        "        'cursor:pointer', 'box-shadow:0 1px 3px "
        "rgba(0,0,0,0.15)'\n"
        "    ].join(';');\n"
        '\n'
        "    const controls = document.createElement('div"
        "');\n"
        '    controls.style.cssText = [\n'
        "        'position:absolute','left:10px','top:8px"
        "','z-index:1005',\n"
        "        'display:flex','gap:6px','align-items:ce"
        "nter'\n"
        "    ].join(';');\n"
        '\n'
        "    const labelButton = document.createElement('"
        "button');\n"
        "    labelButton.type = 'button';\n"
        "    labelButton.textContent = 'Labels: adaptive'"
        ';\n'
        '    labelButton.style.cssText = buttonStyle;\n'
        "    labelButton.title = 'Cycle cluster labels: a"
        "daptive → all → hidden';\n"
        '    controls.appendChild(labelButton);\n'
        '\n'
        '    const magnifierButton = document.createEleme'
        "nt('button');\n"
        "    magnifierButton.type = 'button';\n"
        "    magnifierButton.textContent = 'Magnifier: of"
        "f';\n"
        '    magnifierButton.style.cssText = buttonStyle;'
        '\n'
        "    magnifierButton.title = 'Toggle a 2.4× magni"
        "fier lens';\n"
        '    controls.appendChild(magnifierButton);\n'
        '\n'
        "    const fitButton = document.createElement('bu"
        "tton');\n"
        "    fitButton.type = 'button';\n"
        "    fitButton.textContent = 'Fit';\n"
        '    fitButton.style.cssText = buttonStyle;\n'
        "    fitButton.title = 'Fit the complete dendrogr"
        "am to the browser window';\n"
        '    controls.appendChild(fitButton);\n'
        '    if (host) host.appendChild(controls);\n'
        '\n'
        "    const help = document.createElement('div');\n"
        "    help.textContent = 'Wheel: zoom · left/right"
        ' drag: pan · hover: cluster details · double-cli'
        "ck: reset';\n"
        '    help.style.cssText = [\n'
        "        'position:absolute','left:10px','bottom:"
        "8px','z-index:1004',\n"
        "        'font:11px Arial,sans-serif','color:#374"
        "151',\n"
        "        'background:rgba(255,255,255,0.90)','pad"
        "ding:4px 7px',\n"
        "        'border:1px solid #D1D5DB','border-radiu"
        "s:4px','pointer-events:none'\n"
        "    ].join(';');\n"
        '    if (host) host.appendChild(help);\n'
        '\n'
        "    let labelMode = 'adaptive';\n"
        '    const trace = gd.data[CLUSTER_TRACE];\n'
        '    const clusterLabels = trace.customdata.map(r'
        "ow => 'C' + row[0]);\n"
        '\n'
        '    function currentAxisRange(axisName) {\n'
        '        const axis = gd._fullLayout && gd._fullL'
        'ayout[axisName];\n'
        '        return axis && axis.range ? axis.range.s'
        'lice() : null;\n'
        '    }\n'
        '\n'
        '    function updateClusterLabels() {\n'
        '        if (!gd.data[CLUSTER_TRACE]) return;\n'
        '        let labels = new Array(clusterLabels.len'
        "gth).fill('');\n"
        "        if (labelMode === 'all') {\n"
        '            labels = clusterLabels.slice();\n'
        "        } else if (labelMode === 'adaptive') {\n"
        "            const xr = currentAxisRange('xaxis')"
        ';\n'
        "            const yr = currentAxisRange('yaxis')"
        ';\n'
        '            const visible = [];\n'
        '            const xmin = xr ? Math.min(xr[0], xr'
        '[1]) : -Infinity;\n'
        '            const xmax = xr ? Math.max(xr[0], xr'
        '[1]) : Infinity;\n'
        '            const ymin = yr ? Math.min(yr[0], yr'
        '[1]) : -Infinity;\n'
        '            const ymax = yr ? Math.max(yr[0], yr'
        '[1]) : Infinity;\n'
        '            for (let i = 0; i < trace.x.length; '
        'i++) {\n'
        '                if (trace.x[i] >= xmin && trace.'
        'x[i] <= xmax &&\n'
        '                    trace.y[i] >= ymin && trace.'
        'y[i] <= ymax) {\n'
        '                    visible.push(i);\n'
        '                }\n'
        '            }\n'
        '            if (visible.length <= ADAPTIVE_LABEL'
        '_LIMIT) {\n'
        '                for (const i of visible) labels['
        'i] = clusterLabels[i];\n'
        '            } else {\n'
        '                visible.sort((a, b) => trace.cus'
        'tomdata[b][1] - trace.customdata[a][1]);\n'
        '                for (const i of visible.slice(0,'
        ' OVERVIEW_LABEL_LIMIT)) {\n'
        '                    labels[i] = clusterLabels[i]'
        ';\n'
        '                }\n'
        '            }\n'
        '        }\n'
        '        Plotly.restyle(gd, {text: [labels]}, [CL'
        'USTER_TRACE]);\n'
        '    }\n'
        '\n'
        "    labelButton.addEventListener('click', functi"
        'on () {\n'
        "        if (labelMode === 'adaptive') labelMode "
        "= 'all';\n"
        "        else if (labelMode === 'all') labelMode "
        "= 'none';\n"
        "        else labelMode = 'adaptive';\n"
        "        labelButton.textContent = 'Labels: ' + l"
        'abelMode;\n'
        '        updateClusterLabels();\n'
        '    });\n'
        '\n'
        "    fitButton.addEventListener('click', function"
        ' () {\n'
        "        Plotly.relayout(gd, {'xaxis.autorange': "
        "true, 'yaxis.autorange': true})\n"
        '            .then(function () {\n'
        '                Plotly.Plots.resize(gd);\n'
        '                updateClusterLabels();\n'
        '            });\n'
        '    });\n'
        '\n'
        '    let rightPan = null;\n'
        '    let panFrame = null;\n'
        "    gd.addEventListener('contextmenu', function "
        '(event) { event.preventDefault(); });\n'
        "    gd.addEventListener('mousedown', function (e"
        'vent) {\n'
        '        if (event.button !== 2 || !gd._fullLayou'
        't) return;\n'
        '        event.preventDefault();\n'
        "        const xr = currentAxisRange('xaxis');\n"
        "        const yr = currentAxisRange('yaxis');\n"
        '        if (!xr || !yr) return;\n'
        '        rightPan = {x:event.clientX, y:event.cli'
        'entY, xr:xr, yr:yr};\n'
        "        document.body.style.cursor = 'grabbing';"
        '\n'
        '    });\n'
        "    window.addEventListener('mousemove', functio"
        'n (event) {\n'
        '        if (!rightPan || !gd._fullLayout || panF'
        'rame !== null) return;\n'
        '        panFrame = window.requestAnimationFrame('
        'function () {\n'
        '            panFrame = null;\n'
        '            const size = gd._fullLayout._size;\n'
        '            if (!size || size.w <= 0 || size.h <'
        '= 0) return;\n'
        '            const dx = (event.clientX - rightPan'
        '.x) *\n'
        '                (rightPan.xr[1] - rightPan.xr[0]'
        ') / size.w;\n'
        '            const dy = (event.clientY - rightPan'
        '.y) *\n'
        '                (rightPan.yr[1] - rightPan.yr[0]'
        ') / size.h;\n'
        '            Plotly.relayout(gd, {\n'
        "                'xaxis.range': [rightPan.xr[0] -"
        ' dx, rightPan.xr[1] - dx],\n'
        "                'yaxis.range': [rightPan.yr[0] +"
        ' dy, rightPan.yr[1] + dy]\n'
        '            });\n'
        '        });\n'
        '    });\n'
        "    window.addEventListener('mouseup', function "
        '(event) {\n'
        '        if (event.button === 2 && rightPan) {\n'
        '            rightPan = null;\n'
        "            document.body.style.cursor = '';\n"
        '        }\n'
        '    });\n'
        '\n'
        "    const lens = document.createElement('div');\n"
        '    lens.style.cssText = [\n'
        "        'position:absolute','display:none','widt"
        "h:' + LENS_SIZE + 'px',\n"
        "        'height:' + LENS_SIZE + 'px','border:2px"
        " solid #111827',\n"
        "        'border-radius:50%','overflow:hidden','p"
        "ointer-events:none',\n"
        "        'background:#ffffff','z-index:1006',\n"
        "        'box-shadow:0 3px 14px rgba(0,0,0,0.28)'"
        '\n'
        "    ].join(';');\n"
        '    if (host) host.appendChild(lens);\n'
        '\n'
        '    let magnifierOn = false;\n'
        '    let lensClone = null;\n'
        '    let lastPointer = null;\n'
        '\n'
        '    function refreshLensClone() {\n'
        '        if (!magnifierOn) return;\n'
        "        const source = gd.querySelector('.svg-co"
        "ntainer');\n"
        '        if (!source) return;\n'
        '        const rect = gd.getBoundingClientRect();'
        '\n'
        "        lens.innerHTML = '';\n"
        '        lensClone = source.cloneNode(true);\n'
        "        lensClone.querySelectorAll('.modebar-con"
        "tainer,.hoverlayer,.draglayer').forEach(\n"
        '            node => node.remove()\n'
        '        );\n'
        "        lensClone.style.position = 'absolute';\n"
        "        lensClone.style.width = rect.width + 'px"
        "';\n"
        "        lensClone.style.height = rect.height + '"
        "px';\n"
        "        lensClone.style.pointerEvents = 'none';\n"
        "        lensClone.style.transformOrigin = '0 0';"
        '\n'
        '        lens.appendChild(lensClone);\n'
        '        if (lastPointer) positionLens(lastPointe'
        'r);\n'
        '    }\n'
        '\n'
        '    function positionLens(event) {\n'
        '        if (!magnifierOn || !lensClone) return;\n'
        '        lastPointer = event;\n'
        '        const rect = gd.getBoundingClientRect();'
        '\n'
        '        const x = event.clientX - rect.left;\n'
        '        const y = event.clientY - rect.top;\n'
        '        if (x < 0 || y < 0 || x > rect.width || '
        'y > rect.height) {\n'
        "            lens.style.display = 'none';\n"
        '            return;\n'
        '        }\n'
        "        lens.style.display = 'block';\n"
        '        const maxLeft = Math.max(6, rect.width -'
        ' LENS_SIZE - 6);\n'
        '        const maxTop = Math.max(6, rect.height -'
        ' LENS_SIZE - 6);\n'
        '        lens.style.left = Math.min(Math.max(6, x'
        " + 22), maxLeft) + 'px';\n"
        '        lens.style.top = Math.min(Math.max(6, y '
        "+ 22), maxTop) + 'px';\n"
        '        const half = LENS_SIZE / 2;\n'
        '        const tx = half - x * LENS_SCALE;\n'
        '        const ty = half - y * LENS_SCALE;\n'
        '        lensClone.style.transform =\n'
        "            'translate(' + tx + 'px,' + ty + 'px"
        ") scale(' + LENS_SCALE + ')';\n"
        '    }\n'
        '\n'
        "    magnifierButton.addEventListener('click', fu"
        'nction () {\n'
        '        magnifierOn = !magnifierOn;\n'
        "        magnifierButton.textContent = 'Magnifier"
        ": ' + (magnifierOn ? 'on' : 'off');\n"
        '        if (magnifierOn) refreshLensClone();\n'
        "        else lens.style.display = 'none';\n"
        '    });\n'
        "    gd.addEventListener('mousemove', positionLen"
        's);\n'
        "    gd.addEventListener('mouseleave', function ("
        ') {\n'
        "        if (magnifierOn) lens.style.display = 'n"
        "one';\n"
        '    });\n'
        '\n'
        "    gd.on('plotly_relayout', function () {\n"
        "        if (labelMode === 'adaptive') updateClus"
        'terLabels();\n'
        '        if (magnifierOn) window.requestAnimation'
        'Frame(refreshLensClone);\n'
        '    });\n'
        '\n'
        "    window.addEventListener('resize', function ("
        ') {\n'
        '        Plotly.Plots.resize(gd);\n'
        '        if (magnifierOn) window.requestAnimation'
        'Frame(refreshLensClone);\n'
        '    });\n'
        '\n'
        '    updateClusterLabels();\n'
        '})();'
    ).replace("__TRACE_INDEX__", str(cluster_trace_index))

    html = fig.to_html(
        config=plot_config,
        post_script=post_script,
        full_html=True,
        include_plotlyjs=True,
        default_width="100%",
        default_height="100vh",
    )
    write_text_atomically(output_path, html)
    print(f"Dendrogram saved to '{output_path}'")

# ======================================================================
# Block 4: Command-Line Execution
# ======================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        Parsed command-line namespace.

    Raises:
        SystemExit: If command-line usage is invalid or help is
            requested.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tree-states",
        type=Path,
        default=Path("clustering_results") / TREE_STATES_FILE,
        help="Path to immutable total_data.csv (default: %(default)s).",
    )
    parser.add_argument(
        "--topology",
        type=Path,
        default=None,
        help=(
            "Path to tree_splits.txt. By default, use tree_splits.txt beside "
            "the --tree-states file."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        nargs="*",
        default=[
            Path(CANNOT_LINKS_FILE),
            Path(MUST_LINKS_FILE),
            Path(NO_LINKS_FILE),
        ],
        help="Config file(s) listing input folders, one path per line "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DENDROGRAM_FILE),
        help="Path to write the output HTML to (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional integer seed for reproducible colors.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the command-line workflow.

    Args:
        argv: Command-line arguments, or ``None`` to use process
            arguments.

    Expected errors are reported once at the command boundary.
    """
    args = parse_args(argv)
    rng = random.Random(args.seed)

    if not args.tree_states.is_file():
        print(
            f"Error: Data file missing at {args.tree_states}",
            file=sys.stderr,
        )
        sys.exit(1)

    topology_path = (
        args.topology or args.tree_states.with_name(TREE_TOPOLOGY_FILE)
    )
    if not topology_path.is_file():
        print(
            f"Error: Topology file missing at {topology_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        topology_input = (
            args.topology or args.tree_states.with_name(TREE_TOPOLOGY_FILE)
        )
        verify_artifacts([args.tree_states, topology_input])
        raw_data = read_tree_states(args.tree_states)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(
            f"Error: Could not read '{args.tree_states}': {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    if raw_data.shape[1] < 3 or len(raw_data) < 2:
        print(
            (
                f"Error: '{args.tree_states}' must have at least 2 rows "
                f"and one filename column plus immutable "
                f"hierarchy states (got shape {raw_data.shape})."
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    filenames = raw_data["0"]
    compressed_matrix = raw_data.drop(columns=["0"])

    try:
        topology = load_tree_topology(topology_path, len(raw_data))
        history = (
            _decode_tree_states(compressed_matrix, topology)
        )
    except (OSError, ValueError) as exc:
        print(f"Error: invalid immutable hierarchy: {exc}", file=sys.stderr)
        sys.exit(1)

    leaf_labels = history.iloc[:, -1]
    folders = load_folder_categories(args.config)
    try:
        leaf_colors, folder_name_to_color = map_clusters_to_colors(
            filenames, leaf_labels, folders, rng
        )
        z_matrix, linkage_colors, tree_to_linkage = build_linkage_matrix(
            topology, leaf_colors, rng
        )
    except ValueError as exc:
        print(f"Error building immutable linkage: {exc}", file=sys.stderr)
        sys.exit(1)

    def lookup_linkage_branch_color(linkage_id: int) -> str:
        """Return the branch color for a SciPy linkage node."""
        return linkage_colors.get(linkage_id, UNMATCHED_COLOR)

    tree_depth = _linkage_tree_depth(z_matrix, len(history))
    print(
        (
            f'Building dendrogram geometry for {len(history)} '
            f'samples and {topology.node_count} immutable '
            f'nodes (tree depth {tree_depth})...'
        ), file=sys.stderr)
    try:
        dendro_result: DendrogramResult = dendrogram(
            z_matrix, no_plot=True, link_color_func=lookup_linkage_branch_color
        )
    except (ValueError, RecursionError) as exc:
        print(f"Error: SciPy could not build a dendrogram: {exc}",
            file=sys.stderr)
        sys.exit(1)

    try:
        write_dendrogram_html(
            dendro_result=dendro_result,
            z_matrix=z_matrix,
            tree_to_linkage=tree_to_linkage,
            linkage_colors=linkage_colors,
            topology=topology,
            output_path=args.output,
            tree_depth=tree_depth,
            source_name=args.tree_states.name,
            folder_name_to_color=folder_name_to_color,
        )
        dependencies = [args.tree_states, topology_path]
        dependencies.extend(path for path in args.config if path.is_file())
        write_provenance(
            provenance_path(args.output), stage="dendrogram", outputs=[
                args.output],
            inputs=dependencies,
            parents={provenance_path(path) for path in dependencies
                if provenance_path(path).is_file()},
            producer=Path(__file__),
        )
    except (OSError, ValueError) as exc:
        print((
            f'Error: [dendrogram] cannot publish '
            f'{args.output}: {exc}'
        ), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
