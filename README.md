# dendrofan

Circular ("fan") dendrograms for Python, in the visual style of R's
[`ape`](https://cran.r-project.org/package=ape) package
(`plot.phylo(type = "fan")`, Paradis & Schliep, 2019) — computed and
rendered entirely with NumPy, SciPy, and Matplotlib. No R dependency,
no reimplementation of clustering.

## Why this exists

`scipy.cluster.hierarchy` has no circular/fan layout. The usual
workaround is a one-off script: call
`scipy.cluster.hierarchy.dendrogram(..., no_plot=True)`, take its
rectangular `icoord`/`dcoord` output, and manually re-project it into
polar coordinates with Matplotlib. That works for one figure, but it
tends to:

- assume a full 360° circle with no gap for labels at the seam,
- divide by `dcoord.max()` without checking for a degenerate,
  all-zero-height tree,
- hardcode SciPy's 10-units-per-leaf spacing convention,
- have no story for trees with 1 or 2 leaves,
- mix clustering, geometry, and plotting into a single script, so
  none of it is reusable for the next dataset.

dendrofan factors the actually-reusable idea — the rectangular-to-polar
coordinate transform — into a small, tested, documented library, and
builds a real plotting API on top of it.

## Install

```bash
pip install -e .          # from a source checkout
pip install -e ".[dev]"   # + pytest, for running the test suite
```

## Quick start

```python
import numpy as np
from dendrofan import circular_dendrogram

rng = np.random.default_rng(0)
data = rng.normal(size=(20, 6))
labels = [f"sample_{i:02d}" for i in range(20)]

result = circular_dendrogram(
    data,
    labels=labels,
    method="ward",       # forwarded to scipy.cluster.hierarchy.linkage
    span=350,             # leave a small angular gap at the seam
    inner_radius=0.15,    # small central gap, ape's fan style
)
result.fig.savefig("tree.png", dpi=200, bbox_inches="tight")
```

`circular_dendrogram` accepts the same three kinds of input as SciPy's
own `dendrogram`: raw observations (`data=...`), a precomputed linkage
matrix (`Z=...`), or a precomputed condensed distance vector
(`condensed_distances=...`).

See [`examples/quickstart.py`](examples/quickstart.py) and
[`examples/reproduce_station_dendrogram.py`](examples/reproduce_station_dendrogram.py)
(a fully worked, colored-by-group example) for more.

## What it handles that the ad hoc version didn't

| Case | Ad hoc script | dendrofan |
|---|---|---|
| Angular gap at the seam (for label room) | not supported (full circle only) | `span=350` (or any value `<= 360`) |
| All-merge-heights-equal-zero tree | `ZeroDivisionError` / silent NaNs | falls back to placing all nodes at the leaf radius |
| 1 or 2 leaves | untested, breaks silently | validated; raises `DegenerateTreeError` for < 2, works for 2 |
| Mismatched label count | silent misalignment | raises `LabelMismatchError` |
| Invalid linkage matrix / distance vector | undefined behavior | raises `InvalidLinkageError` before plotting |
| Root at centre vs. rim, radius scale | hardcoded linear, root-at-centre | `inner_radius`/`outer_radius`/`invert_radius`, plus optional `"sqrt"` area-preserving radius scale |
| Per-clade coloring | manual, one-off | `color_threshold`/`link_color_func` forwarded to SciPy, or a `label_colors` dict/callable |
| Scale reference / clade highlighting | not present | `dendrofan.annotations.add_scale_ring`, `highlight_sector` |
| Reuse across datasets | copy-paste the script | one function call |

## API overview

- `dendrofan.circular_dendrogram(...)` — the main entry point; draws
  the tree and returns a `CircularDendrogramResult` (figure, axes,
  layout, and the `PolarTransform` used, for further annotation).
- `dendrofan.geometry.PolarTransform` — the reusable rectangular-to-polar
  coordinate map, if you want to bend your own geometry onto the same
  annulus (e.g. a custom decoration).
- `dendrofan.clustering.build_layout(...)` — validated wrapper around
  `scipy.cluster.hierarchy.linkage` / `dendrogram`, decoupled from
  plotting.
- `dendrofan.styling.resolve_leaf_colors`, `legend_handles` — color and
  legend helpers.
- `dendrofan.annotations.add_scale_ring`, `highlight_sector` — optional
  decorations (a distance-reference ring; shaded clade sectors).

Every public function and class has a full docstring; `help(...)` in a
Python session is the fastest way to see the complete parameter
reference.

## Scope

dendrofan draws circular dendrograms from hierarchical clustering
(SciPy linkage matrices) — it does not parse Newick/phylogenetic tree
files or handle unequal-tip-depth phylogenies the way `ape` itself
does. If you need that, `ape` (R) or `ete3`/`Bio.Phylo` (Python) are a
better fit; dendrofan specifically fills the "I have a SciPy
dendrogram and want it circular" gap.

## Testing

```bash
pytest
```

The test suite specifically exercises the edge cases listed above
(degenerate trees, mismatched labels, invalid linkage matrices, partial
spans, single/two-leaf trees).

## Citation

If you use dendrofan in a manuscript, please also cite the underlying
methods it wraps:

- Virtanen, P. et al. (2020). SciPy 1.0: fundamental algorithms for
  scientific computing in Python. *Nature Methods*, 17, 261-272.
  *(hierarchical clustering / linkage, which dendrofan builds on)*
- Paradis, E. & Schliep, K. (2019). ape 5.0: an environment for modern
  phylogenetics and evolutionary analyses in R. *Bioinformatics*,
  35(3), 526-528. *(for the fan-plot visual convention dendrofan follows)*

## License

MIT — see [LICENSE](LICENSE).
