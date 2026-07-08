"""
Reproduces the original ad hoc circular dendrogram (Ward linkage on
standardized SPI-12 series, colored by region, ``ape``-style fan layout)
using dendrofan instead of the one-off Matplotlib script it replaces.

This uses synthetic data standing in for the real 31-station SPI-12
matrix, so the example is self-contained and runnable without the
station-level CSV files from the manuscript's data pipeline. Swap
`SPI_CSV` / `REGION_CSV` for real files to reproduce the actual figure.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dendrofan import circular_dendrogram, legend_handles

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]

REGION_COLORS = {
    "Semi-arid": "#E69F00",
    "Highlands": "#0072B2",
    "Mountains": "#009E73",
    "Canyons": "#D55E00",
}
EXCLUDED_COLOR = "#999999"
EXCLUDED = {"Chalchihuites", "Concepcion del Oro"}

# --- Stand-in data: 31 synthetic "stations", 4 clustered regions -----------
rng = np.random.default_rng(42)
n_per_region = {"Semi-arid": 8, "Highlands": 7, "Mountains": 8, "Canyons": 6}
station_region = {}
rows = []
station_names = []
for region, n in n_per_region.items():
    centre = rng.normal(size=60)
    for i in range(n):
        name = f"{region[:3]}_{i:02d}"
        station_names.append(name)
        station_region[name] = region
        rows.append(centre + rng.normal(scale=0.6, size=60))
for name in EXCLUDED:
    station_names.append(name)
    rows.append(rng.normal(size=60))

spi = pd.DataFrame(np.array(rows).T, columns=station_names)

# --- Color and label styling, matching the original script -----------------
def label_color(station: str) -> str:
    if station in EXCLUDED:
        return EXCLUDED_COLOR
    return REGION_COLORS[station_region[station]]


def label_text(station: str) -> str:
    return station + (" (excl.)" if station in EXCLUDED else "")


result = circular_dendrogram(
    spi.T.values,
    labels=station_names,
    method="ward",
    metric="euclidean",
    figsize=(14, 14),
    inner_radius=0.12,
    label_colors=label_color,
    label_formatter=label_text,
    label_fontsize=13,
    label_fontstyle="italic",
    label_fontweight="bold",
    label_offset=0.05,
)

handles, legend_labels = legend_handles({**REGION_COLORS, "Excluded": EXCLUDED_COLOR})
result.ax.legend(handles, legend_labels, loc="lower right", fontsize=10, frameon=False)

result.fig.savefig("dendrogram_circular.png", dpi=300, bbox_inches="tight")
print("Saved dendrogram_circular.png")
