"""
Reproduces the manuscript's circular dendrogram (Ward linkage, Euclidean
distance on standardized SPI-12 series, 31 candidate stations, colored by
region) from the real station-level data, using dendrofan instead of the
original ad hoc Matplotlib script.

This script expects the two station-level CSVs produced by the data
cleaning / imputation / regionalization pipeline (Materials and Methods)
in a local `data/` folder next to this script:
  - data/spi12_31stations.csv      (31 stations x SPI-12 monthly values)
  - data/station_region_FINAL.csv  (station -> region assignment, 29 stations)
Edit SPI_CSV / REGION_CSV below if your files live elsewhere.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from dendrofan import circular_dendrogram, legend_handles

DATA_DIR = Path(__file__).parent / "data"
SPI_CSV = DATA_DIR / "spi12_31stations.csv"
REGION_CSV = DATA_DIR / "station_region_FINAL.csv"
OUTPUT_PNG = "dendrogram_circular_real.png"

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]

REGION_COLORS = {
    "Semi-arid": "#E69F00",
    "Highlands": "#0072B2",
    "Mountains": "#009E73",
    "Canyons": "#D55E00",
}
EXCLUDED_COLOR = "#999999"

# --- 1. Load SPI-12 series for the 31 candidate stations --------------------
spi = pd.read_csv(SPI_CSV, index_col=0, parse_dates=True)
spi_complete = spi.dropna(how="any")
stations = spi_complete.columns.tolist()
print(f"Using {len(spi_complete)} months with complete data across all "
      f"{len(stations)} stations "
      f"({spi_complete.index.min().date()} to {spi_complete.index.max().date()})")

# --- 2. Region assignment (29 of the 31 stations; 2 excluded on documented
#        geographic grounds, per the manuscript's Materials and Methods) ----
assign = pd.read_csv(REGION_CSV).set_index("station")["region"].to_dict()
excluded = [s for s in stations if s not in assign]
print(f"Excluded stations (no region assignment): {excluded}")


def label_color(station: str) -> str:
    if station in assign:
        return REGION_COLORS[assign[station]]
    return EXCLUDED_COLOR


def label_text(station: str) -> str:
    return station + (" (excl.)" if station not in assign else "")


# --- 3. Cluster (Ward, Euclidean, on standardized series) and draw ---------
X = spi_complete.T.values  # one row per station, one column per month

result = circular_dendrogram(
    X,
    labels=stations,
    method="ward",
    metric="euclidean",
    figsize=(16, 16),
    inner_radius=0.10,
    label_colors=label_color,
    label_formatter=label_text,
    label_fontsize=14,
    label_fontstyle="italic",
    label_fontweight="bold",
    label_offset=0.05,
)

handles, legend_labels = legend_handles({**REGION_COLORS, "Excluded": EXCLUDED_COLOR})
result.ax.legend(handles, legend_labels, loc="lower right", fontsize=11, frameon=False)

result.fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
print(f"Saved {OUTPUT_PNG}")
