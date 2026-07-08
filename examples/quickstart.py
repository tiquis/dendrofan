"""Minimal dendrofan example: cluster random data and draw a fan dendrogram."""
import numpy as np

from dendrofan import circular_dendrogram

rng = np.random.default_rng(0)
data = rng.normal(size=(20, 6))
labels = [f"sample_{i:02d}" for i in range(20)]

result = circular_dendrogram(
    data,
    labels=labels,
    method="ward",
    span=350,           # leave a small angular gap at the seam
    inner_radius=0.15,  # small central gap, ape's fan style
    label_fontsize=9,
)
result.fig.savefig("quickstart_dendrogram.png", dpi=200, bbox_inches="tight")
print("Saved quickstart_dendrogram.png")
