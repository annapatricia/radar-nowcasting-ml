import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def radar_colormap():
    bounds = [0, 5, 10, 20, 30, 40, 50, 60, 70]
    colors = [
        "#e0f7fa",
        "#b2ebf2",
        "#81c784",
        "#fff176",
        "#ffb74d",
        "#ef5350",
        "#ab47bc",
        "#6d4c41",
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds

def plot_radar_dbz(dbz, title, outpath=None):
    cmap, norm, bounds = radar_colormap()

    plt.figure(figsize=(7, 6))
    im = plt.imshow(dbz, cmap=cmap, norm=norm, origin="lower")
    cbar = plt.colorbar(im, ticks=bounds)
    cbar.set_label("dBZ")

    plt.title(title)
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=150)
    else:
        plt.show()

    plt.close()
