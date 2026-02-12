import os
import numpy as np
from viz import plot_radar_dbz

def make_synthetic_dbz(h=200, w=200, seed=0):
    rng = np.random.default_rng(seed)
    dbz = np.clip(10 + 20 * rng.standard_normal((h, w)), 0, 65)

    yy, xx = np.ogrid[:h, :w]
    mask = (xx - 120) ** 2 + (yy - 80) ** 2 < 25 ** 2
    dbz[mask] = np.clip(dbz[mask] + 30, 0, 70)

    return dbz

def main():
    os.makedirs("outputs/figures", exist_ok=True)
    dbz = make_synthetic_dbz()
    outpath = "outputs/figures/radar_sintetico.png"
    plot_radar_dbz(dbz, "Radar sintético (dBZ)", outpath)
    print("Imagem salva em:", outpath)

if __name__ == "__main__":
    main()
