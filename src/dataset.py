import os
import numpy as np
from preprocess import normalize_dbz, label_strong_cell

def make_synthetic_dbz(h=200, w=200, seed=None, strong_prob=0.5):
    rng = np.random.default_rng(seed)
    dbz = np.clip(5 + 10 * rng.standard_normal((h, w)), 0, 35)

    if rng.random() < strong_prob:
        cx = rng.integers(low=w//4, high=3*w//4)
        cy = rng.integers(low=h//4, high=3*h//4)
        r = rng.integers(low=12, high=30)

        yy, xx = np.ogrid[:h, :w]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        dbz[mask] = np.clip(dbz[mask] + 25, 0, 70)

    return dbz

def build_dataset():
    os.makedirs("data/processed", exist_ok=True)

    n_samples = 1000
    X = np.zeros((n_samples, 1, 200, 200), dtype=np.float32)
    y = np.zeros((n_samples,), dtype=np.int64)

    for i in range(n_samples):
        dbz = make_synthetic_dbz(seed=i)
        y[i] = label_strong_cell(dbz)
        X[i, 0] = normalize_dbz(dbz)

    np.save("data/processed/X.npy", X)
    np.save("data/processed/y.npy", y)

    print("Dataset criado com sucesso!")

if __name__ == "__main__":
    build_dataset()
