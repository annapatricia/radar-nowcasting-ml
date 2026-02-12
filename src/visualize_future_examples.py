import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from models.cnn_future_classifier import FutureRadarCNN
from preprocess import DBZ_MAX

def denorm(x01):
    return x01 * DBZ_MAX

def pick_first(mask):
    idx = np.where(mask)[0]
    return int(idx[0]) if len(idx) else None

def plot_sequence_grid(seq01, title, outpath):
    """
    seq01: (5, H, W) em [0,1]
    Salva uma figura com 5 frames lado a lado.
    """
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(14, 3))
    for i in range(5):
        axes[i].imshow(denorm(seq01[i]), origin="lower")
        axes[i].set_title(f"t-{4-i}")
        axes[i].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def main():
    # carregar dataset
    X = np.load("data/processed_seq/X_seq.npy")     # (N,5,200,200) em 0..1
    y = np.load("data/processed_seq/y_future.npy")  # (N,)

    # split igual ao treino (seed=42, 80/20)
    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    n_train = int(n * 0.8)
    val_idx = idx[n_train:]

    Xv = X[val_idx]
    yv = y[val_idx]

    # carregar modelo
    device = torch.device("cpu")
    model = FutureRadarCNN().to(device)
    model.load_state_dict(torch.load("outputs/models/future_cnn_best.pt", map_location=device))
    model.eval()

    with torch.no_grad():
        probs = model(torch.from_numpy(Xv).float().to(device)).squeeze(1).cpu().numpy()

    y_pred = (probs >= 0.5).astype(int)

    # máscaras
    tp = (yv == 1) & (y_pred == 1)
    tn = (yv == 0) & (y_pred == 0)
    fp = (yv == 0) & (y_pred == 1)
    fn = (yv == 1) & (y_pred == 0)

    out_dir = "outputs/figures/future_examples"
    os.makedirs(out_dir, exist_ok=True)

    def save_case(mask, name):
        i = pick_first(mask)
        if i is None:
            print(f"⚠️ Sem exemplo para {name}")
            return
        seq01 = Xv[i]  # (5,H,W)
        title = f"{name} | prob={probs[i]:.3f} | real={yv[i]} pred={y_pred[i]}"
        outpath = os.path.join(out_dir, f"{name}.png")
        plot_sequence_grid(seq01, title, outpath)
        print("✅ salvo:", outpath)

    save_case(tp, "TP")
    save_case(tn, "TN")
    save_case(fp, "FP")
    save_case(fn, "FN")

    print("\nConfusion matrix (val):")
    print(confusion_matrix(yv, y_pred, labels=[0, 1]))

if __name__ == "__main__":
    main()
