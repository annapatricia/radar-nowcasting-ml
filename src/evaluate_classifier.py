import os
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from models.cnn_classifier import RadarCNN
from viz import plot_radar_dbz
from preprocess import DBZ_MAX

def denormalize(x01):
    return x01 * DBZ_MAX  # volta pra dBZ (aprox)

def main():
    # dados
    X = np.load("data/processed/X.npy")  # (N,1,200,200)
    y = np.load("data/processed/y.npy")  # (N,)

    # mesmo split do treino
    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    n_train = int(n * 0.8)
    val_idx = idx[n_train:]

    Xv = torch.from_numpy(X[val_idx]).float()
    yv = y[val_idx]

    # modelo
    device = torch.device("cpu")
    model = RadarCNN().to(device)
    model.load_state_dict(torch.load("outputs/models/radar_cnn_best.pt", map_location=device))
    model.eval()

    with torch.no_grad():
        probs = model(Xv.to(device)).squeeze(1).cpu().numpy()

    y_pred = (probs >= 0.5).astype(int)

    print("Confusion matrix:")
    print(confusion_matrix(yv, y_pred))
    print()
    print(classification_report(yv, y_pred, digits=3))

    # salvar exemplos
    os.makedirs("outputs/figures/examples", exist_ok=True)

    def save_example(mask, name):
        if not np.any(mask):
            print(f"⚠️ Nenhum exemplo para {name}")
            return
        i = np.where(mask)[0][0]  # primeiro exemplo
        img01 = X[val_idx[i], 0]          # 0..1
        dbz = denormalize(img01)          # volta p/ dBZ
        out = f"outputs/figures/examples/{name}.png"
        plot_radar_dbz(dbz, f"{name} | prob={probs[i]:.3f}", outpath=out)
        print("✅ salvo:", out)

    tp = (yv == 1) & (y_pred == 1)
    tn = (yv == 0) & (y_pred == 0)
    fp = (yv == 0) & (y_pred == 1)
    fn = (yv == 1) & (y_pred == 0)

    save_example(tp, "TP_true_positive")
    save_example(tn, "TN_true_negative")
    save_example(fp, "FP_false_positive")
    save_example(fn, "FN_false_negative")

if __name__ == "__main__":
    main()
