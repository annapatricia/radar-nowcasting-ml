import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from models.cnn_future_classifier import FutureRadarCNN

def main():
    # carregar dataset
    X = np.load("data/processed_seq/X_seq.npy")    # (N,5,200,200)
    y = np.load("data/processed_seq/y_future.npy") # (N,)

    # mesmo split usado no treino (seed=42 e 80/20)
    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    n_train = int(n * 0.8)
    val_idx = idx[n_train:]

    Xv = torch.from_numpy(X[val_idx]).float()
    yv = y[val_idx]

    # carregar modelo
    device = torch.device("cpu")
    model = FutureRadarCNN().to(device)
    model.load_state_dict(torch.load("outputs/models/future_cnn_best.pt", map_location=device))
    model.eval()

    with torch.no_grad():
        probs = model(Xv.to(device)).squeeze(1).cpu().numpy()

    y_pred = (probs >= 0.5).astype(int)

    print("Confusion matrix (val):")
    print(confusion_matrix(yv, y_pred, labels=[0,1]))
    print()
    print(classification_report(yv, y_pred, labels=[0,1], digits=3))

if __name__ == "__main__":
    main()
