import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.cnn_classifier import RadarCNN

def split_train_val(X, y, train_frac=0.8, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    n_train = int(len(y) * train_frac)
    train_idx, val_idx = idx[:n_train], idx[n_train:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

def accuracy(probs, y_true, thr=0.5):
    preds = (probs >= thr).float()
    return (preds == y_true).float().mean().item()

def main():
    X = np.load("data/processed/X.npy")  # (N,1,200,200)
    y = np.load("data/processed/y.npy")  # (N,)
    X_train, y_train, X_val, y_val = split_train_val(X, y)

    # tensores
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float()
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).float()

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=64, shuffle=False)

    device = torch.device("cpu")
    model = RadarCNN().to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs("outputs/models", exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, 11):
        # treino
        model.train()
        train_losses, train_accs = [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            probs = model(xb).squeeze(1)
            loss = criterion(probs, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_accs.append(accuracy(probs.detach(), yb.detach()))

        # validação
        model.eval()
        val_losses, val_accs = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                probs = model(xb).squeeze(1)
                loss = criterion(probs, yb)
                val_losses.append(loss.item())
                val_accs.append(accuracy(probs, yb))

        train_loss = float(np.mean(train_losses))
        train_acc = float(np.mean(train_accs))
        val_loss = float(np.mean(val_losses))
        val_acc = float(np.mean(val_accs))

        print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.3f} | val loss {val_loss:.4f} acc {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "outputs/models/radar_cnn_best.pt")

    print("✅ Melhor val_acc:", round(best_val_acc, 3))
    print("✅ Modelo salvo em outputs/models/radar_cnn_best.pt")

if __name__ == "__main__":
    main()
