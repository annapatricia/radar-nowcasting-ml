import os
import numpy as np
from preprocess import normalize_dbz, label_strong_cell

def moving_blob_sequence(T=6, h=200, w=200, seed=0, will_be_strong=True):

    """
    Gera uma sequência de T frames (dBZ) com um 'núcleo' que se move e muda intensidade.
    Vamos usar T=6 para ter 5 entradas + 1 alvo (t+1).
    """
    rng = np.random.default_rng(seed)

    # fundo fraco
    base = lambda: np.clip(5 + 10 * rng.standard_normal((h, w)), 0, 35)

    # posição inicial e movimento
    cx = rng.integers(w//4, 3*w//4)
    cy = rng.integers(h//4, 3*h//4)
    vx = rng.integers(-3, 4)  # -3..3 px por frame
    vy = rng.integers(-3, 4)

    r = rng.integers(12, 26)

    # intensidade cresce ou decresce ao longo do tempo
    start_boost = rng.uniform(5, 18)

    if will_be_strong:
        # futuro tende a ficar forte (>=45)
        end_boost = rng.uniform(22, 40)
    else:
        # futuro tende a NÃO ficar forte
        end_boost = rng.uniform(5, 18)

    boosts = np.linspace(start_boost, end_boost, T)

    frames = []
    yy, xx = np.ogrid[:h, :w]

    for t in range(T):
        dbz = base()

        # move
        cxt = int(np.clip(cx + t*vx, r, w-r-1))
        cyt = int(np.clip(cy + t*vy, r, h-r-1))

        mask = (xx - cxt)**2 + (yy - cyt)**2 < r**2
        dbz[mask] = np.clip(dbz[mask] + boosts[t], 0, 70)

        frames.append(dbz)

    return np.stack(frames, axis=0)  # (T,H,W)

def build_sequence_dataset(
    out_dir="data/processed_seq",
    n_sequences=1000,
    input_len=5,
    h=200,
    w=200,
    threshold_dbz=45.0,
    seed=123
):
    os.makedirs(out_dir, exist_ok=True)

    # Vamos gerar T = input_len + 1 frames
    T = input_len + 1

    X = np.zeros((n_sequences, input_len, h, w), dtype=np.float32)
    y = np.zeros((n_sequences,), dtype=np.int64)

    for i in range(n_sequences):
        will_be_strong = (i % 2 == 0)  # alterna 0/1
        seq = moving_blob_sequence(T=T, h=h, w=w, seed=seed + i, will_be_strong=will_be_strong)


        # Entrada: primeiros 5 frames (normalizados)
        Xin = np.stack([normalize_dbz(seq[t]) for t in range(input_len)], axis=0)  # (5,H,W)
        X[i] = Xin.astype(np.float32)

        # Saída: label do frame futuro (t+1)
        y[i] = label_strong_cell(seq[input_len], threshold_dbz=threshold_dbz)

    np.save(os.path.join(out_dir, "X_seq.npy"), X)
    np.save(os.path.join(out_dir, "y_future.npy"), y)

    print("✅ Dataset de sequência salvo em:", out_dir)
    print("X_seq:", X.shape, "y_future:", y.shape, "positivos:", int(y.sum()), "negativos:", int((y==0).sum()))

if __name__ == "__main__":
    build_sequence_dataset()
