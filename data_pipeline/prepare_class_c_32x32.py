import numpy as np
from pathlib import Path

QPS = [22, 27, 32, 37]

def augment_minority(patches: np.ndarray, target_count: int, rng: np.random.Generator) -> np.ndarray:
    n = len(patches)
    if n >= target_count:
        return patches[rng.choice(n, size=target_count, replace=False)]
    expanded = np.concatenate([
        patches, patches[:, :, ::-1], patches[:, ::-1, :], patches[:, ::-1, ::-1]
    ], axis=0)
    replace = len(expanded) < target_count
    return expanded[rng.choice(len(expanded), size=target_count, replace=replace)]

def main():
    rng = np.random.default_rng(42)
    per_seq_target = 800  # 4 支序列 x 800 = 3200 筆/類 -> 6400 筆/QP

    for qp in QPS:
        in_path = Path(f"datasets_classC/32x32/QP_{qp}/raw_balanced.npz")
        data = np.load(in_path)
        patches, labels, seq_ids = data["patches"], data["labels"], data["seq_ids"]

        bal_p, bal_y, bal_s = [], [], []
        for seq_id in [4, 5, 6, 7]:
            mask = (seq_ids == seq_id)
            sp, sy = patches[mask], labels[mask]
            for c in [0, 1]:
                sampled = augment_minority(sp[sy == c], per_seq_target, rng)
                bal_p.append(sampled)
                bal_y.append(np.full(len(sampled), c, dtype=np.uint8))
                bal_s.append(np.full(len(sampled), seq_id, dtype=np.uint8))

        X = np.concatenate(bal_p, axis=0)
        Y = np.concatenate(bal_y, axis=0)
        S = np.concatenate(bal_s, axis=0)
        perm = rng.permutation(len(Y))

        out_path = Path(f"datasets_classC/32x32/QP_{qp}/train_ready.npz")
        np.savez_compressed(out_path, patches=X[perm], labels=Y[perm], seq_ids=S[perm])
        print(f"[READY] ClassC 32x32 QP_{qp} -> {out_path} | Shape={X.shape} | Dist={{0: {(Y==0).sum()}, 1: {(Y==1).sum()}}}")

if __name__ == "__main__":
    main()