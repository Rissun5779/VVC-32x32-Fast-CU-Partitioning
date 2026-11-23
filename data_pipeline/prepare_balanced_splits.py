import numpy as np
from pathlib import Path

QPS = [22, 27, 32, 37]
VALID_CLASSES = {
    32: [0, 1],          # 0: NO_SPLIT, 1: QT
    16: [0, 1, 2, 3, 4, 5], # 0: NO_SPLIT, 1: QT, 2: BTH, 3: BTV, 4: TTH, 5: TTV
    8:  [0, 1, 2, 3],    # 0: NO_SPLIT, 1: QT (4x4), 2: BTH, 3: BTV
}

def augment_minority(patches: np.ndarray, target_count: int, rng: np.random.Generator) -> np.ndarray:
    n = len(patches)
    if n == 0:
        return np.empty((0, patches.shape[1], patches.shape[2]), dtype=np.uint8)
    if n >= target_count:
        idx = rng.choice(n, size=target_count, replace=False)
        return patches[idx]

    aug_pool = [
        patches,
        patches[:, :, ::-1],
        patches[:, ::-1, :],
        patches[:, ::-1, ::-1],
    ]
    expanded = np.concatenate(aug_pool, axis=0)
    if len(expanded) >= target_count:
        idx = rng.choice(len(expanded), size=target_count, replace=False)
        return expanded[idx]
    idx = rng.choice(len(expanded), size=target_count, replace=True)
    return expanded[idx]

def main():
    rng = np.random.default_rng(42)
    # 每個序列、每個類別的目標樣本數（4 支序列合計為 4 倍）
    per_seq_target = {32: 500, 16: 900, 8: 2500}

    for sz in [32, 16, 8]:
        valid = VALID_CLASSES[sz]
        label_map = {orig_c: new_c for new_c, orig_c in enumerate(valid)}
        for qp in QPS:
            in_path = Path(f"datasets/{sz}x{sz}/QP_{qp}/raw_balanced.npz")
            data = np.load(in_path)
            patches, labels, seq_ids = data["patches"], data["labels"], data["seq_ids"]

            bal_p, bal_y, bal_s = [], [], []
            for seq_id in sorted(np.unique(seq_ids)):
                seq_mask = (seq_ids == seq_id)
                sp, sy = patches[seq_mask], labels[seq_mask]
                for orig_c in valid:
                    c_patches = sp[sy == orig_c]
                    sampled = augment_minority(c_patches, per_seq_target[sz], rng)
                    bal_p.append(sampled)
                    bal_y.append(np.full(len(sampled), label_map[orig_c], dtype=np.uint8))
                    bal_s.append(np.full(len(sampled), seq_id, dtype=np.uint8))

            X = np.concatenate(bal_p, axis=0)
            Y = np.concatenate(bal_y, axis=0)
            S = np.concatenate(bal_s, axis=0)
            perm = rng.permutation(len(Y))
            X, Y, S = X[perm], Y[perm], S[perm]

            out_path = Path(f"datasets/{sz}x{sz}/QP_{qp}/train_ready.npz")
            np.savez_compressed(out_path, patches=X, labels=Y, seq_ids=S)
            dist = {int(c): int((Y == c).sum()) for c in range(len(valid))}
            print(f"[READY] {sz:2d}x{sz:<2d} QP_{qp} -> {out_path} | Shape={X.shape} | Balanced Dist={dist}")

if __name__ == "__main__":
    main()