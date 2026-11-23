import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    npz_path = Path("datasets/32x32/QP_32/train_ready.npz")
    data = np.load(npz_path)
    patches, labels = data["patches"], data["labels"]

    rng = np.random.default_rng(42)
    p0 = patches[labels == 0][rng.choice(int((labels == 0).sum()), 50, replace=False)]
    p1 = patches[labels == 1][rng.choice(int((labels == 1).sum()), 50, replace=False)]
    grid_patches = np.concatenate([p0, p1], axis=0)  # (100, 32, 32)

    fig, axes = plt.subplots(10, 10, figsize=(10, 10.5))
    for idx, ax in enumerate(axes.flat):
        ax.imshow(grid_patches[idx], cmap="gray", vmin=0, vmax=255)
        ax.axis("off")

    # 在第 5 列與第 6 列之間畫一條紅色分隔線
    fig.subplots_adjust(hspace=0.08, wspace=0.08, top=0.93, bottom=0.02, left=0.02, right=0.98)
    line = plt.Line2D([0.02, 0.98], [0.475, 0.475], transform=fig.transFigure, color="red", linewidth=2)
    fig.add_artist(line)
    fig.suptitle("Top 5 Rows: Class 0 (NO_SPLIT)  |  Bottom 5 Rows: Class 1 (QT)  [32x32 QP32]", fontsize=11, fontweight="bold")

    out_path = Path("tmp_out/mosaic_32x32_qp32.png")
    plt.savefig(out_path, dpi=150)
    print(f"[OK] 10x10 Mosaic 已儲存至: {out_path}")
    plt.show()

if __name__ == "__main__":
    main()