import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

LABEL_NAMES = {
    0: "NO_SPLIT",
    1: "QT",
    2: "BTH",
    3: "BTV",
    4: "TTH",
    5: "TTV",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="datasets/16x16/QP_32/raw_balanced.npz", type=Path)
    parser.add_argument("--samples-per-class", default=6, type=int)
    parser.add_argument("--out-img", default="tmp_out/preview_samples.png", type=Path)
    args = parser.parse_args()

    if not args.npz.exists():
        raise FileNotFoundError(f"找不到檔案: {args.npz}")

    data = np.load(args.npz)
    patches = data["patches"]
    labels = data["labels"]

    print(f"=== 檢視檔案: {args.npz} ===")
    print(f"Keys       : {list(data.keys())}")
    print(f"Patches    : shape={patches.shape}, dtype={patches.dtype}, min={patches.min()}, max={patches.max()}")
    print(f"Labels     : shape={labels.shape}, dtype={labels.dtype}")

    unique_labels = sorted(np.unique(labels))
    for c in unique_labels:
        cnt = int((labels == c).sum())
        print(f"  Class {c} ({LABEL_NAMES.get(int(c), 'UNK'):8s}): {cnt:6d} 筆 ({cnt/len(labels)*100:5.2f}%)")

    # 繪製各類別實際像素畫面對照網格
    rows = len(unique_labels)
    cols = args.samples_per_class
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.2), squeeze=False)
    rng = np.random.default_rng(42)

    for r_idx, c in enumerate(unique_labels):
        c_patches = patches[labels == c]
        chosen = rng.choice(len(c_patches), size=min(cols, len(c_patches)), replace=False)
        for col_idx in range(cols):
            ax = axes[r_idx, col_idx]
            ax.axis("off")
            if col_idx < len(chosen):
                ax.imshow(c_patches[chosen[col_idx]], cmap="gray", vmin=0, vmax=255)
                if col_idx == 0:
                    ax.set_title(f"{c}: {LABEL_NAMES.get(int(c), str(c))}", fontsize=10, loc="left", fontweight="bold")

    plt.suptitle(f"{args.npz} (Sample Patches)", fontsize=12)
    plt.tight_layout()
    args.out_img.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_img, dpi=150)
    print(f"[OK] 預覽圖已儲存至: {args.out_img}")
    plt.show()

if __name__ == "__main__":
    main()