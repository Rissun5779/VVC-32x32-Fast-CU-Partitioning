import argparse
import numpy as np
from pathlib import Path

RECORD_BYTES = 1032
SPLIT_NAMES = ["NO_SPLIT", "QT", "BTH", "BTV", "TTH", "TTV"]

def parse_and_deduplicate(bin_path: Path):
    raw = np.fromfile(bin_path, dtype=np.uint8)
    if raw.size == 0 or raw.size % RECORD_BYTES != 0:
        raise ValueError(f"Invalid binary size {raw.size}")
    records = raw.reshape(-1, RECORD_BYTES)
    unique_map = {}
    for row in records:
        meta = np.frombuffer(row[1026:1032].tobytes(), dtype=np.uint16)
        unique_map[(int(meta[0]), int(meta[1]), int(meta[2]))] = row
    sorted_keys = sorted(unique_map.keys())
    return np.stack([unique_map[k] for k in sorted_keys], axis=0), sorted_keys

def get_sub_blocks(x: int, y: int, label: int):
    if label == 1:   # QT (4 個 16x16)
        return [(x, y, 16, 16), (x+16, y, 16, 16), (x, y+16, 16, 16), (x+16, y+16, 16, 16)]
    elif label == 2: # BTH (上下 32x16)
        return [(x, y, 32, 16), (x, y+16, 32, 16)]
    elif label == 3: # BTV (左右 16x32)
        return [(x, y, 16, 32), (x+16, y, 16, 32)]
    elif label == 4: # TTH (上中下 32x8, 32x16, 32x8)
        return [(x, y, 32, 8), (x, y+8, 32, 16), (x, y+24, 32, 8)]
    elif label == 5: # TTV (左中右 8x32, 16x32, 8x32)
        return [(x, y, 8, 32), (x+8, y, 16, 32), (x+24, y, 8, 32)]
    return [(x, y, 32, 32)] # NO_SPLIT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", required=True, type=Path)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    clean_records, keys = parse_and_deduplicate(args.bin)
    labels = clean_records[:, 1025]
    for idx, name in enumerate(SPLIT_NAMES):
        cnt = int((labels == idx).sum())
        print(f"  Label {idx} ({name:8s}): {cnt:4d} ({cnt/len(keys)*100:5.1f}%)")

    lines = [
        "# VTMBMS Block Statistics",
        f"# Sequence size: [{args.width}x{args.height}]",
        "# Block Statistic Type: SplitMode; Integer; [1, 6]",
    ]
    for row, (poc, x, y) in zip(clean_records, keys):
        label = int(row[1025])
        for bx, by, bw, bh in get_sub_blocks(x, y, label):
            lines.append(f"BlockStat: POC {poc} @({bx:4d},{by:4d}) [{bw:2d}x{bh:2d}] SplitMode={label + 1}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Exported {args.out} ({len(keys)} 32x32 CUs)")

if __name__ == "__main__":
    main()