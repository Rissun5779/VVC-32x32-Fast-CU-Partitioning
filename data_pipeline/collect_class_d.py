import os
import subprocess
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

CLASS_D_SEQUENCES = [
    ("BasketballPass", "../Jem_Test_Sequence/ClassD/BasketballPass_416x240_50.yuv", 416, 240, 50, 100),
    ("BlowingBubbles", "../Jem_Test_Sequence/ClassD/BlowingBubbles_416x240_50.yuv", 416, 240, 50, 100),
    ("BQSquare",       "../Jem_Test_Sequence/ClassD/BQSquare_416x240_60.yuv",       416, 240, 60, 100),
    ("RaceHorses",     "../Jem_Test_Sequence/ClassD/RaceHorses_416x240_30.yuv",     416, 240, 30, 100),
]
QPS = [22, 27, 32, 37]
SIZES = [32, 16, 8]

def run_encode_task(seq_idx: int, seq_name: str, yuv_path: str, w: int, h: int, fps: int, frames: int, qp: int):
    tmp_dir = Path("tmp_out/batch_d")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{seq_name}_qp{qp}"
    bins = {sz: tmp_dir / f"{prefix}_{sz}x{sz}.bin" for sz in SIZES}
    for b in bins.values():
        b.unlink(missing_ok=True)

    env = os.environ.copy()
    env["VVC_DUMP_32X32_PATH"] = str(bins[32])
    env["VVC_DUMP_16X16_PATH"] = str(bins[16])
    env["VVC_DUMP_8X8_PATH"]   = str(bins[8])

    cmd = [
        "./bin/EncoderAppStatic",
        "-c", "./external/VTM24.0/cfg/encoder_intra_vtm.cfg",
        "-i", yuv_path,
        "-wdt", str(w), "-hgt", str(h), "-fr", str(fps), "-f", str(frames),
        "--InputBitDepth=8", "-q", str(qp),
        "-b", str(tmp_dir / f"{prefix}.bin"),
        "-o", "/dev/null"
    ]
    subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    return seq_idx, seq_name, qp, bins

def deduplicate_bin(bin_path: Path, size: int):
    rec_bytes = size * size + 8
    raw = np.fromfile(bin_path, dtype=np.uint8)
    if raw.size == 0 or raw.size % rec_bytes != 0:
        return np.empty((0, size, size), dtype=np.uint8), np.empty((0,), dtype=np.uint8)
    records = raw.reshape(-1, rec_bytes)
    unique_map = {}
    for row in records:
        meta = np.frombuffer(row[size*size+2 : size*size+8].tobytes(), dtype=np.uint16)
        unique_map[(int(meta[0]), int(meta[1]), int(meta[2]))] = row
    clean = np.stack([unique_map[k] for k in sorted(unique_map.keys())], axis=0)
    patches = clean[:, :size*size].reshape(-1, size, size)
    labels = clean[:, size*size+1]
    return patches, labels

def main():
    buffer = {sz: {qp: [] for qp in QPS} for sz in SIZES}
    tasks = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for seq_idx, (seq_name, yuv_path, w, h, fps, frames) in enumerate(CLASS_D_SEQUENCES):
            for qp in QPS:
                tasks.append(pool.submit(run_encode_task, seq_idx, seq_name, yuv_path, w, h, fps, frames, qp))

        for fut in as_completed(tasks):
            seq_idx, seq_name, qp, bins = fut.result()
            print(f"[DONE] {seq_name:15s} (seq_id={seq_idx}) QP={qp}")
            for sz in SIZES:
                p, y = deduplicate_bin(bins[sz], sz)
                s = np.full(len(y), seq_idx, dtype=np.uint8)
                buffer[sz][qp].append((p, y, s))
                bins[sz].unlink(missing_ok=True)

    for sz in SIZES:
        for qp in QPS:
            out_dir = Path(f"datasets/{sz}x{sz}/QP_{qp}")
            out_dir.mkdir(parents=True, exist_ok=True)
            all_p = np.concatenate([x[0] for x in buffer[sz][qp]], axis=0)
            all_y = np.concatenate([x[1] for x in buffer[sz][qp]], axis=0)
            all_s = np.concatenate([x[2] for x in buffer[sz][qp]], axis=0)
            counts = {int(c): int((all_y == c).sum()) for c in np.unique(all_y)}
            print(f"  -> {sz:2d}x{sz:<2d} QP_{qp}: Total={len(all_y):6d} | Class Dist={counts}")
            np.savez_compressed(out_dir / "raw_balanced.npz", patches=all_p, labels=all_y, seq_ids=all_s)

if __name__ == "__main__":
    main()