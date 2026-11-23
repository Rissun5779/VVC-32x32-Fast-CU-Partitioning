import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class Cu32Dataset(Dataset):
    def __init__(self, patches: np.ndarray, qps: np.ndarray, labels: np.ndarray):
        self.patches = torch.from_numpy(patches).float().unsqueeze(1) / 255.0  # (N, 1, 32, 32)
        self.qps = torch.from_numpy(qps).float().unsqueeze(1) / 51.0          # (N, 1)
        self.labels = torch.from_numpy(labels).long()                         # (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.patches[idx], self.qps[idx], self.labels[idx]

class FastCu32Net(nn.Module):
    """硬體友善輕量級 32x32 二元分割預測網路 (NO_SPLIT vs QT)"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 16x16 -> 8x8
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 8x8 -> 4x4
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 4x4 -> 1x1 Global Average Pooling
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 + 1, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
        )

    def forward(self, x_patch: torch.Tensor, x_qp: torch.Tensor) -> torch.Tensor:
        feat = self.features(x_patch).flatten(1)       # (B, 64)
        fused = torch.cat([feat, x_qp], dim=1)         # (B, 65)
        return self.classifier(fused)

def load_sequence_split(qps: list[int]):
    tr_p, tr_q, tr_y = [], [], []
    va_p, va_q, va_y = [], [], []
    te_p, te_q, te_y = [], [], []

    for qp in qps:
        data = np.load(f"datasets/32x32/QP_{qp}/train_ready.npz")
        p, y, s = data["patches"], data["labels"], data["seq_ids"]
        q_arr = np.full(len(y), qp, dtype=np.float32)

        # Train: seq 0 (BasketballPass) & seq 1 (BlowingBubbles)
        m_tr = np.isin(s, [0, 1])
        tr_p.append(p[m_tr]); tr_q.append(q_arr[m_tr]); tr_y.append(y[m_tr])

        # Val: seq 2 (BQSquare)
        m_va = (s == 2)
        va_p.append(p[m_va]); va_q.append(q_arr[m_va]); va_y.append(y[m_va])

        # Test: seq 3 (RaceHorses)
        m_te = (s == 3)
        te_p.append(p[m_te]); te_q.append(q_arr[m_te]); te_y.append(y[m_te])

    train_ds = Cu32Dataset(np.concatenate(tr_p), np.concatenate(tr_q), np.concatenate(tr_y))
    val_ds   = Cu32Dataset(np.concatenate(va_p), np.concatenate(va_q), np.concatenate(va_y))
    test_ds  = Cu32Dataset(np.concatenate(te_p), np.concatenate(te_q), np.concatenate(te_y))
    return train_ds, val_ds, test_ds

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    correct, total = 0, 0
    tp = tn = fp = fn = 0
    for patches, qps, labels in loader:
        patches, qps, labels = patches.to(device), qps.to(device), labels.to(device)
        preds = model(patches, qps).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        tp += ((preds == 1) & (labels == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()
    acc = correct / max(total, 1)
    return acc, (tn, fp, fn, tp)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qps", nargs="+", type=int, default=[22, 27, 32, 37])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device} | QPs: {args.qps}")

    train_ds, val_ds, test_ds = load_sequence_split(args.qps)
    print(f"[INFO] Sequence-Split Sizes -> Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = FastCu32Net().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] FastCu32Net Parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    ckpt_path = Path("checkpoints/fast_cu_32x32.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, tr_correct, tr_total = 0.0, 0, 0
        for patches, qps, labels in train_loader:
            patches, qps, labels = patches.to(device), qps.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(patches, qps)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            tr_correct += (logits.argmax(dim=1) == labels).sum().item()
            tr_total += labels.size(0)

        scheduler.step()
        tr_acc = tr_correct / tr_total
        val_acc, _ = evaluate(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{args.epochs:02d} | Loss: {total_loss/tr_total:.4f} | "
                  f"Train Acc: {tr_acc*100:5.2f}% | Val Acc (BQSquare): {val_acc*100:5.2f}%")

    # 載入最佳 Val 權重，在從未見過的 RaceHorses 測試集評估
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    test_acc, (tn, fp, fn, tp) = evaluate(model, test_loader, device)
    print("\n=== Unseen Sequence Test Result (RaceHorses) ===")
    print(f"Best Val Acc : {best_val_acc*100:5.2f}%")
    print(f"Test Acc     : {test_acc*100:5.2f}%")
    print(f"Confusion Matrix -> TN(NO_SPLIT): {tn}, FP: {fp} | FN: {fn}, TP(QT): {tp}")

    # 分別檢視各 QP 在 Test Set 上的準確率
    for qp in args.qps:
        _, _, sub_test = load_sequence_split([qp])
        sub_acc, _ = evaluate(model, DataLoader(sub_test, batch_size=256), device)
        print(f"  -> Test QP_{qp}: {sub_acc*100:5.2f}%")

if __name__ == "__main__":
    main()