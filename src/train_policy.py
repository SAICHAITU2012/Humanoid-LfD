from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class HandMapper(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, out_features))

    def forward(self, x):
        return self.net(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small supervised human-landmark → G1 hand-target mapper.")
    parser.add_argument("landmarks", type=Path)
    parser.add_argument("actions", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--history-out", type=Path, help="Save per-epoch loss history as JSON here.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    torch.manual_seed(0)
    records = {}
    with args.landmarks.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if not item["hands"]:
                continue
            hand = item["hands"][0]
            records[int(item["frame"])] = np.asarray(hand["landmarks"], dtype=np.float32).reshape(-1)
    action_rows = list(csv.DictReader(args.actions.open(encoding="utf-8", newline="")))
    joint_names = [k for k in action_rows[0] if k.startswith("right_hand_")]
    pairs = [(records[int(r["frame"])], [float(r[j]) for j in joint_names]) for r in action_rows if int(r["frame"]) in records]
    if len(pairs) < 20:
        raise SystemExit(f"Need at least 20 aligned frames; found {len(pairs)}. Record a longer, clearer clip.")
    x = np.stack([p[0] for p in pairs]).astype(np.float32)
    y = np.asarray([p[1] for p in pairs], dtype=np.float32)
    # Split in time, not randomly, so adjacent frames do not leak across train and validation.
    split = max(1, min(len(x) - 1, int(0.8 * len(x))))
    xm, xs = x[:split].mean(0), x[:split].std(0).clip(1e-4)
    ym, ys = y[:split].mean(0), y[:split].std(0).clip(1e-4)
    x = (x - xm) / xs
    y = (y - ym) / ys
    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else ("cpu" if args.device == "auto" else args.device)
    model = HandMapper(x.shape[1], y.shape[1]).to(device)
    xt = torch.from_numpy(x[:split]).to(device)
    yt = torch.from_numpy(y[:split]).to(device)
    xv = torch.from_numpy(x[split:]).to(device)
    yv = torch.from_numpy(y[split:]).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    for epoch in range(args.epochs):
        model.train()
        optim.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(xt), yt)
        loss.backward()
        optim.step()
        entry = {"epoch": epoch + 1, "train_mse": float(loss.detach().cpu())}
        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            model.eval()
            with torch.no_grad():
                entry["val_rmse"] = float(torch.sqrt(nn.functional.mse_loss(model(xv), yv)).cpu())
            history.append(entry)
    if args.history_out:
        args.history_out.parent.mkdir(parents=True, exist_ok=True)
        args.history_out.write_text(json.dumps({"device": device, "aligned_frames": len(pairs), "train_frames": split, "val_frames": len(x) - split, "epochs": args.epochs, "history": history}, indent=2), encoding="utf-8")
    model.eval()
    with torch.no_grad():
        val_rmse = float(torch.sqrt(nn.functional.mse_loss(model(xv), yv)).cpu())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "input_mean": xm, "input_std": xs, "target_mean": ym, "target_std": ys, "joint_names": joint_names, "val_rmse": val_rmse, "aligned_frames": len(pairs)}, args.out)
    print(f"Device: {device}; aligned frames: {len(pairs)}; train: {split}; validation: {len(x)-split}")
    print(f"Normalized validation RMSE: {val_rmse:.4f}; weights: {args.out}")
    print("This model learns the retargeter's pseudo-label mapping. Task-level success still requires object/contact evaluation.")


if __name__ == "__main__":
    main()

