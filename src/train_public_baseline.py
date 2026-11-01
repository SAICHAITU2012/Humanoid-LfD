from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/public/g1_cup_mit"
SOURCE = "lewislf/G1_Dex3_pick_white_cup_right8_1squeeze (MIT)"
ARM_JOINTS = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]


def load() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = sorted(glob.glob(str(DATA_DIR / "data/chunk-000/*.parquet")))
    if not files:
        raise SystemExit(f"No parquet files under {DATA_DIR}")
    cols = ["action", "observation.state", "episode_index"]
    tables = [pq.read_table(f, columns=cols) for f in files]
    x = np.concatenate([t.column("observation.state").to_pylist() for t in tables])
    y = np.concatenate([t.column("action").to_pylist() for t in tables])
    e = np.concatenate([t.column("episode_index").to_pylist() for t in tables])
    return np.asarray(x, np.float32), np.asarray(y, np.float32), np.asarray(e, int)


class Policy(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, out_features))

    def forward(self, x):
        return self.net(x)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Train the Experiment-A BC baseline on public G1 Dex3 teleop episodes ({SOURCE})."
    )
    parser.add_argument("--out", type=Path, default=ROOT / "models/public_baseline.pt")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()

    x, y, episodes = load()
    order = np.sort(np.unique(episodes))
    split_ep = order[int(0.8 * len(order))]  # time-ordered split by episode: no leakage
    train_mask = episodes < split_ep
    val_mask = ~train_mask
    x_tr, y_tr, x_va, y_va = x[train_mask], y[train_mask], x[val_mask], y[val_mask]
    print(f"Episodes: {len(order)} ({int(train_mask.sum())} train frames / {int(val_mask.sum())} val frames, "
          f"split at episode {split_ep})")

    xm, xs = x_tr.mean(0), x_tr.std(0).clip(1e-4)
    ym, ys = y_tr.mean(0), y_tr.std(0).clip(1e-4)
    x_tr_n = (x_tr - xm) / xs
    y_tr_n = (y_tr - ym) / ys
    x_va_n = (x_va - xm) / xs
    y_va_n = (y_va - ym) / ys

    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else ("cpu" if args.device == "auto" else args.device)
    model = Policy(x.shape[1], y.shape[1]).to(device)
    xt = torch.from_numpy(x_tr_n).to(device)
    yt = torch.from_numpy(y_tr_n).to(device)
    xv = torch.from_numpy(x_va_n).to(device)
    yv = torch.from_numpy(y_va_n).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    batch = 4096
    generator = np.random.default_rng(0)
    for epoch in range(args.epochs):
        model.train()
        perm = generator.permutation(len(xt))
        for start in range(0, len(xt), batch):
            idx = perm[start:start + batch]
            optim.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(xt[idx]), yt[idx])
            loss.backward()
            optim.step()
        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                val = float(torch.sqrt(nn.functional.mse_loss(model(xv), yv)))
            print(f"epoch {epoch + 1}: val normalized RMSE {val:.4f}")

    model.eval()
    with torch.no_grad():
        pred = model(xv).cpu().numpy() * ys + ym
    err = pred - y_va
    print("\nHeld-out action-prediction error (original units, radians / squeeze scalar):")
    names = ARM_JOINTS + ["grasp_squeeze"]
    for i, name in enumerate(names):
        print(f"  {name:15s} RMSE {float(np.sqrt((err[:, i] ** 2).mean())):.4f}  "
              f"(target std {float(y_va[:, i].std()):.3f})")
    print(f"  overall arm RMSE {float(np.sqrt((err[:, :7] ** 2).mean())):.4f} rad")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "x_mean": xm, "x_std": xs, "y_mean": ym, "y_std": ys,
                "episode_split": int(split_ep), "source": SOURCE, "input_names": "observation.state(14)",
                "output_names": names}, args.out)
    print(f"\nWeights: {args.out}")


if __name__ == "__main__":
    main()
