from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn

from build_cup_scene import ensure_scene


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/public/g1_cup_mit"
CKPT = ROOT / "models/public_baseline.pt"
ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
TIPS = ["right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link"]
SOURCE = "lewislf/G1_Dex3_pick_white_cup_right8_1squeeze (MIT)"


def load_all() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = sorted(glob.glob(str(DATA_DIR / "data/chunk-000/*.parquet")))
    cols = ["action", "observation.state", "episode_index"]
    tables = [pq.read_table(f, columns=cols) for f in files]
    x = np.asarray(np.concatenate([t.column("observation.state").to_pylist() for t in tables]), np.float32)
    y = np.asarray(np.concatenate([t.column("action").to_pylist() for t in tables]), np.float32)
    e = np.asarray(np.concatenate([t.column("episode_index").to_pylist() for t in tables]), int)
    return x, y, e


def squeeze_to_fingers(train_states: np.ndarray, train_actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear per-joint map from the squeeze scalar to observed finger angles (fit on train episodes)."""
    sq = train_actions[:-1, 7]
    fingers = train_states[1:, 7:14]
    design = np.vstack([np.ones_like(sq), sq]).T
    coef, *_ = np.linalg.lstsq(design, fingers, rcond=None)
    return coef[0], coef[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual outputs of the Experiment-A BC policy (public data).")
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="Held-out episode indices to show.")
    args = parser.parse_args()

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(14, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 8))

        def forward(self, x):
            return self.net(x)

    model = Policy()
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    xm, xs = ckpt["x_mean"], ckpt["x_std"]
    ym, ys = ckpt["y_mean"], ckpt["y_std"]
    split_ep = ckpt["episode_split"]

    x, y, episodes = load_all()
    order = np.sort(np.unique(episodes))
    val_eps = order[order >= split_ep]
    chosen = args.episodes or [int(v) for v in val_eps[:3]]
    print(f"Held-out episodes: {chosen} (split at episode {split_ep})")

    # ---- Figure 1: predicted vs true actions on held-out episodes ----
    fig, axes = plt.subplots(3, 3, figsize=(13, 8), sharex=False)
    rows = [("shoulder_pitch", 0), ("elbow", 3), ("grasp_squeeze", 7)]
    for col, ep in enumerate(chosen):
        m = episodes == ep
        states, actions = x[m], y[m]
        with torch.no_grad():
            pred = model(torch.from_numpy((states - xm) / xs).to(device)).cpu().numpy() * ys + ym
        t = np.arange(len(states)) / 30.0
        for r, (name, dim) in enumerate(rows):
            ax = axes[r, col]
            ax.plot(t, actions[:, dim], "k-", lw=1.6, label="true (teleop)")
            ax.plot(t, pred[:, dim], "tab:red", lw=1.2, ls="--", label="BC policy")
            rmse = float(np.sqrt(((pred[:, dim] - actions[:, dim]) ** 2).mean()))
            ax.set_title(f"ep {ep} - {name} - RMSE {rmse:.3f}", fontsize=9)
            if col == 0:
                ax.set_ylabel(name)
            if r == 2:
                ax.set_xlabel("time (s)")
            if r == 0 and col == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Experiment A: BC policy (public G1 Dex3 data) vs real teleop — held-out episodes")
    fig.tight_layout()
    fig_path = ROOT / "outputs/fig_public_baseline_predictions.png"
    fig.savefig(fig_path, dpi=130)
    print(f"Figure: {fig_path}")

    # ---- Rollout: the policy drives the skeleton closed-loop (arm state = its own last command) ----
    ep = chosen[0]
    m = episodes == ep
    true_states, true_actions = x[m], y[m]
    finger_b, finger_w = squeeze_to_fingers(x[episodes < split_ep], y[episodes < split_ep])

    scene = ensure_scene()
    model_mj = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model_mj)
    stand = mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model_mj, data, stand)
    arm_qadr = model_mj.jnt_qposadr[[mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]]
    hand_qadr = model_mj.jnt_qposadr[[mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_JOINT, n) for n in
                                      ["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
                                       "right_hand_index_0_joint", "right_hand_index_1_joint",
                                       "right_hand_middle_0_joint", "right_hand_middle_1_joint"]]]
    tips = [mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_BODY, n) for n in TIPS]

    data.qpos[arm_qadr] = true_states[0, :7]
    data.qpos[hand_qadr] = finger_b + finger_w * true_actions[0, 7]
    mujoco.mj_forward(model_mj, data)

    import cv2
    renderer = mujoco.Renderer(model_mj, height=480, width=480)
    video_path = ROOT / "outputs/public_baseline_rollout.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (480, 480))
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.25, -0.05, 0.85]
    cam.distance = 2.0
    cam.azimuth = 140
    cam.elevation = -12

    arm_err = []
    joint_limits = model_mj.jnt_range[[mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]]
    for frame in range(len(true_states)):
        state_in = np.concatenate([data.qpos[arm_qadr], data.qpos[hand_qadr]]).astype(np.float32)
        with torch.no_grad():
            action = model(torch.from_numpy((state_in - xm) / xs).to(device)).cpu().numpy() * ys + ym
        action[:7] = np.clip(action[:7], joint_limits[:, 0], joint_limits[:, 1])
        action[7] = float(np.clip(action[7], 0.0, 1.0))
        data.qpos[arm_qadr] = action[:7]
        data.qpos[hand_qadr] = finger_b + finger_w * action[7]
        data.qvel[:] = 0
        mujoco.mj_forward(model_mj, data)
        arm_err.append(float(np.sqrt(((data.qpos[arm_qadr] - true_states[frame, :7]) ** 2).mean())))
        renderer.update_scene(data, camera=cam)
        img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        cv2.putText(img, f"BC policy rollout - held-out ep {ep} - frame {frame}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"arm RMSE so far {float(np.mean(arm_err)):.3f} rad", (12, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 2)
        writer.write(img)
    writer.release()
    renderer.close()
    print(f"Closed-loop rollout: {len(true_states)} frames, mean arm RMSE vs demo {float(np.mean(arm_err)):.4f} rad")
    print(f"Video: {video_path}")


if __name__ == "__main__":
    main()
