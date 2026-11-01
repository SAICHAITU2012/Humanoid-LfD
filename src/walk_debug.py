"""Render walk frames + trace for visual debugging. Run from src/:
    ../.venv/bin/python walk_debug.py
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

import tune_walk as tw
from gait import GaitParams, HIP0, KNEE0, walk_targets

ROOT = Path(__file__).resolve().parents[1]
m = mujoco.MjModel.from_xml_path(str(tw.MODEL))
d = mujoco.MjData(m)
renderer = mujoco.Renderer(m, height=360, width=640)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0, 0, 0.7]
cam.distance = 2.6
cam.azimuth = 150
cam.elevation = -10
pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

freq = float(sys.argv[1]) if len(sys.argv) > 1 else 0.42
reach = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
hipS = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
p = GaitParams(freq=freq, reach=reach, knee_amp=0.40, shift=0.09, hip_sign=hipS)

SETTLE = 0.8
dt = m.opt.timestep
mujoco.mj_resetDataKeyframe(m, d, 0)
for a in range(m.nu):
    jid = m.actuator_trnid[a, 0]
    d.ctrl[a] = d.qpos[m.jnt_qposadr[jid]]
crouch = {"left_hip_pitch_joint": HIP0, "right_hip_pitch_joint": HIP0,
          "left_knee_joint": KNEE0, "right_knee_joint": KNEE0,
          "left_ankle_pitch_joint": p.ankle_level, "right_ankle_pitch_joint": p.ankle_level,
          "left_hip_roll_joint": p.spread, "right_hip_roll_joint": -p.spread,
          "left_ankle_roll_joint": 0.0, "right_ankle_roll_joint": 0.0, "waist_roll_joint": 0.0}
act = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in tw.LEG_JOINTS}
jn = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in tw.LEG_JOINTS}
prev_r = prev_p = 0.0
rr = rp = 0.0
outdir = ROOT / "outputs/walk_frames"
outdir.mkdir(exist_ok=True)
print("    t     x      y      z    pitch  feet  Lknee  Lank  Rknee  Rank  LfootZ RfootZ")
for i in range(int(5.0 / dt)):
    t = i * dt
    l_c = r_c = True
    if t < SETTLE:
        blend = t / SETTLE
        targets = dict(crouch)
    else:
        blend = 1.0
        sc = min(1.0, (t - SETTLE) / 1.2)
        targets = walk_targets(t - SETTLE, p, sc)
        roll = float(d.xquat[pelvis][1])
        pitch = float(d.xquat[pelvis][2])
        rr = 0.7 * rr + 0.3 * (roll - prev_r) / dt
        rp = 0.7 * rp + 0.3 * (pitch - prev_p) / dt
        prev_r, prev_p = roll, pitch
        u_roll = max(-0.08, min(0.08, 1.2 * roll + 0.25 * rr))
        u_pit = max(-0.12, min(0.12, 1.5 * pitch + 0.3 * rp))
        l_c, r_c = tw.foot_contact(m, d, "left"), tw.foot_contact(m, d, "right")
        if l_c and r_c:
            for s in ("left", "right"):
                targets[f"{s}_hip_pitch_joint"] += u_pit
        elif l_c:
            targets["left_hip_roll_joint"] -= u_roll
            targets["left_hip_pitch_joint"] -= u_pit
        elif r_c:
            targets["right_hip_roll_joint"] -= u_roll
            targets["right_hip_pitch_joint"] -= u_pit
    for n, a in act.items():
        base = crouch.get(n, 0.0)
        d.ctrl[a] = (1 - blend) * base + blend * targets.get(n, base)
    mujoco.mj_step(m, d)
    if i % 100 == 0:  # every 0.2s
        lk = d.qpos[m.jnt_qposadr[jn["left_knee_joint"]]]
        la = d.qpos[m.jnt_qposadr[jn["left_ankle_pitch_joint"]]]
        rk = d.qpos[m.jnt_qposadr[jn["right_knee_joint"]]]
        ra = d.qpos[m.jnt_qposadr[jn["right_ankle_pitch_joint"]]]
        lf = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")][2]
        rf = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")][2]
        ft = ("L" if l_c else "") + ("R" if r_c else "") or "-"
        xp = d.xpos[pelvis]
        print(f"{t:5.2f} {xp[0]:+.3f} {xp[1]:+.3f} {xp[2]:.3f} {float(d.xquat[pelvis][2]):+.3f}  {ft}  "
              f"{lk:.3f} {la:+.3f} {rk:.3f} {ra:+.3f}  {lf:.3f}  {rf:.3f}")
    if i % 150 == 0:  # every 0.3s -> frame
        mujoco.mj_forward(m, d)
        renderer.update_scene(d, cam)
        from PIL import Image
        Image.fromarray(renderer.render()).save(outdir / f"frame_{i*dt:04.1f}s.png")
    if not np.isfinite(d.qpos).all() or d.xpos[pelvis][2] < 0.45:
        print(f"FELL at {t:.2f}")
        break
print("done")
