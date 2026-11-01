"""Headless gait sweep for the quasi-static walker in src/gait.py.

Ranks parameter sets by survival + forward distance. Run:
    .venv/bin/python src/tune_walk.py
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path

import mujoco
import numpy as np

from gait import GaitParams, HIP0, KNEE0, walk_targets

MODEL = Path(__file__).resolve().parents[1] / "vendor/mujoco_menagerie/unitree_g1/scene_with_hands.xml"
SETTLE_S = 0.8    # ramp stand -> crouch before walking
RAMP_S = 1.2      # ramp step scale in after settle
WALK_S = 10.0
FALL_Z = 0.45

LEG_JOINTS = ("left_hip_pitch_joint", "left_hip_roll_joint", "left_knee_joint", "left_ankle_pitch_joint",
              "right_hip_pitch_joint", "right_hip_roll_joint", "right_knee_joint", "right_ankle_pitch_joint",
              "left_ankle_roll_joint", "right_ankle_roll_joint", "waist_roll_joint")


def foot_contact(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> bool:
    foot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
    for c in range(data.ncon):
        g1, g2 = data.contact[c].geom1, data.contact[c].geom2
        if (model.geom_bodyid[g1] == foot or model.geom_bodyid[g2] == foot) and data.contact[c].dist < 0.005:
            return True
    return False


def run_once(model, data, p: GaitParams) -> dict:
    act = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in LEG_JOINTS}
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    mujoco.mj_resetDataKeyframe(model, data, 0)
    for a in range(model.nu):  # hold everything at the stand pose
        jid = model.actuator_trnid[a, 0]
        data.ctrl[a] = data.qpos[model.jnt_qposadr[jid]]

    crouch = {f"{s}_hip_pitch_joint": HIP0 for s in ("left", "right")}
    crouch.update({"left_knee_joint": KNEE0, "right_knee_joint": KNEE0,
                   "left_ankle_pitch_joint": p.ankle_level, "right_ankle_pitch_joint": p.ankle_level,
                   "left_hip_roll_joint": p.spread, "right_hip_roll_joint": -p.spread,
                   "left_ankle_roll_joint": 0.0, "right_ankle_roll_joint": 0.0, "waist_roll_joint": 0.0})

    prev_r = prev_p = 0.0
    rr = rp = 0.0
    dt = model.opt.timestep
    x0 = float(data.xpos[pelvis][0])
    fell_at = None
    for i in range(int((SETTLE_S + RAMP_S + WALK_S) / dt)):
        t = i * dt
        if t < SETTLE_S:
            blend = t / SETTLE_S
            targets = dict(crouch)
        else:
            blend = 1.0
            sc = min(1.0, (t - SETTLE_S) / RAMP_S)
            targets = walk_targets(t - SETTLE_S, p, sc)
            roll = float(data.xquat[pelvis][1])
            pitch = float(data.xquat[pelvis][2])
            rr = 0.7 * rr + 0.3 * (roll - prev_r) / dt
            rp = 0.7 * rp + 0.3 * (pitch - prev_p) / dt
            prev_r, prev_p = roll, pitch
            u_roll = max(-0.08, min(0.08, 1.2 * roll + 0.25 * rr))
            u_pit = max(-0.12, min(0.12, 1.5 * pitch + 0.3 * rp))
            l_c, r_c = foot_contact(model, data, "left"), foot_contact(model, data, "right")
            if l_c and r_c:      # double support: hold the trunk, both hips
                for s in ("left", "right"):
                    targets[f"{s}_hip_pitch_joint"] += u_pit
            elif l_c:            # single support: rotate about the stance foot
                targets["left_hip_roll_joint"] -= u_roll
                targets["left_hip_pitch_joint"] -= u_pit
            elif r_c:
                targets["right_hip_roll_joint"] -= u_roll
                targets["right_hip_pitch_joint"] -= u_pit
        for n, a in act.items():
            base = crouch.get(n, 0.0)
            data.ctrl[a] = (1.0 - blend) * base + blend * targets.get(n, base)
        mujoco.mj_step(model, data)
        z = float(data.xpos[pelvis][2])
        if not np.isfinite(data.qpos).all() or z < FALL_Z:
            fell_at = t
            break
    return {"fell_at": fell_at, "distance": float(data.xpos[pelvis][0]) - x0,
            "lateral": float(data.xpos[pelvis][1])}


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    print(f"sweep over freq/reach/knee/shift/sign: {32} runs x {WALK_S:.0f}s sim")

    grid = list(itertools.product(
        (0.42, 0.55),      # freq (full cycle = 2 steps)
        (0.25, 0.35),      # reach
        (0.4, 0.55),       # knee_amp
        (0.06, 0.09),      # shift
        (1.0, -1.0),       # hip_sign (walk direction)
    ))
    results = []
    for freq, reach, knee_amp, shift, hip_sign in grid:
        p = GaitParams(freq=freq, reach=reach, knee_amp=knee_amp, shift=shift, hip_sign=hip_sign)
        r = run_once(model, data, p)
        results.append({"params": p, **r})
    results.sort(key=lambda r: (r["fell_at"] is not None, -r["distance"]))
    print(f"\n{'fell':>5} {'dist(m)':>8} {'lat(m)':>7}  freq reach knee shift hipS")
    for r in results[:12]:
        p = r["params"]
        fell = f"{r['fell_at']:.1f}s" if r["fell_at"] is not None else "no"
        print(f"{fell:>5} {r['distance']:+8.3f} {r['lateral']:+7.3f}  {p.freq:.2f}  {p.reach:.2f}  "
              f"{p.knee_amp:.2f}  {p.shift:.2f}  {p.hip_sign:+.0f}")


if __name__ == "__main__":
    main()
