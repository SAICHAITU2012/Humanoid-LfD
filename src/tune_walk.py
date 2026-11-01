"""Headless gait sweep with lateral balance feedback (v2).

Ranks parameter sets by survival + forward distance. Run:
    .venv/bin/python src/tune_walk.py
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path

import mujoco
import numpy as np

from gait import GaitParams, HIP0, KNEE0, leg_targets, stand_leg_targets

MODEL = Path(__file__).resolve().parents[1] / "vendor/mujoco_menagerie/unitree_g1/scene_with_hands.xml"
SETTLE_S = 0.8   # ramp stand -> crouch before stepping
WALK_S = 8.0
FALL_Z = 0.45


def solve_level_ankle(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Ankle pitch that keeps the left foot level at the crouch pose."""
    foot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in
           ("left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint")]
    lo, hi = -0.8, 0.5
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qpos[model.jnt_qposadr[ids[0]]] = HIP0
        data.qpos[model.jnt_qposadr[ids[1]]] = KNEE0
        data.qpos[model.jnt_qposadr[ids[2]]] = mid
        mujoco.mj_forward(model, data)
        pitch = math.asin(-float(data.xmat[foot][6]))  # foot pitch about y
        # Foot pitch increases 1:1 with ankle angle, so a positive pitch
        # means the level ankle lies below mid.
        if pitch > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


class Balance:
    """Lateral stabilizer: pelvis-roll feedback to hip roll, ankle roll, waist roll."""

    def __init__(self, hip_sign: float, kp_h: float, kd_h: float,
                 ankle_sign: float, kp_a: float,
                 waist_sign: float, kw: float) -> None:
        self.hip_sign, self.kp_h, self.kd_h = hip_sign, kp_h, kd_h
        self.ankle_sign, self.kp_a = ankle_sign, kp_a
        self.waist_sign, self.kw = waist_sign, kw
        self.prev_roll = 0.0
        self.rate = 0.0

    def update(self, roll: float, dt: float) -> None:
        self.rate = 0.7 * self.rate + 0.3 * (roll - self.prev_roll) / dt
        self.prev_roll = roll

    def apply(self, targets: dict[str, float]) -> None:
        u = self.kp_h * self.prev_roll + self.kd_h * self.rate
        hip = max(-0.15, min(0.15, self.hip_sign * u))
        targets["left_hip_roll_joint"] += hip
        targets["right_hip_roll_joint"] += hip
        ua = max(-0.15, min(0.15, self.ankle_sign * self.kp_a * self.prev_roll))
        targets["left_ankle_roll_joint"] = ua
        targets["right_ankle_roll_joint"] = -ua
        targets["waist_roll_joint"] = max(-0.4, min(0.4, self.waist_sign * self.kw * self.prev_roll))


def run_once(model, data, p: GaitParams, ankle_level: float, bal: Balance) -> dict:
    legs = ("left_hip_pitch_joint", "left_hip_roll_joint", "left_knee_joint", "left_ankle_pitch_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_knee_joint", "right_ankle_pitch_joint")
    extra = ("left_ankle_roll_joint", "right_ankle_roll_joint", "waist_roll_joint")
    act = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in legs + extra}
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    mujoco.mj_resetDataKeyframe(model, data, 0)
    stand = stand_leg_targets()
    p.ankle_level = ankle_level
    crouch = {n: HIP0 for n in stand}
    crouch.update({"left_knee_joint": KNEE0, "right_knee_joint": KNEE0,
                   "left_ankle_pitch_joint": ankle_level, "right_ankle_pitch_joint": ankle_level,
                   "left_hip_roll_joint": p.spread, "right_hip_roll_joint": -p.spread})
    crouch.update({"left_ankle_roll_joint": 0.0, "right_ankle_roll_joint": 0.0, "waist_roll_joint": 0.0})

    def set_legs(blend: float, targets: dict[str, float]) -> None:
        for n, aid in act.items():
            data.ctrl[aid] = (1.0 - blend) * crouch.get(n, 0.0) + blend * targets.get(n, 0.0)

    dt = model.opt.timestep
    x0 = float(data.xpos[pelvis][0])
    fell_at = None
    for i in range(int((SETTLE_S + WALK_S) / dt)):
        t = i * dt
        if t < SETTLE_S:
            blend = t / SETTLE_S
            targets = dict(crouch)
        else:
            blend = 1.0
            targets = leg_targets(t - SETTLE_S, p)
            bal.update(float(data.xquat[pelvis][1]), dt)  # quat x ~= roll for small angles
            bal.apply(targets)
        set_legs(blend, targets)
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
    ankle_level = solve_level_ankle(model, data)
    print(f"solved ankle_level: {ankle_level:+.3f} rad")

    grid = list(itertools.product(
        (1.2, 1.6),        # freq
        (1.0, -1.0),       # hip fb sign
        (1.0, -1.0),       # ankle fb sign
        (1.0, -1.0),       # waist fb sign
        (0.0, 0.04),       # weight-shift roll amp
    ))
    results = []
    for freq, hip_sign, ankle_sign, waist_sign, roll_amp in grid:
        p = GaitParams(freq=freq, roll_amp=roll_amp)
        bal = Balance(hip_sign=hip_sign, kp_h=0.6, kd_h=0.12,
                      ankle_sign=ankle_sign, kp_a=0.5,
                      waist_sign=waist_sign, kw=0.35)
        r = run_once(model, data, p, ankle_level, bal)
        results.append({"params": p, "bal": bal, **r})
    results.sort(key=lambda r: (r["fell_at"] is not None, -r["distance"]))
    print(f"\n{'fell':>5} {'dist(m)':>8} {'lat(m)':>7}  freq rollA  hipS ankS wstS")
    for r in results[:12]:
        p = r["params"]
        fell = f"{r['fell_at']:.1f}s" if r["fell_at"] is not None else "no"
        print(f"{fell:>5} {r['distance']:+8.3f} {r['lateral']:+7.3f}  {p.freq:.1f}  {p.roll_amp:.2f}  "
              f"{r['bal'].hip_sign:+.0f}   {r['bal'].ankle_sign:+.0f}   {r['bal'].waist_sign:+.0f}")


if __name__ == "__main__":
    main()
