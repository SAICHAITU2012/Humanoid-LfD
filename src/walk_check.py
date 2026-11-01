"""Validate walking configs through the EXACT viewer control path (15 s trials).

Mirrors src/view_g1_web.py frame-for-frame so tuner results transfer.
Run: ../.venv/bin/python src/walk_check.py
"""
from __future__ import annotations

import itertools
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from gait import GaitParams, HIP0, KNEE0, walk_targets

MODEL = Path(__file__).resolve().parents[1] / "vendor/mujoco_menagerie/unitree_g1/scene_with_hands.xml"
SETTLE_S = 0.8
RAMP_S = 1.2
FALL_Z = 0.45
WALK_S = 15.0
WALK_WAIST_PITCH = 0.08
WALK_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_knee_joint", "left_ankle_pitch_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_knee_joint", "right_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint", "waist_roll_joint", "waist_pitch_joint",
]


def crouch_pose(gait: GaitParams, waist_pitch: float = WALK_WAIST_PITCH) -> dict[str, float]:
    pose = {n: 0.0 for n in WALK_JOINTS}
    for s in ("left", "right"):
        pose[f"{s}_hip_pitch_joint"] = HIP0
        pose[f"{s}_knee_joint"] = KNEE0
        pose[f"{s}_ankle_pitch_joint"] = gait.ankle_level
        pose[f"{s}_hip_roll_joint"] = gait.spread if s == "left" else -gait.spread
    pose["waist_pitch_joint"] = waist_pitch
    return pose


def run(gait: GaitParams, speed: float = 1.0, secs: float = WALK_S,
        waist_pitch: float = WALK_WAIST_PITCH) -> dict:
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    stand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, stand)
    for aid in range(model.nu):
        jid = model.actuator_trnid[aid, 0]
        data.ctrl[aid] = data.qpos[model.jnt_qposadr[jid]]
    mujoco.mj_forward(model, data)

    walk_aids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in WALK_JOINTS}
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    foot_body = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link")
                 for s in ("left", "right")}
    crouch = crouch_pose(gait, waist_pitch)
    dt = model.opt.timestep
    bal = {"pr": 0.0, "pp": 0.0, "rr": 0.0, "rp": 0.0}
    x0 = float(data.xpos[pelvis][0])
    phase = "settle"
    phase_t0 = 0.0
    fell = None
    min_z = 1.0

    def foot_contact(side: str) -> bool:
        body = foot_body[side]
        for c in range(data.ncon):
            g1, g2 = data.contact[c].geom1, data.contact[c].geom2
            if (model.geom_bodyid[g1] == body or model.geom_bodyid[g2] == body) and data.contact[c].dist < 0.005:
                return True
        return False

    for i in range(int(secs / dt)):
        if phase == "settle":
            blend = min(1.0, (data.time - phase_t0) / SETTLE_S)
            for n, a in walk_aids.items():
                data.ctrl[a] = blend * crouch[n]
            if blend >= 1.0:
                phase = "step"
                phase_t0 = data.time
        elif phase == "step":
            wt = data.time - phase_t0
            step_scale = min(1.0, wt / RAMP_S)
            gait_speed = replace(gait, freq=gait.freq * speed)
            leg_targets = walk_targets(wt, gait_speed, step_scale)
            roll = float(data.xquat[pelvis][1])
            pitch = float(data.xquat[pelvis][2])
            bal["rr"] = 0.7 * bal["rr"] + 0.3 * (roll - bal["pr"]) / dt
            bal["rp"] = 0.7 * bal["rp"] + 0.3 * (pitch - bal["pp"]) / dt
            bal["pr"], bal["pp"] = roll, pitch
            u_roll = max(-0.08, min(0.08, 1.2 * roll + 0.25 * bal["rr"]))
            u_pit = max(-0.12, min(0.12, 1.5 * pitch + 0.3 * bal["rp"]))
            l_c, r_c = foot_contact("left"), foot_contact("right")
            if l_c and r_c:
                for s in ("left", "right"):
                    leg_targets[f"{s}_hip_pitch_joint"] += u_pit
            elif l_c:
                leg_targets["left_hip_roll_joint"] -= u_roll
                leg_targets["left_hip_pitch_joint"] -= u_pit
            elif r_c:
                leg_targets["right_hip_roll_joint"] -= u_roll
                leg_targets["right_hip_pitch_joint"] -= u_pit
            for n, a in walk_aids.items():
                data.ctrl[a] = leg_targets.get(n, crouch[n])
            z = float(data.xpos[pelvis][2])
            min_z = min(min_z, z)
            if z < FALL_Z or not np.isfinite(data.qpos).all():
                fell = round(data.time, 2)
                break
        mujoco.mj_step(model, data)
    x = data.xpos[pelvis]
    return {"fell": fell, "dist": float(x[0]) - x0, "lat": float(x[1]), "min_z": min_z}


def main() -> None:
    print(f"{'fell':>5} {'dist':>7} {'lat':>7} {'minZ':>5}  freq reach knee  dip shift wst")
    for freq, reach, knee, dip, shift, wst in itertools.product(
            (0.38, 0.42, 0.46), (0.22, 0.28), (0.40,), (0.45, 0.60), (0.09,), (0.08, 0.14)):
        gait = GaitParams(freq=freq, reach=reach, knee_amp=knee, dip_amp=dip, shift=shift)
        r = run(gait, waist_pitch=wst)
        # waist bias is applied via crouch; emulate by patching the module constant
        mark = " <<<" if r["fell"] is None else ""
        print(f"{str(r['fell']):>5} {r['dist']:+7.3f} {r['lat']:+7.3f} {r['min_z']:.2f}  "
              f"{freq:.2f}  {reach:.2f}  {knee:.2f}  {dip:.2f}  {shift:.2f}  {wst:.2f}{mark}")


if __name__ == "__main__":
    main()
