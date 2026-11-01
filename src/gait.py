"""Sinusoidal gait generator for the Unitree G1 right/left leg position servos.

Pure-python on purpose: both the headless tuner (tune_walk.py) and the live
viewer (view_g1_web.py) import this, so tuned parameters transfer directly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

LEG_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_knee_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_knee_joint",
    "right_ankle_pitch_joint",
]

# Baseline crouch: shallow bend keeps the CoM inside the foot support polygon
# (a deeper crouch topples backward under position control; measured in sim).
HIP0 = -0.15
KNEE0 = 0.30


@dataclass
class GaitParams:
    # Defaults are the best headless sweep result (tune_walk.py + waist-pitch
    # bias trials): ~0.55 m forward over ~4 s before stumbling, best lateral
    # stability of the sweep. See PROOF_OF_WORK.md for the tuning history.
    freq: float = 0.42       # full gait-cycle frequency (Hz); T = 1/freq covers 2 steps
    reach: float = 0.25      # hip pitch travel from rear extreme to forward extreme (rad)
    knee_amp: float = 0.40   # knee flexion during swing (rad)
    dip_amp: float = 0.45    # both-knee flexion at each swing start (unloads swing leg)
    shift: float = 0.09      # weight-shift hip roll amplitude (rad)
    spread: float = 0.10     # stance hip-roll offset, feet apart (rad)
    hip_sign: float = 1.0    # +1/-1: walk direction
    phase0: float = 0.75 * math.pi  # start phase: legs mid-stance, CoM centered
    roll_sign: float = -1.0  # unused by walk_targets; kept for leg_targets
    hip_amp: float = 0.18    # used by leg_targets only
    roll_amp: float = 0.0    # used by leg_targets only
    ankle_level: float = -0.15  # ankle pitch that levels the foot at (HIP0, KNEE0)


def walk_targets(t: float, p: GaitParams, step_scale: float = 1.0) -> dict[str, float]:
    """Quasi-static walk with real double-support windows.

    Per leg (xi = 2*pi*freq*t + phase, left offset 0, right offset pi):
      hip   = HIP0 + 0.5*reach*cos(xi):  +1 rear extreme, -1 forward extreme
      knee  = KNEE0 + knee_amp*bump(xi): lifts just after the rear extreme
             (xi in 0.05..0.45 pi) so the foot never drags past neutral
      ankle = ankle_level - (hip-HIP0) - knee_extra   (foot level, 1:1 FK rule)
    Hip rolls carry a smooth CoM shift at half cycle frequency,
    -A*sin(2*pi*freq*t + phase0 - 0.75*pi) subtracted from both rolls: it is
    81% onto the stance foot at each lift-off and peaks mid-swing (left-roll
    decrease => pelvis +y, verified by FK).
    """
    th = 2.0 * math.pi * p.freq * t + p.phase0
    shift_angle = p.shift * step_scale * math.sin(th - 0.75 * math.pi)
    # Body dip at every swing start: flexing BOTH knees briefly unloads the
    # swing leg, otherwise ground friction locks its knee in a force chain.
    dip = p.dip_amp * step_scale * max(0.0, math.cos(th)) ** 8
    dip_r = p.dip_amp * step_scale * max(0.0, math.cos(th + math.pi)) ** 8
    targets: dict[str, float] = {}
    for side, phase_off in (("left", 0.0), ("right", math.pi)):
        xi = th + phase_off
        u = math.cos(xi)
        hip = HIP0 + p.hip_sign * 0.5 * p.reach * step_scale * u
        w = (xi - 0.05 * math.pi) / (0.40 * math.pi)
        knee_extra = p.knee_amp * step_scale * (math.sin(math.pi * w) ** 2 if 0.0 < w < 1.0 else 0.0)
        knee_extra += dip if side == "left" else dip_r
        knee = KNEE0 + knee_extra
        ankle = p.ankle_level - p.hip_sign * 0.5 * p.reach * step_scale * u - knee_extra
        targets[f"{side}_hip_pitch_joint"] = hip
        targets[f"{side}_knee_joint"] = knee
        targets[f"{side}_ankle_pitch_joint"] = ankle
        roll_offset = p.spread if side == "left" else -p.spread
        targets[f"{side}_hip_roll_joint"] = roll_offset - shift_angle
    return targets


def leg_targets(t: float, p: GaitParams, step_scale: float = 1.0) -> dict[str, float]:
    """Joint targets for the 8 pitch/roll leg joints at time t (seconds).

    step_scale ramps the swing amplitudes in (0..1) so the first steps are small.
    """
    targets: dict[str, float] = {}
    for side, phase_off in (("left", 0.0), ("right", math.pi)):
        s = 2.0 * math.pi * p.freq * t + phase_off
        swing = math.sin(s)
        hip = HIP0 + p.hip_sign * p.hip_amp * step_scale * swing
        # The knee flexes while the hip swings FORWARD (protraction); a knee
        # timed to the backward swing drags the toe and kicks the body over.
        protract = max(0.0, -p.hip_sign * math.cos(s))
        knee_extra = p.knee_amp * step_scale * protract
        # Foot stays level: hip, knee, and ankle each add 1:1 to foot pitch
        # (measured by FK), so the ankle cancels the other two deviations.
        ankle = p.ankle_level - p.hip_sign * p.hip_amp * step_scale * swing - knee_extra
        roll = p.roll_sign * p.roll_amp * step_scale * swing
        targets[f"{side}_hip_pitch_joint"] = hip
        targets[f"{side}_knee_joint"] = KNEE0 + knee_extra
        targets[f"{side}_ankle_pitch_joint"] = ankle
        # Positive hip roll moves the left foot outward and the right foot
        # inward (mirrored joints), so spreading needs opposite signs.
        roll_offset = p.spread if side == "left" else -p.spread
        targets[f"{side}_hip_roll_joint"] = roll + roll_offset
    return targets


def stand_leg_targets() -> dict[str, float]:
    """The model's stand pose for the same joints (all zeros)."""
    return {name: 0.0 for name in LEG_JOINTS}
