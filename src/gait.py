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

# Baseline crouch: thighs slightly forward, knees bent, feet kept level by ankle0.
HIP0 = -0.30
KNEE0 = 0.60
KNEE_LAG = 0.35 * math.pi  # knee flexes after the hip starts swinging forward


@dataclass
class GaitParams:
    freq: float = 1.6        # stride frequency (Hz)
    hip_amp: float = 0.25    # hip pitch swing amplitude (rad)
    knee_amp: float = 0.60   # extra knee flexion during swing (rad)
    roll_amp: float = 0.04   # weight-shift hip roll amplitude (rad)
    spread: float = 0.10     # stance hip-roll offset, feet apart (rad)
    hip_sign: float = 1.0    # +1/-1: hip swing phase direction
    roll_sign: float = 1.0   # +1/-1: weight-shift direction
    ankle_level: float = -0.30  # ankle pitch that levels the foot at (HIP0, KNEE0)


def leg_targets(t: float, p: GaitParams) -> dict[str, float]:
    """Joint targets for the 8 pitch/roll leg joints at time t (seconds)."""
    targets: dict[str, float] = {}
    for side, phase_off in (("left", 0.0), ("right", math.pi)):
        s = 2.0 * math.pi * p.freq * t + phase_off
        swing = math.sin(s)
        hip = HIP0 + p.hip_sign * p.hip_amp * swing
        knee = KNEE0 + p.knee_amp * max(0.0, math.sin(s + KNEE_LAG))
        # Foot stays level: hip, knee, and ankle each add 1:1 to foot pitch
        # (measured by FK), so the ankle cancels the other two deviations.
        ankle = p.ankle_level - p.hip_sign * p.hip_amp * swing - p.knee_amp * max(0.0, math.sin(s + KNEE_LAG))
        roll = p.roll_sign * p.roll_amp * swing
        targets[f"{side}_hip_pitch_joint"] = hip
        targets[f"{side}_knee_joint"] = knee
        targets[f"{side}_ankle_pitch_joint"] = ankle
        # Positive hip roll moves the left foot outward and the right foot
        # inward (mirrored joints), so spreading needs opposite signs.
        roll_offset = p.spread if side == "left" else -p.spread
        targets[f"{side}_hip_roll_joint"] = roll + roll_offset
    return targets


def stand_leg_targets() -> dict[str, float]:
    """The model's stand pose for the same joints (all zeros)."""
    return {name: 0.0 for name in LEG_JOINTS}
