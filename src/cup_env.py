from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from build_cup_scene import ensure_scene


ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
HAND_JOINTS = [
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
]
HAND_CLOSED = np.array([-0.55, -0.35, -1.10, 1.20, 1.35, 1.20, 1.35])
FINGER_TIPS = ["right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link"]
TABLE_TOP_Z = 0.730
COASTER_XY = np.array([0.40, 0.01])
COASTER_TOP_Z = 0.746
CUP_HALF_HEIGHT = 0.055
LIFT_HEIGHT = 0.12
GRASP_POSE = np.array([-0.6, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
CUP_MASS = 0.04
OBS_DIM = 37
ACT_DIM = 8
CONTROL_HZ = 30


def _ids(model: mujoco.MjModel, kind: mujoco.mjtObj, names: list[str]) -> list[int]:
    result = [mujoco.mj_name2id(model, kind, name) for name in names]
    if min(result) < 0:
        raise RuntimeError(f"Missing MuJoCo element: {names[result.index(min(result))]}")
    return result


def solve_arm_ik(
    model: mujoco.MjModel,
    source_data: mujoco.MjData,
    arm_joint_ids: list[int],
    wrist_body: int,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    max_iters: int = 100,
    seed_q: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Damped-least-squares IK for the G1 arm's wrist pose, with joint-limit clipping."""
    qpos_adrs = model.jnt_qposadr[arm_joint_ids]
    dof_adrs = model.jnt_dofadr[arm_joint_ids]
    data = mujoco.MjData(model)
    data.qpos[:] = source_data.qpos
    if seed_q is not None:
        data.qpos[qpos_adrs] = seed_q
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(max_iters):
        mujoco.mj_forward(model, data)
        current_position = data.xpos[wrist_body].copy()
        current_rotation = data.xmat[wrist_body].reshape(3, 3).copy()
        position_error = target_position - current_position
        rotation_error = Rotation.from_matrix(target_rotation @ current_rotation.T).as_rotvec()
        error = np.concatenate((position_error, rotation_error))
        if np.linalg.norm(position_error) < 0.002 and np.linalg.norm(rotation_error) < 0.02:
            break
        mujoco.mj_jacBody(model, data, jacp, jacr, wrist_body)
        jacobian = np.vstack((jacp[:, dof_adrs], jacr[:, dof_adrs]))
        step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 0.03 * np.eye(6), error)
        data.qpos[qpos_adrs] += np.clip(step, -0.08, 0.08)
        for joint_id, qpos_adr in zip(arm_joint_ids, qpos_adrs):
            if model.jnt_limited[joint_id]:
                low, high = model.jnt_range[joint_id]
                data.qpos[qpos_adr] = np.clip(data.qpos[qpos_adr], low, high)
    mujoco.mj_forward(model, data)
    return data.qpos[qpos_adrs].copy(), float(np.linalg.norm(target_position - data.xpos[wrist_body]))


class CupEnv:
    """Randomized G1 cup pick-and-place: reset -> observation -> joint-target action -> success check."""

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(ensure_scene()))
        self.data = mujoco.MjData(self.model)
        stand = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if stand < 0:
            raise RuntimeError("The G1 model has no stand keyframe")
        self.stand = stand
        self.arm_joints = _ids(self.model, mujoco.mjtObj.mjOBJ_JOINT, ARM_JOINTS)
        self.arm_actuators = _ids(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, ARM_JOINTS)
        self.hand_actuators = _ids(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, HAND_JOINTS)
        self.hand_joints = _ids(self.model, mujoco.mjtObj.mjOBJ_JOINT, HAND_JOINTS)
        self.tip_bodies = _ids(self.model, mujoco.mjtObj.mjOBJ_BODY, FINGER_TIPS)
        self.wrist_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
        self.cup_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_cup")
        self.cup_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_cup_free")
        self.cup_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_cup_geom")
        self.coaster_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_coaster")
        if min(self.wrist_body, self.cup_body, self.cup_joint, self.cup_geom, self.coaster_geom) < 0:
            raise RuntimeError("Cup scene is missing the expected G1 wrist or task object")
        self.arm_qpos_adrs = self.model.jnt_qposadr[self.arm_joints]
        self.hand_qpos_adrs = self.model.jnt_qposadr[self.hand_joints]
        self.cup_qpos_adr = self.model.jnt_qposadr[self.cup_joint]
        self.coaster_xy = COASTER_XY.copy()
        self.arm_low = self.model.jnt_range[self.arm_joints, 0]
        self.arm_high = self.model.jnt_range[self.arm_joints, 1]
        self.hand_qpos0 = self.data.qpos[self.hand_qpos_adrs].copy()
        # Canonical wrist->cup offset at the calibrated GRASP_POSE; the expert re-uses it for any spawn.
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.stand)
        self.data.qpos[self.arm_qpos_adrs] = GRASP_POSE
        mujoco.mj_forward(self.model, self.data)
        self.grasp_rotation = self.data.xmat[self.wrist_body].reshape(3, 3).copy()
        cup_start = self.data.xpos[self.tip_bodies].mean(axis=0) + np.array([-0.05, -0.02, 0.0])
        cup_start[2] = TABLE_TOP_Z + CUP_HALF_HEIGHT
        self.wrist_to_cup_offset = self.grasp_rotation.T @ (cup_start - self.data.xpos[self.wrist_body])
        self.grasp_q = GRASP_POSE.copy()
        # Ready pose: wrist above-left of the workspace, solved once so every episode starts from a reachable home.
        self.home_q, home_err = solve_arm_ik(
            self.model, self.data, self.arm_joints, self.wrist_body,
            np.array([0.25, -0.25, 1.10]), self.grasp_rotation,
        )
        if home_err > 0.02:
            raise RuntimeError(f"Home ready pose unreachable (IK error {home_err:.3f} m)")
        # Fixed-base setup: reaching forward tips the free-standing keyframe pose over, so the
        # pelvis pose is re-imposed every step (disclosed as fixed-base manipulation in the report).
        root_joint = 0
        if self.model.jnt_type[root_joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError("Expected the pelvis free joint at qpos address 0")
        self.root_qpos_adr = self.model.jnt_qposadr[root_joint]
        self.root_dof_adr = self.model.jnt_dofadr[root_joint]
        mujoco.mj_forward(self.model, self.data)
        self.root_qpos0 = self.data.qpos[self.root_qpos_adr:self.root_qpos_adr + 7].copy()

    # -- scene state -------------------------------------------------------

    def reset(self, coaster_xy: np.ndarray, cup_mass: float = CUP_MASS, friction_scale: float = 1.0) -> None:
        """Grasp-ready start with a randomized GOAL: the arm sits at the exact calibrated grasp
        pose with the cup seeded at the fingertips (the scripted demo's construction — the close
        is only robust there), and the coaster (the target) is repositioned per episode."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.stand)
        self.data.qpos[self.arm_qpos_adrs] = GRASP_POSE
        self.model.body_mass[self.cup_body] = cup_mass
        self.model.geom_friction[self.cup_geom, 0] = 3.0 * friction_scale
        self.model.geom_pos[self.coaster_geom, 0] = coaster_xy[0]
        self.model.geom_pos[self.coaster_geom, 1] = coaster_xy[1]
        self.coaster_xy = np.asarray(coaster_xy, dtype=np.float64).copy()
        mujoco.mj_forward(self.model, self.data)
        cup_start = self.data.xpos[self.tip_bodies].mean(axis=0) + np.array([-0.05, -0.02, 0.0])
        cup_start[2] = TABLE_TOP_Z + CUP_HALF_HEIGHT
        self.data.qpos[self.cup_qpos_adr:self.cup_qpos_adr + 3] = cup_start
        self.data.qpos[self.cup_qpos_adr + 3:self.cup_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0
        self._last_cmd = GRASP_POSE.copy()
        mujoco.mj_forward(self.model, self.data)

    def grasp_position(self, cup_position: np.ndarray) -> np.ndarray:
        return cup_position - self.grasp_rotation @ self.wrist_to_cup_offset

    def wrist_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.xpos[self.wrist_body].copy(), self.data.xmat[self.wrist_body].reshape(3, 3).copy()

    def cup_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.xpos[self.cup_body].copy(), self.data.xmat[self.cup_body].reshape(3, 3).copy()

    def hand_flexion(self) -> float:
        """Mean absolute finger flexion, 0 = open (stand pose), 1 = HAND_CLOSED."""
        q = self.data.qpos[self.hand_qpos_adrs]
        span = np.abs(HAND_CLOSED - self.hand_qpos0)
        progress = np.abs(q - self.hand_qpos0) / np.maximum(span, 1e-3)
        return float(np.clip(progress.mean(), 0.0, 1.0))

    def hand_contact_count(self) -> int:
        count = 0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            bodies = [self.model.geom_bodyid[contact.geom1], self.model.geom_bodyid[contact.geom2]]
            if self.cup_body not in bodies:
                continue
            other = bodies[1] if bodies[0] == self.cup_body else bodies[0]
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, other) or ""
            if "hand" in name or "wrist" in name:
                count += 1
        return count

    def cup_coaster_contact(self) -> bool:
        for i in range(self.data.ncon):
            if {self.data.contact[i].geom1, self.data.contact[i].geom2} == {self.cup_geom, self.coaster_geom}:
                return True
        return False

    # -- observations and actions -----------------------------------------

    def observation(self) -> np.ndarray:
        wrist_pos, wrist_rot = self.wrist_pose()
        cup_pos, cup_rot = self.cup_pose()
        rotvec = Rotation.from_matrix(cup_rot).as_rotvec()
        vel = self.data.qvel[self.cup_joint:self.cup_joint + 3] if self.model.jnt_type[self.cup_joint] == mujoco.mjtJoint.mjJNT_FREE else np.zeros(3)
        obs = np.concatenate([
            wrist_pos,
            Rotation.from_matrix(wrist_rot).as_rotvec(),
            cup_pos,
            rotvec,
            vel,
            self.data.qpos[self.arm_qpos_adrs],
            self.data.qpos[self.hand_qpos_adrs],
            cup_pos - wrist_pos,
            cup_pos[:2] - self.coaster_xy,
            [cup_pos[2] - (COASTER_TOP_Z + CUP_HALF_HEIGHT)],
            [float(self.hand_contact_count())],
            [self.hand_flexion()],
        ])
        assert obs.shape == (OBS_DIM,), f"obs dim {obs.shape[0]} != {OBS_DIM}"
        return obs.astype(np.float32)

    def step(self, action: np.ndarray, steps_per_control: int | None = None) -> None:
        """Apply [7 arm joint targets, hand amount]; arm targets are clipped to joint limits."""
        action = np.asarray(action, dtype=np.float64)
        arm_target = np.clip(action[:7], self.arm_low, self.arm_high)
        hand_amount = float(np.clip(action[7], 0.0, 1.0))
        for actuator_id, value in zip(self.arm_actuators, arm_target):
            self.data.ctrl[actuator_id] = value
        for actuator_id, value in zip(self.hand_actuators, HAND_CLOSED * hand_amount):
            self.data.ctrl[actuator_id] = value
        if steps_per_control is None:
            steps_per_control = max(1, round((1.0 / CONTROL_HZ) / self.model.opt.timestep))
        for _ in range(steps_per_control):
            mujoco.mj_step(self.model, self.data)

    def success(self) -> tuple[bool, dict]:
        cup_pos, cup_rot = self.cup_pose()
        xy_error = float(np.linalg.norm(cup_pos[:2] - self.coaster_xy))
        tilt_deg = float(np.degrees(np.arccos(np.clip(cup_rot[2, 2], -1.0, 1.0))))
        resting = cup_pos[2] < COASTER_TOP_Z + CUP_HALF_HEIGHT + 0.02
        released = self.hand_flexion() < 0.25  # hand opened; the open fingers may still rest against the cup
        ok = xy_error < 0.05 and tilt_deg < 20.0 and resting and released
        return ok, {
            "xy_error_m": round(xy_error, 4), "tilt_deg": round(tilt_deg, 2),
            "resting": bool(resting), "released": bool(released),
            "cup_x": round(float(cup_pos[0]), 4), "cup_y": round(float(cup_pos[1]), 4), "cup_z": round(float(cup_pos[2]), 4),
        }
