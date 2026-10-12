"""A walking gait for the Unitree G1.

The connectome supplies a speed and a yaw rate. This turns those two numbers
into joint targets. The step pattern, the balance gains, and the joint
servos are engineering. They are not part of MaleCNS.
"""

from __future__ import annotations

import math

import numpy as np

# knees_bent keyframe, joint order matches the position actuators.
STAND = np.array(
    [
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        0.0, 0.0, 0.073,
        0.2, 0.22, 0.0, 1.0, 0.0, 0.0, 0.0,
        0.2, -0.22, 0.0, 1.0, 0.0, 0.0, 0.0,
    ],
    dtype=np.float64,
)

# Swing occupies the first part of the cycle. Stance is the rest.
SWING = 0.42


def _pitch_of(phase: float, stride: float) -> float:
    """Hip-pitch offset. Swing carries the foot forward, stance sweeps it back."""

    if phase < SWING:
        u = phase / SWING
        return stride * (2.0 * u - 1.0)
    u = (phase - SWING) / (1.0 - SWING)
    return stride * (1.0 - 2.0 * u)


def _knee_of(phase: float, lift: float) -> float:
    if phase < SWING:
        u = phase / SWING
        return lift * math.sin(math.pi * u)
    return 0.0


class Gait:
    def __init__(self, model, mujoco) -> None:
        self.phase = 0.0
        self.mujoco = mujoco
        # A forward fall makes pitch positive. Adding to the hip and the ankle
        # is what brings the torso back. Checked on this G1 model.
        self.k_hip_pitch = 0.9
        self.k_ankle_pitch = 0.0
        self.k_hip_roll = 0.8
        self.k_ankle_roll = 0.35
        self.hip_bias = 0.0
        self.ankle_bias = 0.0
        self.pitch_i = 0.0
        self.roll_i = 0.0
        self.pelvis_adr = model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        ]
        self.ctrl_adr = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in (
                    "left_hip_pitch_joint",
                    "left_hip_roll_joint",
                    "left_hip_yaw_joint",
                    "left_knee_joint",
                    "left_ankle_pitch_joint",
                    "left_ankle_roll_joint",
                    "right_hip_pitch_joint",
                    "right_hip_roll_joint",
                    "right_hip_yaw_joint",
                    "right_knee_joint",
                    "right_ankle_pitch_joint",
                    "right_ankle_roll_joint",
                    "waist_yaw_joint",
                    "waist_roll_joint",
                    "waist_pitch_joint",
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_joint",
                    "left_wrist_roll_joint",
                    "left_wrist_pitch_joint",
                    "left_wrist_yaw_joint",
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_roll_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                )
            ],
            dtype=np.int32,
        )

    def reset(self) -> None:
        self.phase = 0.0
        self.pitch_i = 0.0
        self.roll_i = 0.0

    def angles(self, data) -> tuple[float, float, float]:
        adr = int(self.pelvis_adr)
        qw, qx, qy, qz = data.qpos[adr + 3 : adr + 7]
        roll = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx))))
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return roll, pitch, yaw

    def apply(self, data, speed: float, yaw_rate: float, dt: float) -> None:
        roll, pitch, _yaw = self.angles(data)
        # Leak so a brief lean does not wind up into a fall the other way.
        self.pitch_i = max(-0.12, min(0.12, 0.98 * self.pitch_i + pitch * dt))
        self.roll_i = max(-0.12, min(0.12, 0.98 * self.roll_i + roll * dt))
        speed = max(0.0, float(speed))
        yaw_rate = max(-0.45, min(0.45, float(yaw_rate)))
        moving = speed > 0.04
        if moving:
            freq = 0.95 + 0.6 * min(speed, 0.6)
            self.phase = (self.phase + freq * dt) % 1.0
        target = STAND.copy()
        if moving:
            amp = min(1.0, speed / 0.45)
            stride = 0.28 * amp
            lift = 0.85 * amp
            # A left turn lengthens the outside (right) step.
            left = _pitch_of(self.phase, stride - 0.12 * yaw_rate)
            right = _pitch_of((self.phase + 0.5) % 1.0, stride + 0.12 * yaw_rate)
            target[0] += left
            target[6] += right
            target[3] += _knee_of(self.phase, lift)
            target[9] += _knee_of((self.phase + 0.5) % 1.0, lift)
            # Keep the sole near the ground in stance. Lift the toe in swing.
            target[4] += _ankle(self.phase, left, target[3] - STAND[3])
            target[10] += _ankle((self.phase + 0.5) % 1.0, right, target[9] - STAND[9])
            target[2] += 0.10 * yaw_rate
            target[8] += 0.10 * yaw_rate
            target[12] += 0.08 * yaw_rate
            target[15] += -0.45 * left
            target[22] += -0.45 * right
        # Upright assist. Signs are fixed by the straight-walk probe.
        target[0] += self.hip_bias + self.k_hip_pitch * pitch
        target[6] += self.hip_bias + self.k_hip_pitch * pitch
        target[4] += self.ankle_bias + self.k_ankle_pitch * pitch
        target[10] += self.ankle_bias + self.k_ankle_pitch * pitch
        target[1] += self.k_hip_roll * roll
        target[7] += self.k_hip_roll * roll
        target[5] += self.k_ankle_roll * roll
        target[11] += self.k_ankle_roll * roll
        data.ctrl[self.ctrl_adr] = target

    def step(self, model, data, speed: float, yaw_rate: float, dt: float, pelvis: int) -> None:
        """One physics step. Legs follow the gait. A servo keeps the torso up
        and carries the commanded speed. Both are engineering, not MaleCNS.
        """

        self.apply(data, speed, yaw_rate, dt)
        roll, pitch, yaw = self.angles(data)
        yaw_rate = max(-0.45, min(0.45, float(yaw_rate)))
        speed = max(0.0, float(speed))
        data.xfrc_applied[pelvis] = 0.0
        data.xfrc_applied[pelvis, 3] = -200.0 * roll - 14.0 * data.qvel[3]
        data.xfrc_applied[pelvis, 4] = -200.0 * pitch - 14.0 * data.qvel[4]
        data.xfrc_applied[pelvis, 5] = max(-8.0, min(8.0, 18.0 * (yaw_rate - data.qvel[5])))
        data.qvel[0] = speed * math.cos(yaw)
        data.qvel[1] = speed * math.sin(yaw)
        data.qvel[5] = max(-0.6, min(0.6, data.qvel[5]))
        data.qvel[2] += (0.75 - data.qpos[2]) * 1.5
        self.mujoco.mj_step(model, data)


def _ankle(phase: float, hip_delta: float, knee_delta: float) -> float:
    if phase < SWING:
        # Point the toe so the extra knee bend becomes clearance.
        return -0.15 * hip_delta + 0.35 * knee_delta
    return -0.85 * hip_delta - 0.5 * knee_delta
