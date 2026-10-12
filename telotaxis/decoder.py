"""Two descending rates become a speed and a yaw rate.

Which side is stronger comes from the connectome. The sign that turns the car
toward that side is declared here. Once the onset timer has passed and the
drive has held, there is no stop state: a blocked car keeps being told to go.
"""

from __future__ import annotations

import math

from telotaxis.constants import (
    MAX_SPEED_MPS,
    MAX_YAW_RATE,
    MIN_CRUISE_MPS,
    ONSET_HOLD_S,
    ONSET_STAND_S,
    RATE_THRESHOLD_HZ,
)


class Decoder:
    def __init__(self) -> None:
        self.locomoting = False
        self.hold_s = 0.0
        self.state = "standing"

    def reset(self) -> None:
        self.locomoting = False
        self.hold_s = 0.0
        self.state = "standing"

    def update(self, rate_left: float, rate_right: float, dt: float, t: float) -> tuple[float, float]:
        drive = 0.5 * (rate_left + rate_right)
        if t < ONSET_STAND_S:
            self.hold_s = 0.0
            self.state = "standing"
            return 0.0, 0.0
        if not self.locomoting:
            if drive >= RATE_THRESHOLD_HZ:
                self.hold_s += dt
            else:
                self.hold_s = 0.0
            if self.hold_s >= ONSET_HOLD_S:
                self.locomoting = True
            else:
                self.state = "waiting"
                return 0.0, 0.0
        self.state = "locomoting"
        extra = max(0.0, drive - RATE_THRESHOLD_HZ)
        speed = MIN_CRUISE_MPS + (MAX_SPEED_MPS - MIN_CRUISE_MPS) * math.tanh(extra / 40.0)
        # Positive yaw is a left turn. Positive (left − right) means the left
        # posterior descending pool is the stronger one.
        yaw = MAX_YAW_RATE * math.tanh((rate_left - rate_right) / 22.0)
        return speed, yaw
