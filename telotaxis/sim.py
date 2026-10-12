"""Closed loop. The body spends each interval under the previous command."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from telotaxis.brain import Pathway
from telotaxis.constants import COUPLING_S, NEAR_SURFACE_M, NEURAL_STEPS, PEER_RADIUS_M, PHYSICS_STEPS
from telotaxis.decoder import Decoder
from telotaxis.vision import SceneObject, encode
from telotaxis.world import ArenaObject, World


@dataclass
class History:
    t: list[float] = field(default_factory=list)
    rate_left: list[np.ndarray] = field(default_factory=list)
    rate_right: list[np.ndarray] = field(default_factory=list)
    speed: list[np.ndarray] = field(default_factory=list)

    def append(self, t: float, left: np.ndarray, right: np.ndarray, speed: np.ndarray) -> None:
        self.t.append(float(t))
        self.rate_left.append(np.asarray(left, dtype=np.float32).copy())
        self.rate_right.append(np.asarray(right, dtype=np.float32).copy())
        self.speed.append(np.asarray(speed, dtype=np.float32).copy())
        if len(self.t) > 4000:
            self.t = self.t[-3000:]
            self.rate_left = self.rate_left[-3000:]
            self.rate_right = self.rate_right[-3000:]
            self.speed = self.speed[-3000:]

    def column(self, name: str) -> np.ndarray:
        rows = getattr(self, name)
        if not rows:
            return np.zeros((0, 1), dtype=np.float32)
        return np.stack(rows, axis=0)


@dataclass
class Summary:
    variant: str
    n: int
    locomoting: int
    against_object: int
    nearer_food: int
    median_food_delta_m: float
    displacement_m: tuple[float, float]
    neurons: int
    synapses: int


class Swarm:
    def __init__(self, n: int = 12, seed: int = 1, body: str = "car", stimulus: bool = True) -> None:
        if n < 1:
            raise ValueError("need at least one vehicle")
        self.n = n
        self.seed = seed
        self.body = body
        self.stimulus = stimulus
        self.world = World(n, body, seed)
        self.pathway = Pathway(n, seed)
        self.decoders = [Decoder() for _ in range(n)]
        self.history = History()
        self.t = 0.0
        self.held_speed = np.zeros(n, dtype=np.float64)
        self.held_yaw = np.zeros(n, dtype=np.float64)
        self.command_speed = np.zeros(n, dtype=np.float64)
        self.command_yaw = np.zeros(n, dtype=np.float64)
        self.column_drive = np.zeros((n, self.pathway.n_columns), dtype=np.float64)
        self.scene_info: list[dict] = [{} for _ in range(n)]
        self.states = ["standing"] * n
        self._start = self.world.poses().copy()
        self._food_start = self._food_distance(self._start)

    @property
    def variant(self) -> str:
        return "exact" if self.stimulus else "stimulus-absent"

    def reset(self, seed: int | None = None, stimulus: bool | None = None) -> None:
        if seed is not None:
            self.seed = seed
        if stimulus is not None:
            self.stimulus = stimulus
        self.world.reset(self.seed)
        self.pathway.reset(self.seed)
        for decoder in self.decoders:
            decoder.reset()
        self.history = History()
        self.t = 0.0
        self.held_speed[:] = 0.0
        self.held_yaw[:] = 0.0
        self._start = self.world.poses().copy()
        self._food_start = self._food_distance(self._start)
        self.states = ["standing"] * self.n

    def step_interval(self) -> None:
        poses = self.world.poses()
        static = self.world.scene_objects()
        drives = np.zeros_like(self.column_drive)
        centers = self.pathway.column_azimuth
        for index in range(self.n):
            seen = list(static)
            for other in range(self.n):
                if other == index:
                    continue
                seen.append(
                    SceneObject(f"peer-{other}", self.body, float(poses[other, 0]), float(poses[other, 1]), PEER_RADIUS_M)
                )
            drive, info = encode(
                seen,
                float(poses[index, 0]),
                float(poses[index, 1]),
                float(poses[index, 2]),
                centers,
                stimulus=self.stimulus,
            )
            drives[index] = drive
            self.scene_info[index] = info
        self.column_drive = drives
        self.pathway.advance(drives, NEURAL_STEPS)
        rate_left, rate_right = self.pathway.descending()
        speed = np.zeros(self.n, dtype=np.float64)
        yaw = np.zeros(self.n, dtype=np.float64)
        for index, decoder in enumerate(self.decoders):
            speed[index], yaw[index] = decoder.update(float(rate_left[index]), float(rate_right[index]), COUPLING_S, self.t)
            self.states[index] = decoder.state
        self.world.set_drive(self.held_speed, self.held_yaw)
        self.world.step(PHYSICS_STEPS)
        self.held_speed = speed
        self.held_yaw = yaw
        self.command_speed = speed
        self.command_yaw = yaw
        self.t += COUPLING_S
        self.history.append(self.t, rate_left, rate_right, speed)

    def run(self, seconds: float, progress: bool = False) -> None:
        steps = int(round(seconds / COUPLING_S))
        mark = 0.0
        for _ in range(steps):
            self.step_interval()
            if progress and self.t >= mark:
                moving = sum(state == "locomoting" for state in self.states)
                print(f"  t={self.t:5.2f}s  locomoting {moving}/{self.n}", flush=True)
                mark += 2.0

    def _food_distance(self, poses: np.ndarray) -> np.ndarray:
        food = [obj for obj in self.world.arena if obj.kind == "food"]
        out = np.empty(self.n, dtype=np.float64)
        for index in range(self.n):
            out[index] = min(math.hypot(poses[index, 0] - obj.x, poses[index, 1] - obj.y) - obj.radius for obj in food)
        return out

    def surface_gap(self, poses: np.ndarray | None = None) -> np.ndarray:
        if poses is None:
            poses = self.world.poses()
        out = np.empty(self.n, dtype=np.float64)
        for index in range(self.n):
            out[index] = min(
                math.hypot(poses[index, 0] - obj.x, poses[index, 1] - obj.y) - obj.radius for obj in self.world.arena
            )
        return out

    def summarize(self) -> Summary:
        poses = self.world.poses()
        delta = poses[:, :2] - self._start[:, :2]
        displacement = np.hypot(delta[:, 0], delta[:, 1])
        food_now = self._food_distance(poses)
        food_delta = self._food_start - food_now
        return Summary(
            variant=self.variant,
            n=self.n,
            locomoting=sum(decoder.locomoting for decoder in self.decoders),
            against_object=int(np.count_nonzero(self.surface_gap(poses) < NEAR_SURFACE_M)),
            nearer_food=int(np.count_nonzero(food_delta > 0.30)),
            median_food_delta_m=float(np.median(food_delta)),
            displacement_m=(float(displacement.min()), float(displacement.max())),
            neurons=int(self.pathway.meta["neurons"]),
            synapses=int(self.pathway.meta["synapses"]),
        )
