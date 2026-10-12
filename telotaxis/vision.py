"""Analytic vision. Bearing and angular size go in. Column drive comes out.

There is no camera, no ray cast, no occlusion, and no colour. A column keeps
the strongest object, because a pile of weak ones must not outrank one large
one. Column centres are the MaleCNS optic-lobe hexes, not a made-up lattice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from telotaxis.constants import LOOM_HALF_DEG, SIGMA_DEG, SIGMA_GROW


@dataclass(frozen=True)
class SceneObject:
    object_id: str
    kind: str
    x: float
    y: float
    radius: float


def encode(
    objects: list[SceneObject],
    x: float,
    y: float,
    yaw: float,
    column_centers_deg: np.ndarray,
    *,
    stimulus: bool,
) -> tuple[np.ndarray, dict]:
    """Return per-column drive in [0, 1]. Positive azimuth is the agent's left."""

    centers = np.asarray(column_centers_deg, dtype=np.float64)
    drive = np.zeros(centers.shape[0], dtype=np.float64)
    info: dict = {"stimulus": bool(stimulus and objects), "driven_columns": 0, "strongest": None}
    if not stimulus or not objects:
        return drive, info

    ox = np.fromiter((obj.x for obj in objects), dtype=np.float64, count=len(objects))
    oy = np.fromiter((obj.y for obj in objects), dtype=np.float64, count=len(objects))
    radius = np.fromiter((obj.radius for obj in objects), dtype=np.float64, count=len(objects))
    dx = ox - x
    dy = oy - y
    distance = np.hypot(dx, dy)
    bearing = np.arctan2(dy, dx) - yaw
    bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
    bearing_deg = np.degrees(bearing)
    ratio = np.where(distance > radius, radius / np.maximum(distance, 1e-9), 1.0)
    angular = np.degrees(np.arcsin(np.clip(ratio, -1.0, 1.0)))
    loom = angular / (angular + LOOM_HALF_DEG)
    sigma = SIGMA_DEG + SIGMA_GROW * angular
    delta = centers[:, None] - bearing_deg[None, :]
    spatial = np.exp(-0.5 * (delta / sigma[None, :]) ** 2)
    per_object = spatial * loom[None, :]
    drive = per_object.max(axis=1)
    peak = per_object.max(axis=0)
    winner = int(np.argmax(peak))
    info["driven_columns"] = int(np.count_nonzero(drive > 0.05))
    info["strongest"] = {
        "object_id": objects[winner].object_id,
        "bearing_deg": float(bearing_deg[winner]),
        "angular_deg": float(angular[winner]),
    }
    return drive, info
