"""An oval lap and one Unitree G1. The mesh is the Menagerie model."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "assets" / "unitree_g1" / "scene_mjx.xml"

# Clockwise oval. From the west gate the centre sits to the runner's left,
# so a left turn, the positive yaw of this project, stays on the lap.
A_M = 8.5
B_M = 4.8
N_GATES = 20
BEACON_RADIUS_M = 0.22


def gates() -> list[tuple[float, float]]:
    points = []
    for index in range(N_GATES):
        theta = math.pi + 2.0 * math.pi * index / N_GATES
        points.append((A_M * math.cos(theta), B_M * math.sin(theta)))
    return points


def heading(here: tuple[float, float], there: tuple[float, float]) -> float:
    return math.atan2(there[1] - here[1], there[0] - here[0])


def build(mujoco):
    spec = mujoco.MjSpec.from_file(str(ROOT))
    floor = next(geom for geom in spec.geoms if geom.name == "floor")
    floor.material = ""
    floor.rgba[:] = [0.38, 0.36, 0.33, 1.0]
    world = spec.worldbody
    world.add_light(
        name="sun",
        pos=[0.0, -2.0, 10.0],
        dir=[0.15, 0.35, -1.0],
        directional=True,
        diffuse=[0.85, 0.82, 0.74],
        specular=[0.15, 0.14, 0.12],
        castshadow=True,
    )
    points = gates()
    for index, (x, y) in enumerate(points):
        nxt = points[(index + 1) % N_GATES]
        mx = 0.5 * (x + nxt[0])
        my = 0.5 * (y + nxt[1])
        yaw = heading((x, y), nxt)
        length = math.hypot(nxt[0] - x, nxt[1] - y)
        dash = world.add_geom(
            name=f"dash{index}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[length * 0.22, 0.045, 0.008],
            pos=[mx, my, 0.012],
            quat=[math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
            rgba=[0.92, 0.90, 0.84, 1.0],
            contype=0,
            conaffinity=0,
        )
        del dash
        red = index % 2 == 0
        for side, scale in ((-1.0, A_M + 1.35), (1.0, A_M - 1.15)):
            # Curbs sit on a slightly larger and smaller oval, visual only.
            pass
        ox = (A_M + 1.45) / A_M * x
        oy = (B_M + 1.15) / B_M * y
        ix = (A_M - 1.55) / A_M * x
        iy = (B_M - 1.15) / B_M * y
        curb = [0.86, 0.18, 0.16, 1] if red else [0.93, 0.93, 0.90, 1]
        for name, px, py in ((f"curb_o{index}", ox, oy), (f"curb_i{index}", ix, iy)):
            world.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.28, 0.10, 0.045],
                pos=[px, py, 0.045],
                quat=[math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
                rgba=curb,
                contype=0,
                conaffinity=0,
            )
    bx, by = points[1]
    beacon = world.add_body(name="beacon", mocap=True, pos=[bx, by, 1.05])
    beacon.add_geom(
        name="beacon",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[BEACON_RADIUS_M, 0.0, 0.0],
        rgba=[0.93, 0.24, 0.14, 1.0],
        contype=0,
        conaffinity=0,
    )
    model = spec.compile()
    model.vis.global_.offwidth = 1400
    model.vis.global_.offheight = 800
    model.actuator_gainprm[:, 0] *= 5.0
    model.actuator_biasprm[:, 1] *= 5.0
    model.actuator_biasprm[:, 2] *= 4.0
    return model
