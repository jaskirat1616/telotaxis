"""One MuJoCo scene: a checker floor, food spheres, pillars, and N vehicles.

Vehicles do not collide with each other. They do collide with the floor and
with every object, so a sphere stops a car through the solver. Other vehicles
still exist for vision, as spheres of one declared radius, but that is not a
contact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from telotaxis.constants import BODY_Z_M, HALF_TRACK_M, PHYSICS_DT, WHEEL_RADIUS_M
from telotaxis.vision import SceneObject


@dataclass(frozen=True)
class ArenaObject:
    object_id: str
    kind: str
    x: float
    y: float
    radius: float
    rgba: tuple[float, float, float, float]
    height: float = 0.0
    slab: tuple[float, float] | None = None

    def as_scene(self) -> SceneObject:
        return SceneObject(self.object_id, self.kind, self.x, self.y, self.radius)


def default_arena() -> list[ArenaObject]:
    return [
        ArenaObject("food-red", "food", 0.22, 0.42, 0.50, (0.80, 0.16, 0.20, 1)),
        ArenaObject("food-orange", "food", -0.58, -0.02, 0.40, (0.93, 0.48, 0.14, 1)),
        ArenaObject("food-yellow", "food", 0.88, -0.32, 0.34, (0.94, 0.76, 0.22, 1)),
        ArenaObject("food-deep", "food", 0.10, -0.78, 0.46, (0.84, 0.30, 0.12, 1)),
        ArenaObject("pil-a", "pillar", -1.40, 0.62, 0.055, (0.16, 0.16, 0.17, 1), height=0.70),
        ArenaObject("pil-b", "pillar", 1.50, 0.95, 0.050, (0.14, 0.14, 0.15, 1), height=0.88),
        ArenaObject("pil-c", "pillar", -0.10, 1.35, 0.048, (0.18, 0.18, 0.19, 1), height=0.52),
        ArenaObject("pil-d", "pillar", 1.20, -1.40, 0.060, (0.13, 0.13, 0.14, 1), height=0.76),
        ArenaObject("pil-e", "pillar", -1.65, -0.95, 0.045, (0.20, 0.20, 0.21, 1), height=0.44),
        ArenaObject(
            "slab-a", "pillar", -1.85, -0.10, 0.09,
            (0.42, 0.42, 0.44, 1), height=0.16, slab=(0.34, 0.07),
        ),
        ArenaObject(
            "slab-b", "pillar", 1.90, -0.05, 0.09,
            (0.38, 0.38, 0.40, 1), height=0.20, slab=(0.07, 0.42),
        ),
    ]


def _rgba(color: tuple[float, float, float, float]) -> str:
    return " ".join(f"{channel:.4f}" for channel in color)


def _vehicle_geoms(index: int, body: str) -> str:
    name = f"a{index}"
    if body == "robot":
        return f"""
        <geom name="{name}_deck" type="box" pos="0 0 0.055" size="0.20 0.13 0.028" mass="2.4"
              rgba="0.24 0.32 0.38 1" contype="2" conaffinity="1" friction="0.8 0.01 0.001"/>
        <geom name="{name}_mast" type="cylinder" pos="-0.02 0 0.18" size="0.018 0.10" mass="0.04"
              rgba="0.16 0.20 0.24 1" contype="0" conaffinity="0"/>
        <geom name="{name}_head" type="box" pos="-0.02 0 0.30" size="0.055 0.045 0.035" mass="0.06"
              rgba="0.18 0.22 0.26 1" contype="2" conaffinity="1"/>
        <geom name="{name}_eye" type="box" pos="0.04 0 0.305" size="0.012 0.028 0.012"
              rgba="0.95 0.55 0.18 1" contype="0" conaffinity="0" material="lamp"/>
        <geom name="{name}_nose" type="sphere" pos="0.16 0 0.07" size="0.02"
              rgba="0.95 0.78 0.35 1" contype="0" conaffinity="0" material="lamp"/>
        """
    # A low car: hood, cabin, windshield, lamps. Only the chassis contacts objects.
    return f"""
        <geom name="{name}_chassis" type="box" pos="0.00 0 0.050" size="0.20 0.090 0.028" mass="1.7"
              rgba="0.30 0.25 0.21 1" contype="2" conaffinity="1" friction="0.7 0.01 0.001"/>
        <geom name="{name}_hood" type="box" pos="0.145 0 0.058" size="0.075 0.082 0.018" mass="0.15"
              rgba="0.36 0.30 0.25 1" contype="0" conaffinity="0"/>
        <geom name="{name}_cabin" type="box" pos="-0.035 0 0.095" size="0.085 0.078 0.032" mass="0.2"
              rgba="0.20 0.18 0.17 1" contype="0" conaffinity="0"/>
        <geom name="{name}_glass" type="box" pos="0.055 0 0.100" size="0.012 0.070 0.022"
              rgba="0.55 0.62 0.68 0.9" contype="0" conaffinity="0"/>
        <geom name="{name}_nose" type="sphere" pos="0.215 0.055 0.055" size="0.016"
              rgba="1 0.92 0.72 1" contype="0" conaffinity="0" material="lamp"/>
        <geom name="{name}_nose2" type="sphere" pos="0.215 -0.055 0.055" size="0.016"
              rgba="1 0.92 0.72 1" contype="0" conaffinity="0" material="lamp"/>
        """


def _wheels(index: int) -> str:
    name = f"a{index}"
    r = WHEEL_RADIUS_M
    y = HALF_TRACK_M
    # Front wheels are visual. The hidden nose caster and the two driven rear
    # wheels are what the solver actually rolls.
    return f"""
        <body name="{name}_rl" pos="-0.145 {y:.3f} 0">
          <joint name="{name}_rl" type="hinge" axis="0 1 0" damping="0.02" armature="0.001"/>
          <geom name="{name}_wrl" type="cylinder" size="{r:.4f} 0.018" euler="1.5707963 0 0" mass="0.08"
                rgba="0.07 0.07 0.08 1" contype="2" conaffinity="1" friction="1.5 0.02 0.002"/>
        </body>
        <body name="{name}_rr" pos="-0.145 {-y:.3f} 0">
          <joint name="{name}_rr" type="hinge" axis="0 1 0" damping="0.02" armature="0.001"/>
          <geom name="{name}_wrr" type="cylinder" size="{r:.4f} 0.018" euler="1.5707963 0 0" mass="0.08"
                rgba="0.07 0.07 0.08 1" contype="2" conaffinity="1" friction="1.5 0.02 0.002"/>
        </body>
        <body name="{name}_fl" pos="0.145 {y:.3f} 0">
          <geom name="{name}_wfl" type="cylinder" size="{r:.4f} 0.016" euler="1.5707963 0 0" mass="0.02"
                rgba="0.07 0.07 0.08 1" contype="0" conaffinity="0"/>
        </body>
        <body name="{name}_fr" pos="0.145 {-y:.3f} 0">
          <geom name="{name}_wfr" type="cylinder" size="{r:.4f} 0.016" euler="1.5707963 0 0" mass="0.02"
                rgba="0.07 0.07 0.08 1" contype="0" conaffinity="0"/>
        </body>
        <body name="{name}_cast" pos="0.16 0 -0.022">
          <joint name="{name}_cast" type="ball" damping="0.0005"/>
          <geom name="{name}_cast" type="sphere" size="0.026" mass="0.04"
                rgba="0.10 0.10 0.11 1" contype="2" conaffinity="1" friction="0.04 0.001 0.001"/>
        </body>
        """


def build_spec(
    n: int,
    body: str,
    spawns: list[tuple[float, float, float]],
    arena: list[ArenaObject] | None = None,
) -> tuple[str, list[ArenaObject]]:
    if body not in ("car", "robot"):
        raise ValueError(f"unknown body {body!r}")
    if len(spawns) != n:
        raise ValueError("spawn count does not match the number of vehicles")
    arena = list(arena) if arena is not None else default_arena()
    parts = [
        f"""<mujoco model="flycar">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{PHYSICS_DT}" gravity="0 0 -9.81" iterations="12" cone="elliptic" integrator="implicitfast"/>
  <visual>
    <global offwidth="1400" offheight="800" azimuth="128" elevation="-32"/>
    <rgba fog="0.91 0.88 0.83 1" haze="0.91 0.88 0.83 1"/>
    <map znear="0.04" zfar="60" fogstart="18" fogend="40"/>
    <headlight ambient="0.32 0.31 0.29" diffuse="0.22 0.21 0.19" specular="0.04 0.04 0.03"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture name="checker" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.90 0.86 0.80" rgb2="0.76 0.71 0.64"/>
    <material name="floor" texture="checker" texrepeat="8 8" texuniform="false" reflectance="0.015"/>
    <material name="lamp" rgba="1 0.93 0.75 1" emission="0.45"/>
  </asset>
  <worldbody>
    <light name="sun" directional="true" pos="0 -1 8" dir="0.25 0.45 -1"
           diffuse="0.50 0.47 0.42" specular="0.06 0.05 0.04" castshadow="true"/>
    <light name="fill" directional="true" pos="0 2 5" dir="-0.1 -0.45 -1"
           diffuse="0.16 0.16 0.18" specular="0 0 0" castshadow="false"/>
    <geom name="floor" type="plane" size="40 40 0.1" material="floor"
          friction="1.05 0.005 0.0001" contype="1" conaffinity="1"/>
"""
    ]
    for obj in arena:
        rgba = _rgba(obj.rgba)
        if obj.kind == "food":
            parts.append(
                f'    <geom name="{obj.object_id}" type="sphere" pos="{obj.x:.4f} {obj.y:.4f} {obj.radius:.4f}" '
                f'size="{obj.radius:.4f}" rgba="{rgba}" contype="1" conaffinity="1" '
                f'friction="1.0 0.02 0.002"/>\n'
            )
        elif obj.slab is not None:
            sx, sy = obj.slab
            parts.append(
                f'    <geom name="{obj.object_id}" type="box" pos="{obj.x:.4f} {obj.y:.4f} {obj.height / 2:.4f}" '
                f'size="{sx:.4f} {sy:.4f} {obj.height / 2:.4f}" rgba="{rgba}" contype="1" conaffinity="1"/>\n'
            )
        else:
            half = obj.height / 2
            parts.append(
                f'    <geom name="{obj.object_id}" type="cylinder" pos="{obj.x:.4f} {obj.y:.4f} {half:.4f}" '
                f'size="{obj.radius:.4f} {half:.4f}" rgba="{rgba}" contype="1" conaffinity="1"/>\n'
            )
    for index, (x, y, yaw) in enumerate(spawns):
        parts.append(
            f"""    <body name="a{index}" pos="{x:.4f} {y:.4f} {BODY_Z_M:.4f}" euler="0 0 {yaw:.6f}">
      <freejoint name="a{index}_free"/>
      {_vehicle_geoms(index, body)}
      {_wheels(index)}
    </body>
"""
        )
    parts.append("  </worldbody>\n  <actuator>\n")
    for index in range(n):
        parts.append(
            f'    <velocity name="a{index}_rl" joint="a{index}_rl" kv="0.35" ctrlrange="-45 45"/>\n'
            f'    <velocity name="a{index}_rr" joint="a{index}_rr" kv="0.35" ctrlrange="-45 45"/>\n'
        )
    parts.append("  </actuator>\n</mujoco>\n")
    return "".join(parts), arena


def spawn_ring(
    n: int,
    rng: np.random.Generator,
    arena: list[ArenaObject],
) -> list[tuple[float, float, float]]:
    """Scatter vehicles on a ring, headings aimed so food starts in the field."""

    food = [obj for obj in arena if obj.kind == "food"]
    fx = float(np.mean([obj.x for obj in food]))
    fy = float(np.mean([obj.y for obj in food]))
    poses: list[tuple[float, float, float]] = []
    for _ in range(n * 80):
        ang = float(rng.uniform(0.0, 2.0 * math.pi))
        rad = float(rng.uniform(1.85, 2.75))
        x = rad * math.cos(ang)
        y = rad * math.sin(ang)
        if any(math.hypot(x - obj.x, y - obj.y) < obj.radius + 0.75 for obj in arena):
            continue
        if any(math.hypot(x - px, y - py) < 0.85 for px, py, _ in poses):
            continue
        heading = math.atan2(fy - y, fx - x) + float(rng.uniform(-0.45, 0.45))
        poses.append((x, y, heading))
        if len(poses) == n:
            return poses
    raise RuntimeError(f"could only place {len(poses)} of {n} vehicles")


class World:
    def __init__(
        self,
        n: int,
        body: str,
        seed: int,
        arena: list[ArenaObject] | None = None,
    ) -> None:
        import os
        import sys

        if sys.platform == "darwin":
            os.environ.setdefault("MUJOCO_GL", "cgl")
        import mujoco

        self.mujoco = mujoco
        self.n = 1 if body == "g1" else n
        self.body = body
        self.rng = np.random.default_rng(seed)
        if body == "g1":
            self._init_g1()
            return
        self.arena = list(arena) if arena is not None else default_arena()
        self.spawns = spawn_ring(self.n, self.rng, self.arena)
        xml, self.arena = build_spec(n, body, self.spawns, self.arena)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._bind()
        mujoco.mj_forward(self.model, self.data)

    def _bind(self) -> None:
        mujoco = self.mujoco
        model = self.model
        self.free_adr = np.array(
            [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"a{i}_free")] for i in range(self.n)],
            dtype=np.int32,
        )
        self.act_adr = np.array(
            [
                [
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"a{i}_rl"),
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"a{i}_rr"),
                ]
                for i in range(self.n)
            ],
            dtype=np.int32,
        )
        focus_geom = "chassis" if self.body == "car" else "deck"
        self.nose_geom = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"a{i}_{focus_geom}")
                for i in range(self.n)
            ],
            dtype=np.int32,
        )
        self._nose_rgba = self.model.geom_rgba[self.nose_geom].copy()
        self.agent_geoms = np.array(
            [
                gid
                for gid in range(model.ngeom)
                if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid).startswith("a")
            ],
            dtype=np.int32,
        )
        self.agent_geom_set = set(int(gid) for gid in self.agent_geoms)

    def _init_g1(self) -> None:
        from telotaxis.gait import Gait
        from telotaxis.track import BEACON_RADIUS_M, build, gates

        self.beacon_radius = BEACON_RADIUS_M
        self.model = build(self.mujoco)
        self.data = self.mujoco.MjData(self.model)
        self.gait = Gait(self.model, self.mujoco)
        self.pelvis = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.lap = gates()
        self.gate_index = 1
        self.gates_passed = 0
        self.cmd_speed = 0.0
        self.cmd_yaw = 0.0
        self.key_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
        self.free_adr = np.array([int(self.gait.pelvis_adr)], dtype=np.int32)
        self.nose_geom = np.zeros(1, dtype=np.int32)
        self._nose_rgba = np.ones((1, 4))
        self.agent_geom_set = set()
        self._place_g1()

    def _beacon_object(self) -> ArenaObject:
        x, y = self.lap[self.gate_index]
        return ArenaObject("gate", "food", x, y, self.beacon_radius, (0.93, 0.24, 0.14, 1))

    def _place_g1(self) -> None:
        from telotaxis.track import heading

        self.mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        self.gait.reset()
        self.gate_index = 1
        self.gates_passed = 0
        self.cmd_speed = 0.0
        self.cmd_yaw = 0.0
        start = self.lap[0]
        yaw = heading(start, self.lap[1])
        half = 0.5 * yaw
        self.data.qpos[0:7] = (
            start[0],
            start[1],
            0.755,
            math.cos(half),
            0.0,
            0.0,
            math.sin(half),
        )
        self.data.mocap_pos[0] = (self.lap[1][0], self.lap[1][1], 1.05)
        self.arena = [self._beacon_object()]
        self.mujoco.mj_forward(self.model, self.data)

    def _advance_gate(self) -> None:
        x, y = float(self.data.qpos[0]), float(self.data.qpos[1])
        bx, by = self.lap[self.gate_index]
        if math.hypot(x - bx, y - by) > 0.95:
            return
        self.gate_index = (self.gate_index + 1) % len(self.lap)
        self.gates_passed += 1
        bx, by = self.lap[self.gate_index]
        self.data.mocap_pos[0] = (bx, by, 1.05)
        self.arena = [self._beacon_object()]

    def reset(self, seed: int) -> None:
        if self.body == "g1":
            self.rng = np.random.default_rng(seed)
            self._place_g1()
            return
        self.mujoco.mj_resetData(self.model, self.data)
        self.rng = np.random.default_rng(seed)
        self.spawns = spawn_ring(self.n, self.rng, self.arena)
        for index, (x, y, yaw) in enumerate(self.spawns):
            adr = int(self.free_adr[index])
            half = 0.5 * yaw
            self.data.qpos[adr : adr + 7] = (
                x,
                y,
                BODY_Z_M,
                math.cos(half),
                0.0,
                0.0,
                math.sin(half),
            )
        self.data.ctrl[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def scene_objects(self) -> list[SceneObject]:
        return [obj.as_scene() for obj in self.arena]

    def poses(self) -> np.ndarray:
        """Return (n, 3) of x, y, yaw."""

        out = np.zeros((self.n, 3), dtype=np.float64)
        qpos = self.data.qpos
        for index, adr in enumerate(self.free_adr):
            adr = int(adr)
            out[index, 0] = qpos[adr]
            out[index, 1] = qpos[adr + 1]
            qw, qx, qy, qz = qpos[adr + 3 : adr + 7]
            out[index, 2] = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return out

    def heights(self) -> np.ndarray:
        return np.array([self.data.qpos[int(adr) + 2] for adr in self.free_adr], dtype=np.float64)

    def upright(self) -> np.ndarray:
        """World-z of each body's own up axis. 1 is flat, 0 is on its side."""

        out = np.zeros(self.n, dtype=np.float64)
        for index, adr in enumerate(self.free_adr):
            qw, qx, qy, qz = self.data.qpos[int(adr) + 3 : int(adr) + 7]
            out[index] = 1.0 - 2.0 * (qx * qx + qy * qy)
        return out

    def set_drive(self, speed: np.ndarray, yaw_rate: np.ndarray) -> None:
        if self.body == "g1":
            self.cmd_speed = float(speed[0])
            self.cmd_yaw = float(yaw_rate[0])
            return
        left = (speed - yaw_rate * HALF_TRACK_M) / WHEEL_RADIUS_M
        right = (speed + yaw_rate * HALF_TRACK_M) / WHEEL_RADIUS_M
        for index in range(self.n):
            self.data.ctrl[int(self.act_adr[index, 0])] = float(left[index])
            self.data.ctrl[int(self.act_adr[index, 1])] = float(right[index])

    def step(self, n_steps: int) -> None:
        mujoco = self.mujoco
        if self.body == "g1":
            dt = float(self.model.opt.timestep)
            count = max(1, int(round(n_steps * PHYSICS_DT / dt)))
            for _ in range(count):
                self.gait.step(self.model, self.data, self.cmd_speed, self.cmd_yaw, dt, int(self.pelvis))
                self._advance_gate()
            return
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

    def agent_agent_contacts(self) -> int:
        count = 0
        agents = self.agent_geom_set
        for index in range(self.data.ncon):
            geom1 = int(self.data.contact[index].geom1)
            geom2 = int(self.data.contact[index].geom2)
            if geom1 in agents and geom2 in agents:
                # A vehicle's own wheel touching its own caster is not a pair of vehicles.
                name1 = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom1)
                name2 = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom2)
                if name1.split("_", 1)[0] != name2.split("_", 1)[0]:
                    count += 1
        return count

    def mark_focus(self, index: int | None) -> None:
        if self.body == "g1":
            return
        self.model.geom_rgba[self.nose_geom] = self._nose_rgba
        if index is None:
            return
        gid = int(self.nose_geom[index])
        base = self._nose_rgba[index]
        self.model.geom_rgba[gid] = (
            min(1.0, float(base[0]) + 0.35),
            min(1.0, float(base[1]) + 0.22),
            min(1.0, float(base[2]) + 0.08),
            1.0,
        )
