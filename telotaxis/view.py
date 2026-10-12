"""The swarm frame: arena, this car's MaleCNS somata, descending rates below."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from telotaxis.constants import ONSET_STAND_S, RATE_THRESHOLD_HZ

WIDTH = 1600
HEIGHT = 900
BRAIN_W = 480
ARENA_W = WIDTH - BRAIN_W
DESC_H = 34
FOOT_H = 52
ARENA_H = 656
CHART_H = HEIGHT - ARENA_H - DESC_H - FOOT_H

INK = (236, 233, 226)
MUTED = (168, 164, 156)
DIM = (120, 116, 108)
BG = (14, 16, 18)
LINE = (46, 48, 52)
TRACE_L = (126, 182, 214)
TRACE_R = (224, 122, 61)

_FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
_FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def _font(size: int, bold: bool = False):
    try:
        return ImageFont.truetype(_FONT_BOLD if bold else _FONT, size)
    except OSError:
        return ImageFont.load_default()


class Composer:
    def __init__(self, swarm) -> None:
        self.swarm = swarm
        mujoco = swarm.world.mujoco
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.lookat[:] = np.array([0.06, -0.02, 0.02])
        self.cam.distance = 7.2
        self.cam.azimuth = 132.0
        self.cam.elevation = -33.0
        self.speed = 1.0
        self.paused = False
        self.focus_pin: int | None = None
        self.orbit = 148.0
        self.elevation = -16.0
        self._renderer = None
        self._plate = None
        self._brain_im = None
        self._live_i = 0
        self._desc_key = None
        self._font_title = _font(22, bold=True)
        self._font_sub = _font(15)
        self._font_small = _font(13)
        self._font_tiny = _font(12)
        self._font_micro = _font(11)
        self._points = _spray(swarm.pathway.position, seed=7)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def focus(self) -> int:
        if self.focus_pin is not None:
            return int(self.focus_pin) % self.swarm.n
        if self.swarm.t <= 0.0:
            return 0
        return int(self.swarm.t / 3.0) % self.swarm.n

    def frame(self) -> np.ndarray:
        focus = self.focus()
        if self.swarm.body == "g1":
            pose = self.swarm.world.poses()[0]
            self.cam.lookat[:] = [pose[0], pose[1], 0.9]
            self.cam.distance = 5.2
            self.cam.azimuth = float(np.degrees(pose[2]) + self.orbit)
            self.cam.elevation = self.elevation
        self.swarm.world.mark_focus(focus)
        arena = self._arena()
        brain = self._brain(focus)
        canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
        canvas.paste(Image.fromarray(arena), (0, 0))
        canvas.paste(Image.fromarray(brain), (ARENA_W, 0))
        draw = ImageDraw.Draw(canvas)
        self._title(draw)
        self._description(draw)
        self._charts(draw, focus)
        self._footer(draw)
        return np.asarray(canvas)

    def frame_live(self) -> np.ndarray:
        """Same picture as frame(), with the brain redrawn a few times a second.

        The arena is the part that has to track the body. Rebuilding the whole
        1600x900 plate and the soma cloud on every tick was the lag.
        """

        focus = self.focus()
        if self.swarm.body == "g1":
            pose = self.swarm.world.poses()[0]
            self.cam.lookat[:] = [pose[0], pose[1], 0.9]
            self.cam.distance = 5.2
            self.cam.azimuth = float(np.degrees(pose[2]) + self.orbit)
            self.cam.elevation = self.elevation
        self.swarm.world.mark_focus(focus)
        if self._plate is None:
            self._plate = Image.new("RGB", (WIDTH, HEIGHT), BG)
        arena = self._arena()
        self._plate.paste(Image.fromarray(arena), (0, 0))
        if self._brain_im is None or self._live_i % 3 == 0:
            self._brain_im = Image.fromarray(self._brain(focus))
        self._plate.paste(self._brain_im, (ARENA_W, 0))
        draw = ImageDraw.Draw(self._plate)
        # The arena paste wipes the title. The footer never changes, and the
        # description only changes when a gate is passed, so those stay put.
        self._title(draw)
        if self.swarm.body == "g1":
            desc_key = ("g1", int(getattr(self.swarm.world, "gates_passed", 0)))
        else:
            desc_key = (self.swarm.body, self.swarm.n)
        if desc_key != self._desc_key:
            self._description(draw)
            self._footer(draw)
            self._desc_key = desc_key
        self._charts(draw, focus)
        self._live_i += 1
        # Half resolution for the window. The full plate is what made each
        # Tk blit the slow part of the tick.
        return np.ascontiguousarray(np.asarray(self._plate)[::2, ::2])

    def _renderer_for(self):
        if self._renderer is None:
            mujoco = self.swarm.world.mujoco
            renderer = mujoco.Renderer(self.swarm.world.model, height=ARENA_H, width=ARENA_W)
            renderer._scene_option.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
            self._renderer = renderer
        return self._renderer

    def _arena(self) -> np.ndarray:
        renderer = self._renderer_for()
        renderer.update_scene(self.swarm.world.data, camera=self.cam)
        return np.ascontiguousarray(renderer.render())

    def _title(self, draw) -> None:
        if self.swarm.body == "g1":
            title = "One connectome. One Unitree G1."
        else:
            noun = "cars" if self.swarm.body == "car" else "robots"
            title = f"One connectome. Twelve {noun}. Nothing scripted." if self.swarm.n == 12 else (
                f"One connectome. {self.swarm.n} {noun}. Nothing scripted."
            )
        fill = (236, 232, 220) if self.swarm.body == "g1" else (28, 30, 34)
        draw.text((22, 18), title, font=self._font_title, fill=fill)

    def _brain(self, focus: int) -> np.ndarray:
        image = Image.new("RGB", (BRAIN_W, ARENA_H), (0, 0, 0))
        vis = self.swarm.pathway.vis[focus]
        owners = self._points[1]
        flashed = 1.0 - np.exp(-vis[owners] * 0.9)
        activity = 0.36 + 0.64 * flashed
        header = 86
        gap = 8
        view_h = (ARENA_H - header - gap) // 2
        dorsal = _project(self._points[0], activity, BRAIN_W, view_h, 0, 1, 150.0, (BRAIN_W / 2, view_h * 0.52))
        lateral = _project(self._points[0], activity, BRAIN_W, view_h, 0, 2, 132.0, (BRAIN_W / 2, 124.0))
        image.paste(Image.fromarray(dorsal), (0, header))
        image.paste(Image.fromarray(lateral), (0, header + view_h + gap))
        draw = ImageDraw.Draw(image)
        name = "G1" if self.swarm.body == "g1" else ("CAR" if self.swarm.body == "car" else "ROBOT")
        meta = self.swarm.pathway.meta
        draw.text((16, 12), f"{name}-{focus + 1:02d}  ·  MaleCNS v1.0", font=self._font_sub, fill=INK)
        draw.text(
            (16, 34),
            f"{meta['neurons']:,} neurons    {meta['synapses']:,} synapses",
            font=self._font_tiny,
            fill=MUTED,
        )
        info = self.swarm.scene_info[focus] if self.swarm.scene_info else {}
        driven = int(info.get("driven_columns", 0))
        left, right = self.swarm.pathway.descending()
        state = self.swarm.states[focus]
        draw.text(
            (16, 52),
            f"columns {driven}/{self.swarm.pathway.n_columns}    L {left[focus]:.1f} Hz    R {right[focus]:.1f} Hz",
            font=self._font_tiny,
            fill=MUTED,
        )
        kind = self.swarm.pathway.brightest_type(focus)
        draw.text((16, 70), f"{state}    brightest {kind}", font=self._font_tiny, fill=TRACE_R if state == "locomoting" else DIM)
        variant = "exact" if self.swarm.stimulus else "stimulus absent"
        clock = f"t = {self.swarm.t:.2f} s    {'paused' if self.paused else f'x{self.speed:g}'}    {variant}"
        tw = draw.textlength(clock, font=self._font_tiny)
        draw.text((BRAIN_W - tw - 12, 12), clock, font=self._font_tiny, fill=MUTED)
        draw.text((16, header + view_h - 16), "soma positions from the paper", font=self._font_micro, fill=(70, 78, 92))
        return np.asarray(image)

    def _description(self, draw) -> None:
        y = ARENA_H
        draw.rectangle((0, y, WIDTH, y + DESC_H), fill=BG)
        if self.swarm.body == "g1":
            passed = getattr(self.swarm.world, "gates_passed", 0)
            text = (
                f"Unitree G1 on a lap. The connectome sees the next gate. Gates passed {passed}. "
                "Only the two DNp pools steer."
            )
        else:
            noun = "cars" if self.swarm.body == "car" else "robots"
            text = (
                f"Twelve {noun}, one MaleCNS pathway. Lamina hexes see bearing and angular size. "
                "Only the two DNp pools steer."
                if self.swarm.n == 12
                else f"{self.swarm.n} {noun}, one MaleCNS pathway. Only the two DNp pools steer."
            )
        draw.text((18, y + 8), text, font=self._font_small, fill=INK)

    def _charts(self, draw, focus: int) -> None:
        top = ARENA_H + DESC_H
        draw.rectangle((0, top, WIDTH, top + CHART_H), fill=(10, 12, 14))
        left_box = (16, top + 28, ARENA_W - 24, top + CHART_H - 12)
        right_box = (ARENA_W + 8, top + 28, WIDTH - 16, top + CHART_H - 12)
        hist = self.swarm.history
        times = np.asarray(hist.t, dtype=np.float64) if hist.t else np.zeros(0)
        has = int(times.size) > 0
        rl = hist.column("rate_left")
        rr = hist.column("rate_right")
        speed = hist.column("speed")
        focus_l = rl[:, focus] if has else np.zeros(0)
        focus_r = rr[:, focus] if has else np.zeros(0)
        rate_max = 40.0
        if has:
            rate_max = max(40.0, float(np.max(focus_l)), float(np.max(focus_r))) * 1.15
        _chart(draw, left_box, times, [(focus_l, TRACE_L), (focus_r, TRACE_R)], rate_max, RATE_THRESHOLD_HZ)
        series = []
        if has:
            order = [i for i in range(self.swarm.n) if i != focus] + [focus]
            for index in order:
                color = (210, 206, 198) if index == focus else (88, 86, 82)
                series.append((speed[:, index], color))
        _chart(draw, right_box, times, series, 0.8, None)
        name = "G1" if self.swarm.body == "g1" else ("car" if self.swarm.body == "car" else "robot")
        draw.text((left_box[0], top + 6), f"DNp readout, {name}", font=self._font_small, fill=MUTED)
        if has:
            draw.text((left_box[0] + 230, top + 8), f"left {focus_l[-1]:.1f} Hz", font=self._font_tiny, fill=TRACE_L)
            draw.text((left_box[0] + 360, top + 8), f"right {focus_r[-1]:.1f} Hz", font=self._font_tiny, fill=TRACE_R)
        drive_label = "forward drive" if self.swarm.n == 1 else f"forward drive, all {name}s"
        draw.text((right_box[0], top + 6), drive_label, font=self._font_small, fill=MUTED)
        if self.swarm.n > 1:
            draw.text((right_box[2] - 148, top + 8), "one line per vehicle", font=self._font_micro, fill=DIM)

    def _footer(self, draw) -> None:
        y = ARENA_H + DESC_H + CHART_H
        draw.rectangle((0, y, WIDTH, HEIGHT), fill=BG)
        draw.text(
            (16, y + 4),
            "Live window: space pause, V vision, R reset, arrows orbit, esc quits.",
            font=self._font_tiny,
            fill=MUTED,
        )
        draw.text(
            (16, y + 24),
            "MaleCNS v1.0, weight at least 3, optic lobe to descending neurons. "
            f"Onset timer: still {ONSET_STAND_S:.1f} s, then 150 ms above threshold. "
            + (
                "Gait and pelvis wrench are engineering. MuJoCo integrates the contacts. "
                if self.swarm.body == "g1"
                else "Wheels are engineering. "
            )
            + "Central brain and the ventral cord are not in this pathway.",
            font=self._font_micro,
            fill=DIM,
        )


def _spray(position: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    copies = 3
    base = np.repeat(position, copies, axis=0)
    jitter = rng.normal(0.0, 0.012, size=base.shape).astype(np.float32)
    owners = np.repeat(np.arange(position.shape[0], dtype=np.int32), copies)
    return base + jitter, owners


def _project(points, brightness, width, height, horizontal, vertical, scale, origin) -> np.ndarray:
    ox, oy = origin
    sx = ox + points[:, horizontal] * scale
    sy = oy - points[:, vertical] * scale
    xi = np.rint(sx).astype(np.int32)
    yi = np.rint(sy).astype(np.int32)
    keep = (xi >= 1) & (xi < width - 1) & (yi >= 1) & (yi < height - 1)
    field = np.zeros((height, width), np.float32)
    np.maximum.at(field, (yi[keep], xi[keep]), brightness[keep])
    tone = np.clip(field, 0.0, 1.0)
    rgb = np.zeros((height, width, 3), np.uint8)
    rgb[..., 0] = np.clip(tone * 210, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(tone * 226, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(tone * 250, 0, 255).astype(np.uint8)
    return rgb


def _chart(draw, box, times, series, y_max, threshold) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=LINE)
    pad = 8
    if threshold is not None and y_max > 0:
        ty = y1 - pad - (threshold / y_max) * (y1 - y0 - 2 * pad)
        draw.line((x0 + 2, ty, x1 - 2, ty), fill=(60, 52, 40))
    if len(times) < 2:
        return
    t0 = max(0.0, float(times[-1]) - 14.0)
    window = times >= t0
    ts = times[window]
    if len(ts) < 2:
        return
    span = max(float(ts[-1] - ts[0]), 1e-6)
    width = max(1, x1 - x0 - 4)

    def xy(sample):
        px = x0 + 2 + (ts - ts[0]) / span * width
        py = y1 - pad - np.clip(sample / y_max, 0.0, 1.0) * (y1 - y0 - 2 * pad)
        return np.stack([px, py], axis=1)

    for values, color in series:
        pts = xy(np.asarray(values, dtype=np.float64)[window])
        budget = min(width, 160)
        if len(pts) > budget:
            pts = pts[np.linspace(0, len(pts) - 1, budget).astype(int)]
        flat = [tuple(p) for p in pts]
        if len(flat) >= 2:
            draw.line(flat, fill=color, width=1)
