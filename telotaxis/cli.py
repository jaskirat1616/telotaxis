"""Commands: live, run, frame, record."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from telotaxis.sim import Swarm
from telotaxis.view import Composer


def build(args) -> Swarm:
    return Swarm(n=args.cars, seed=args.seed, body=args.body, stimulus=args.variant != "absent")


def _print_summary(summary) -> None:
    print(
        f"{summary.variant:16}  {summary.neurons:,} neurons  {summary.synapses:,} synapses  "
        f"locomoting {summary.locomoting:2}/{summary.n}   "
        f"against an object {summary.against_object:2}/{summary.n}   "
        f"nearer food {summary.nearer_food:2}/{summary.n}   "
        f"median food closure {summary.median_food_delta_m:+.2f} m   "
        f"displacement {summary.displacement_m[0]:.2f}–{summary.displacement_m[1]:.2f} m"
    )


def cmd_run(args) -> None:
    variants = ["exact", "absent"] if args.variant == "both" else [args.variant]
    print("Telotaxis. MaleCNS visual pathway, closed loop, one interval of delay.")
    for variant in variants:
        args.variant = variant
        swarm = build(args)
        print(f"\n{swarm.variant}  seed {args.seed}  {args.body} x{args.cars}  {args.seconds:.1f} s")
        swarm.run(args.seconds, progress=not args.quiet)
        _print_summary(swarm.summarize())


def cmd_frame(args) -> None:
    swarm = build(args)
    if args.at > 0:
        swarm.run(args.at, progress=True)
    composer = Composer(swarm)
    try:
        image = composer.frame()
    finally:
        composer.close()
    _save(args.out, [image], fps=1)
    print(f"wrote {args.out}  t={swarm.t:.2f}s  {swarm.variant}")


def cmd_record(args) -> None:
    swarm = build(args)
    composer = Composer(swarm)
    frames = []
    stride = max(1, int(round((1.0 / args.fps) / 0.015)))
    steps = int(round(args.seconds / 0.015))
    try:
        for index in range(steps):
            swarm.step_interval()
            if index % stride == 0:
                frames.append(composer.frame())
                if len(frames) % 5 == 0:
                    print(f"  frame {len(frames)}  t={swarm.t:.2f}s", flush=True)
    finally:
        composer.close()
    _save(args.out, frames, fps=args.fps)
    _print_summary(swarm.summarize())
    print(f"wrote {args.out}  {len(frames)} frames")


def cmd_live(args) -> None:
    import glfw
    from OpenGL.GL import (
        GL_LINEAR, GL_QUADS, GL_RGB, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
        GL_TEXTURE_MIN_FILTER, GL_UNSIGNED_BYTE, glBegin, glBindTexture, glEnable,
        glEnd, glGenTextures, glTexCoord2f, glTexImage2D, glTexParameteri, glVertex2f, glViewport,
    )

    swarm = build(args)
    composer = Composer(swarm)
    if not glfw.init():
        raise RuntimeError("GLFW failed to start")
    glfw.window_hint(glfw.SCALE_TO_MONITOR, glfw.TRUE)
    window = glfw.create_window(1280, 720, "Telotaxis · MaleCNS", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("GLFW window failed to open")
    glfw.make_context_current(window)
    texture = glGenTextures(1)
    state = {"acc": 0.0, "last": time.perf_counter()}

    def on_key(win, key, _scancode, action, _mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)
        elif key == glfw.KEY_SPACE and action == glfw.PRESS:
            composer.paused = not composer.paused
        elif key == glfw.KEY_V and action == glfw.PRESS:
            swarm.stimulus = not swarm.stimulus
        elif key == glfw.KEY_R and action == glfw.PRESS:
            swarm.reset(stimulus=swarm.stimulus)
        elif key == glfw.KEY_C and action == glfw.PRESS:
            composer.focus_pin = None if composer.focus_pin is not None else composer.focus()
        elif key == glfw.KEY_LEFT:
            composer.cam.azimuth -= 4.0
        elif key == glfw.KEY_RIGHT:
            composer.cam.azimuth += 4.0
        elif key == glfw.KEY_UP:
            composer.cam.elevation = min(-8.0, composer.cam.elevation + 3.0)
        elif key == glfw.KEY_DOWN:
            composer.cam.elevation = max(-80.0, composer.cam.elevation - 3.0)
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            composer.speed = max(0.25, composer.speed / 1.25)
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            composer.speed = min(8.0, composer.speed * 1.25)

    glfw.set_key_callback(window, on_key)
    print("Telotaxis live. Space pauses, V toggles vision, R resets, C pins the brain, arrows orbit, esc quits.")
    try:
        while not glfw.window_should_close(window):
            now = time.perf_counter()
            dt = min(0.05, now - state["last"])
            state["last"] = now
            if not composer.paused:
                state["acc"] += dt * composer.speed
                stepped = 0
                while state["acc"] >= 0.015 and stepped < 2:
                    swarm.step_interval()
                    state["acc"] -= 0.015
                    stepped += 1
            rgb = composer.frame()
            height, width, _ = rgb.shape
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb)
            fbw, fbh = glfw.get_framebuffer_size(window)
            glViewport(0, 0, fbw, fbh)
            glEnable(GL_TEXTURE_2D)
            glBegin(GL_QUADS)
            glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, -1.0)
            glTexCoord2f(1.0, 1.0); glVertex2f(1.0, -1.0)
            glTexCoord2f(1.0, 0.0); glVertex2f(1.0, 1.0)
            glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, 1.0)
            glEnd()
            glfw.swap_buffers(window)
            glfw.poll_events()
    except KeyboardInterrupt:
        pass
    finally:
        composer.close()
        glfw.destroy_window(window)
        glfw.terminate()


def _save(path: str, frames: list, fps: int) -> None:
    import imageio.v2 as imageio

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        imageio.imwrite(destination, frames[-1])
        return
    if destination.suffix.lower() == ".gif":
        imageio.mimwrite(destination, frames, fps=fps, loop=0)
        return
    imageio.mimwrite(destination, frames, fps=fps, codec="libx264", quality=8, macro_block_size=1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="telotaxis", description="MaleCNS connectome, driving vehicles")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(command):
        command.add_argument("--cars", type=int, default=12)
        command.add_argument("--seed", type=int, default=1)
        command.add_argument("--body", choices=("car", "robot"), default="car")
        command.add_argument("--variant", choices=("exact", "absent", "both"), default="exact")

    live = sub.add_parser("live")
    common(live)
    live.set_defaults(func=cmd_live, variant="exact")
    run = sub.add_parser("run")
    common(run)
    run.add_argument("--seconds", type=float, default=12.0)
    run.add_argument("--quiet", action="store_true")
    run.set_defaults(func=cmd_run, variant="both")
    frame = sub.add_parser("frame")
    common(frame)
    frame.add_argument("--at", type=float, default=4.0)
    frame.add_argument("--out", default="docs/media/frame.png")
    frame.set_defaults(func=cmd_frame)
    record = sub.add_parser("record")
    common(record)
    record.add_argument("--seconds", type=float, default=8.0)
    record.add_argument("--fps", type=int, default=8)
    record.add_argument("--out", default="docs/media/telotaxis.mp4")
    record.set_defaults(func=cmd_record)
    args = parser.parse_args(argv)
    if args.cars < 1 or args.cars > 24:
        parser.error("--cars must be from 1 to 24")
    args.func(args)
