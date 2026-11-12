"""Quiz 3 main controller: Intersection Detection and Sign Reading modes.

    python quiz3_app.py              # everything except motors (display + sound + location)
    python quiz3_app.py --execute    # motors on

PS5 controller (touchscreen buttons do the same):
  Square     Intersection Detection mode
  Triangle   Sign Reading mode
  Cross      start the intersection run (robot drives forward by itself)
  Circle     stop: motors stop, run ends (press again to clear)
  Left stick reposition the robot between runs (not during a run)
  Options    quit

Intersection run: the robot creeps forward holding its IMU heading. Every time the
location service (separate process, location_service.py) reports a new phase, the
robot stops, announces the phase, shows it on the display, and waits --pause-s for
the judges, then continues. After "Continue travel" it drives --finish-s more and
stops. It stops on its own if location results stop arriving, the heading turns
more than --max-heading degrees, or --max-run-s passes. The controller makes every
decision; the workers only report what they see.

--record saves front/side video and a per-frame log in quiz3_runs/ for replay.
"""

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np

from capture_quiz3 import HUNG, grab, open_side, read_heading, with_timeout
from location_service import LABELS, PHASES, LocationWorker
from record_quiz3 import CIRCLE, CROSS, SQUARE, TRIANGLE, Pad, drive_command, hold_turn, open_writer

OPTIONS = 9
MODE_TEXT = {"idle": "Choose a mode: Square = Intersection, Triangle = Sign Reading",
             "intersection": "INTERSECTION DETECTION", "sign": "SIGN READING"}
SPOKEN_MODE = {"intersection": "Intersection detection mode", "sign": "Sign reading mode"}
BUTTONS = [
    {"handle": "intersection", "text": "INTERSECT", "bg_color": (30, 80, 200), "text_color": (255, 255, 255)},
    {"handle": "sign", "text": "SIGNS", "bg_color": (120, 60, 160), "text_color": (255, 255, 255)},
    {"handle": "start", "text": "START", "bg_color": (40, 150, 70), "text_color": (255, 255, 255)},
    {"handle": "stop", "text": "STOP", "bg_color": (200, 60, 60), "text_color": (255, 255, 255)},
]
EXTRA_PHRASES = ["Run complete", "Stopped", "Location lost. Stopping.", "Ready",
                 "Sign reading is not ready yet"]


class App:
    def __init__(self, args, display, pad, front, side, imu, saber, announcer, worker, recorder):
        self.args, self.display, self.pad = args, display, pad
        self.front, self.side, self.imu, self.saber = front, side, imu, saber
        self.say, self.worker, self.recorder = announcer.say, worker, recorder
        self.mode, self.state = "idle", "idle"
        self.quit = False
        self.frame_id = 0
        self.frames = {"front": None, "side": None}
        self.result = None           # newest LocationResult
        self.result_time = 0.0
        self.heading = None          # relative to program start
        self.heading_zero = read_heading(imu)
        self.held_heading = None
        self.command = (0, 0)
        self.last_motor = 0.0
        self.pad_state = (set(), set(), (0.0, 0.0))
        self.message = "Choose a mode"
        # run state
        self.run_frame0 = 0
        self.announced = -1
        self.pending = []
        self.phase = None
        self.until = 0.0
        self.run_start = 0.0

    # ------------------------------------------------------------ inputs
    def handle_inputs(self, now):
        pressed, held, stick = self.pad.poll() if self.pad else (set(), set(), (0.0, 0.0))
        self.pad_state = (pressed, held, stick)
        events = set(self.display.poll_events())
        if SQUARE in pressed or "intersection" in events:
            self.set_mode("intersection")
        if TRIANGLE in pressed or "sign" in events:
            self.set_mode("sign")
        if CROSS in pressed or "start" in events:
            self.start_run(now)
        if CIRCLE in pressed or "stop" in events:
            self.stop_run("Stopped")
        if OPTIONS in pressed:
            self.quit = True

    def set_mode(self, mode):
        if self.state in ("driving", "judging", "finishing"):
            self.stop_run("Stopped")
        self.mode = mode
        self.state = "ready" if mode == "intersection" else "signs"
        self.phase = None
        self.message = ("Press Cross to start" if mode == "intersection"
                        else "Show a sign to the front camera")
        self.say(SPOKEN_MODE[mode])
        self.display.set_status(MODE_TEXT[mode], position="top", bg_color=(0, 0, 0),
                                text_color=(255, 255, 0))

    # ------------------------------------------------------------ intersection run
    def start_run(self, now):
        if self.mode != "intersection" or self.state in ("driving", "judging", "finishing"):
            return
        self.worker.reset()
        self.run_frame0 = self.frame_id + 1
        self.announced, self.pending, self.phase = -1, [], None
        self.held_heading = self.heading
        self.run_start = self.result_time = now
        self.state = "driving"
        self.message = "Driving"

    def stop_run(self, reason):
        if self.state in ("driving", "judging", "finishing"):
            self.say(reason)
            self.message = reason
        self.state = "ready" if self.mode == "intersection" else self.state
        self.motors_stop()

    def update_run(self, now):
        result = self.worker.latest()
        if result is not None:
            if "error" in result:
                print("Location worker error:", result["error"])
            elif result["frame_id"] >= self.run_frame0:
                self.result, self.result_time = result, now
                if self.state in ("driving", "judging") and result["zone"] is not None:
                    index = PHASES.index(result["zone"])
                    # Report every phase in order, even if two arrive together.
                    while self.announced < index:
                        self.announced += 1
                        self.pending.append(self.announced)
            elif self.state not in ("driving", "judging", "finishing"):
                self.result = result  # show live location while idle too
        if self.state == "driving":
            if self.pending:
                self.begin_phase(self.pending.pop(0), now)
            elif now - self.result_time > self.args.lost_s:
                self.stop_run("Location lost. Stopping.")
            elif now - self.run_start > self.args.max_run_s:
                self.stop_run("Stopped")
            elif self.heading_error() is not None and abs(self.heading_error()) > self.args.max_heading:
                self.stop_run("Stopped")
        elif self.state == "judging" and now >= self.until:
            if self.pending:
                self.begin_phase(self.pending.pop(0), now)
            elif self.phase == PHASES[-1]:
                self.state, self.until = "finishing", now + self.args.finish_s
                self.result_time = now
            else:
                self.state = "driving"
                self.result_time = now
        elif self.state == "finishing" and now >= self.until:
            self.state = "ready"
            self.motors_stop()
            self.say("Run complete")
            self.message = "Run complete - Cross to run again"

    def begin_phase(self, index, now):
        self.phase = PHASES[index]
        self.state, self.until = "judging", now + self.args.pause_s
        self.motors_stop()
        self.say(LABELS[self.phase])
        self.message = LABELS[self.phase]
        print(f"{now - self.run_start:6.1f} s  phase: {LABELS[self.phase]}"
              + (f"  ({self.result['distance_in']:+.1f} in)" if self.result and self.result.get("distance_in") is not None else ""))

    def heading_error(self):
        if self.heading is None or self.held_heading is None:
            return None
        return (self.heading - self.held_heading + 180) % 360 - 180

    # ------------------------------------------------------------ motors
    def motors_stop(self):
        self.command = (0, 0)
        if self.saber is not None:
            self.saber.stop()

    def update_motors(self, now):
        if self.saber is None or now - self.last_motor < 0.04:
            return
        self.last_motor = now
        if self.state in ("driving", "finishing"):
            error = self.heading_error()
            turn = hold_turn(error, self.args.max_turn) if error is not None else 0
            command = (self.args.speed, turn)
        elif self.state in ("judging",):
            command = (0, 0)
        else:  # between runs: left stick repositions (R1 creeps)
            _, held, stick = self.pad_state
            command = drive_command(held, stick, self.args.speed, self.args.max_turn, 45)
        self.command = command
        if command == (0, 0):
            self.saber.stop()
        else:
            self.saber.drive(*command)

    # ------------------------------------------------------------ frames + display
    def capture(self, now):
        from camera_setup import front_bgr, side_bgr
        try:
            self.frames["front"] = with_timeout(lambda: grab(self.front, front_bgr)[0], 3)
        except TimeoutError:
            HUNG.append(self.front)
            self.stop_run("Stopped")
            print("FRONT camera stopped responding; quitting.")
            self.quit = True
            return
        if self.side is not None:
            try:
                self.frames["side"] = with_timeout(lambda: grab(self.side, side_bgr)[0], 3)
            except Exception as exc:
                if isinstance(exc, TimeoutError):
                    HUNG.append(self.side)
                print(f"SIDE camera stopped responding ({exc}); check its USB cable.")
                self.side, self.frames["side"] = None, None
                self.stop_run("Stopped")
        if self.imu is not None:
            try:
                value = with_timeout(lambda: read_heading(self.imu, tries=1), 0.5)
                if value is not None and self.heading_zero is not None:
                    from test_imu import signed_delta
                    self.heading = signed_delta(value, self.heading_zero, self.imu.SIGN)
            except TimeoutError:
                HUNG.append(self.imu)
                self.imu = None
        self.frame_id += 1
        if self.mode == "intersection" and self.frames["side"] is not None:
            self.worker.submit(self.frame_id, now, self.frames["front"], self.frames["side"])
        if self.recorder:
            self.recorder.write(self, now)

    def show(self):
        import cv2
        from annotations import draw, label
        front, side = self.frames["front"], self.frames["side"]
        canvas = np.zeros((400, 800, 3), dtype=np.uint8)
        annotations = (self.result or {}).get("annotations", []) if self.mode == "intersection" else []
        if front is not None:
            canvas[0:281, 0:500] = cv2.resize(draw(front.copy(), annotations, "front"), (500, 281))
        if side is not None:
            canvas[0:225, 500:800] = cv2.resize(draw(side.copy(), annotations, "side"), (300, 225))
        label(canvas, "FRONT", (8, 275), (255, 255, 255), 0.5)
        label(canvas, "SIDE", (508, 220), (255, 255, 255), 0.5)
        mode = {"intersection": "INTERSECTION DETECTION", "sign": "SIGN READING"}.get(self.mode, "NO MODE")
        label(canvas, mode, (508, 255), (0, 255, 255), 0.6)
        if self.mode == "intersection":
            phase = LABELS[self.phase] if self.phase else "-"
            label(canvas, f"PHASE: {phase}", (10, 330), (0, 255, 0), 1.1)
            loc = self.result
            if loc and loc.get("distance_in") is not None:
                label(canvas, f"location {loc['label']}  {loc['distance_in']:+.0f} in", (508, 285),
                      (255, 255, 255), 0.5)
        label(canvas, self.message, (10, 380), (255, 255, 255), 0.7)
        if self.heading is not None:
            label(canvas, f"heading {self.heading:+.1f}  motors {self.command[0]:+d}/{self.command[1]:+d}",
                  (508, 310), (200, 200, 200), 0.45)
        self.display.set_frame(canvas)

    def run(self):
        next_frame = time.monotonic()
        while not self.quit:
            now = time.monotonic()
            self.handle_inputs(now)
            self.update_run(now)
            self.update_motors(now)
            if now >= next_frame:
                next_frame = max(next_frame + 1 / self.args.fps, now)
                self.capture(now)
                self.show()
            else:
                time.sleep(0.005)


class Recorder:
    """Front/side video plus a per-frame log, for replay_location-style review."""

    def __init__(self, cv2, root):
        from camera_setup import FRONT_SIZE, SIDE_SIZE
        self.folder = root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.folder.mkdir(parents=True)
        self.front = open_writer(cv2, self.folder / "front", 10, FRONT_SIZE)
        self.side = open_writer(cv2, self.folder / "side", 10, SIDE_SIZE)
        self.blank = np.zeros((SIDE_SIZE[1], SIDE_SIZE[0], 3), dtype=np.uint8)
        self.file = (self.folder / "frames.csv").open("x", newline="")
        self.log = csv.writer(self.file)
        self.log.writerow(["frame", "time", "mode", "state", "phase", "location_zone",
                           "distance_in", "heading_rel_deg", "speed", "turn"])
        self.start = time.monotonic()

    def write(self, app, now):
        self.front.write(app.frames["front"])
        side = app.frames["side"]
        self.side.write(side if side is not None else self.blank)
        loc = app.result or {}
        self.log.writerow([app.frame_id, f"{now - self.start:.3f}", app.mode, app.state, app.phase,
                           loc.get("zone"), loc.get("distance_in"), app.heading, *app.command])

    def close(self):
        self.front.release()
        self.side.release()
        self.file.close()
        print(f"Recording saved in {self.folder}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="enable motors")
    parser.add_argument("--speed", type=int, default=22, help="forward speed during the run")
    parser.add_argument("--max-turn", type=int, default=15)
    parser.add_argument("--pause-s", type=float, default=1.5, help="judging pause at each phase")
    parser.add_argument("--finish-s", type=float, default=1.5, help="drive after Continue travel")
    parser.add_argument("--lost-s", type=float, default=2.0, help="stop if no location for this long")
    parser.add_argument("--max-run-s", type=float, default=60.0)
    parser.add_argument("--max-heading", type=float, default=30.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--volume", type=float, default=1.0)
    parser.add_argument("--record", action="store_true", help="save video + log to quiz3_runs/")
    parser.add_argument("--windowed", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.speed <= 40 or not 0 <= args.max_turn <= 40:
        parser.error("--speed and --max-turn must be 0-40")

    front = side = imu = saber = pad = display = announcer = worker = recorder = None
    started = []
    previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}

    def terminate(_signal, _frame):
        raise KeyboardInterrupt

    for s in previous:
        signal.signal(s, terminate)
    code = 0
    try:
        import cv2
        from picamera2 import Picamera2
        from announcer import Announcer
        from calibrate_track import choose_front
        from camera_setup import front_config, side_config
        from capture_side_camera import choose_side
        from display import RobotDisplay

        worker = LocationWorker()  # start early: loading in the other process overlaps setup
        display = RobotDisplay(fullscreen=not args.windowed, fps=30, size=(800, 480), bg_color=(0, 0, 0))
        display.set_status("Starting...", position="top", bg_color=(0, 0, 0), text_color=(255, 255, 255))
        display.set_buttons(BUTTONS, position="bottom")
        announcer = Announcer(list(LABELS.values()) + list(SPOKEN_MODE.values()) + EXTRA_PHRASES,
                              volume=args.volume)
        try:
            pad = Pad(wait_s=20)
            print(f"Controller: {pad.name}")
        except Exception as exc:
            print(f"WARNING: no PS controller ({exc}); touchscreen buttons only.")
        infos = Picamera2.global_camera_info()
        front = Picamera2(choose_front(infos))
        front.configure(front_config(front))
        front.start()
        started.append(front)
        try:
            side = with_timeout(lambda: open_side(Picamera2, side_config, choose_side), 10)
            started.append(side)
        except Exception as exc:
            print(f"WARNING: side camera unavailable ({exc}); intersection detection needs it.")
        try:
            from imu_controller import IMUDevice
            imu = IMUDevice()
        except Exception as exc:
            print(f"WARNING: IMU unavailable ({exc}); no heading hold.")
        time.sleep(2)
        metadata = front.capture_metadata()
        if "ColourGains" in metadata:  # same white-balance lock as the other programs
            front.set_controls({"AwbEnable": False, "ColourGains": metadata["ColourGains"]})
        if args.execute:
            from sabertooth import Sabertooth
            saber = Sabertooth()
            if saber.ser is None or not saber.ser.is_open:
                raise RuntimeError("Motor serial port unavailable")
            saber.stop()
            saber.set_auto_stop(200)  # motors stop by themselves if commands stop
            saber.set_ramping(8)
        if args.record:
            recorder = Recorder(cv2, Path("quiz3_runs"))
        display.set_status(MODE_TEXT["idle"], position="top", bg_color=(0, 0, 0), text_color=(255, 255, 255))
        announcer.say("Ready")
        print("Ready." + ("" if saber else " Motors are OFF (add --execute)."))
        App(args, display, pad, front, side, imu, saber, announcer, worker, recorder).run()
    except KeyboardInterrupt:
        print("\nStopping.")
    except Exception as exc:
        print(f"quiz3_app failed: {exc!r}", file=sys.stderr)
        code = 1
    finally:
        if saber is not None:  # motors first
            for action in (saber.stop, saber.close):
                try:
                    action()
                except Exception as exc:
                    print(f"Motor cleanup warning: {exc}", file=sys.stderr)
        if recorder is not None:
            recorder.close()
        for camera in started:
            if not any(camera is hung for hung in HUNG):
                try:
                    with_timeout(camera.stop, 5)
                    with_timeout(camera.close, 5)
                except Exception:
                    pass
        if imu is not None and not any(imu is hung for hung in HUNG):
            try:
                with_timeout(imu.close, 5)
            except Exception:
                pass
        for closer in (worker, display):
            if closer is not None:
                try:
                    closer.close()
                except Exception:
                    pass
        if pad is not None:
            pad.close()
        if announcer is not None:
            try:
                announcer.close()
            except Exception:
                pass
        for s, handler in previous.items():
            signal.signal(s, handler)
    return code


if __name__ == "__main__":
    code = main()
    if HUNG:
        sys.stdout.flush()
        os._exit(code)
    sys.exit(code)
