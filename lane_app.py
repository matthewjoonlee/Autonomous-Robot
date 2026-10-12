"""Touchscreen lane following: live camera view with START / STOP / QUIT buttons.

    python lane_app.py --execute

The screen shows what the camera sees, with the detected lane (green) and the
calibrated center (cyan). Press START: the robot holds still ~0.5 s to measure
where it is, then centers on the lane and drives straight until STOP, the leg
distance (15 ft), or a fault (lane lost 3 s, 45 deg rotation). It then stops and
waits for the next START. Each run is saved like any other leg for replay.
Motor-free unless --execute.
"""

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import signal
import sys
import time

import cv2
import numpy as np

from calibrate_motors import bounded_int
from calibrate_track import choose_front
from quiz3_autonomous import (RECOVERABLE, TUNABLE, Settings, configure_motion_exposure, leg_timing, run_leg,
                              tuned)
from test_imu import positive_number

GREEN, CYAN, RED, WHITE = (0, 220, 0), (255, 255, 0), (0, 0, 255), (255, 255, 255)
IDLE_BUTTONS = [
    {"handle": "start", "text": "START", "bg_color": (40, 150, 70), "text_color": (255, 255, 255)},
    {"handle": "quit", "text": "QUIT", "bg_color": (80, 80, 80), "text_color": (255, 255, 255)},
]
DRIVING_BUTTONS = [
    {"handle": "stop", "text": "STOP", "bg_color": (200, 50, 50), "text_color": (255, 255, 255)},
]


def render(frame, detection, reference, lines=(), warning=None):
    """Camera frame with the detected lane, the centered reference, and text."""
    from track_vision import ROI_BOTTOM, ROI_TOP
    view = frame.copy()
    height, width = view.shape[:2]
    for fit, color in ((reference["fit"], CYAN), (detection["fit"] if detection else None, GREEN)):
        if fit is not None:
            ends = [(int((fit[0] * y + fit[1]) * width), int(y * height)) for y in (ROI_TOP, ROI_BOTTOM)]
            cv2.line(view, ends[0], ends[1], color, 3)
    for index, line in enumerate(lines):
        cv2.putText(view, line, (10, 30 + 30 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(view, line, (10, 30 + 30 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
    if warning:
        cv2.rectangle(view, (0, height - 50), (width, height), RED, -1)
        cv2.putText(view, warning, (10, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2, cv2.LINE_AA)
    return view


class MotionWatch:
    """Warns when the robot is commanded to drive but neither image nor heading changes.

    In one session every leg 'drove' for up to 23 s with an unchanging image: the
    motors were not moving the robot (battery/power), which looked like a freeze.
    """

    def __init__(self, seconds=2.0, pixel_change=2.0, heading_change=0.3):
        self.seconds, self.pixel_change, self.heading_change = seconds, pixel_change, heading_change
        self.since = None
        self.thumb = None
        self.heading = None

    def update(self, frame, heading, driving, now):
        thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 24)).astype(np.float32)
        if not driving:
            self.since = None
            return False
        if self.since is None:
            self.since, self.thumb, self.heading = now, thumb, heading
            return False
        if (float(np.mean(np.abs(thumb - self.thumb))) > self.pixel_change
                or abs(heading - self.heading) > self.heading_change):
            self.since, self.thumb, self.heading = now, thumb, heading  # Moving: restart the window.
            return False
        return now - self.since >= self.seconds


class LaneApp:
    def __init__(self, display, camera, imu, saber, settings, calibration, output_root, leg_ft=15.0,
                 speed_cm_s=14.0, record_every=3, execute=False, clock=time.monotonic, sleep=time.sleep):
        from lane_center import reference as center_reference
        self.display, self.camera, self.imu, self.saber = display, camera, imu, saber
        self.settings, self.calibration, self.output_root = settings, calibration, output_root
        self.leg_ft, self.speed_cm_s, self.record_every, self.execute = leg_ft, speed_cm_s, record_every, execute
        self.clock, self.sleep = clock, sleep
        self.reference = center_reference(calibration)
        self.pending = set()
        self.legs = 0
        self.message = "Place the robot on the lane and press START."
        self.watch = MotionWatch()
        self.frame_count = 0

    def poll(self):
        self.pending.update(self.display.poll_events())
        return self.pending

    def status(self, text, color=(0, 0, 0)):
        self.display.set_status(text, position="top", bg_color=color, text_color=(255, 255, 255))

    def run(self):
        """Idle until START; drive; repeat. Returns the number of runs when QUIT is pressed."""
        self.display.set_buttons(IDLE_BUTTONS, position="bottom")
        self.status(self.message)
        while True:
            events = self.poll()
            if "quit" in events:
                return self.legs
            if "start" in events:
                self.pending.clear()
                self.drive()
                if "quit" in self.pending:
                    return self.legs
                self.pending.clear()
                self.display.set_buttons(IDLE_BUTTONS, position="bottom")
                self.status(self.message)
                continue
            self.show_idle_frame()
            self.sleep(0.03)

    def show_idle_frame(self):
        from track_vision import detect_lane
        frame = cv2.cvtColor(self.camera.capture_array(), cv2.COLOR_BGRA2BGR)
        detection = detect_lane(frame)
        seen = detection["status"] == "tracked" and detection["confidence"] >= self.settings.min_confidence
        line = (f"Lane seen ({detection['confidence']:.2f}) - ready" if seen else "No lane in view")
        self.display.set_frame(render(frame, detection, self.reference, [line]))

    def drive(self):
        self.legs += 1
        self.display.set_buttons(DRIVING_BUTTONS, position="bottom")
        self.status(f"Run {self.legs}: measuring position (hold still)...", (30, 60, 140))
        self.watch = MotionWatch()
        self.frame_count = 0
        try:
            reason, _ = run_leg(self.legs, self.camera, self.imu, self.saber, self.settings, self.calibration,
                                self.output_root, self.leg_ft, self.speed_cm_s, self.record_every, self.execute,
                                self.stop_requested, self.clock, self.sleep, observer=self.observe)
        except RuntimeError as exc:  # Non-recoverable (IMU, timing): show it and stay stopped.
            reason = str(exc)
        if self.saber is not None:
            self.saber.stop()
        done = {"estimated_distance": f"reached {self.leg_ft:g} ft", "finish_marker": "stopped"}.get(reason, reason)
        self.message = f"Run {self.legs}: {done}. Press START to run again."
        if not (reason in ("estimated_distance", "finish_marker") or reason.startswith(RECOVERABLE)):
            self.message = f"Run {self.legs} stopped: {reason}"

    def stop_requested(self):
        return bool(self.poll() & {"stop", "quit"})

    def observe(self, frame, detection, row):
        """Per-frame display update while driving (every 2nd frame to save time)."""
        self.frame_count += 1
        not_moving = self.watch.update(frame, row.get("heading_deg") or 0.0, row.get("speed", 0) > 0, self.clock())
        if self.frame_count % 2 and not not_moving:
            return
        lateral = row.get("lateral_error")
        lines = [f"{row['state']}  turn {row['physical_turn']:+d}  speed {row['speed']}",
                 f"offset {lateral * 640:+.0f} px  heading {row.get('heading_deg') or 0:+.1f} deg"
                 if lateral is not None else "",
                 f"distance ~{(row.get('distance_cm') or 0) / 30.48:.1f} ft"]
        warning = "NOT MOVING - check motor power / battery" if not_moving else None
        self.display.set_frame(render(frame, detection, self.reference, lines, warning))
        if self.frame_count % 20 == 0 or not_moving:
            self.status(f"Run {self.legs}: driving - press STOP to stop", (200, 50, 50) if not_moving else (0, 90, 0))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="enable physical motor commands")
    parser.add_argument("--leg-ft", type=positive_number, default=15.0, help="maximum distance per run (<= 20 ft)")
    parser.add_argument("--estimated-speed-cm-s", type=positive_number, default=14.0)
    parser.add_argument("--max-turn", type=bounded_int(1, 20), default=20)
    parser.add_argument("--camera", type=int)
    parser.add_argument("--exposure-us", type=bounded_int(0, 20000), default=5000)
    parser.add_argument("--record-every", type=bounded_int(0, 10), default=3)
    parser.add_argument("--keep-searching", action="store_true", help="never end a run for losing the lane")
    parser.add_argument("--windowed", action="store_true", help="window instead of fullscreen (desktop testing)")
    parser.add_argument("--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
                        help=f"tuning overrides; keys: {', '.join(TUNABLE)}")
    parser.add_argument("--output", type=Path, default=Path("centering_results"))
    args = parser.parse_args(argv)
    if args.leg_ft > 20:
        parser.error("--leg-ft must be <= 20")
    try:
        settings = tuned(Settings(max_turn=args.max_turn), args.set)
    except ValueError as exc:
        parser.error(str(exc))
    if args.keep_searching:
        settings = replace(settings, lost_limit=math.inf)

    display = camera = imu = saber = None
    started = False
    handlers = {}

    def terminate(_signal, _frame):
        raise KeyboardInterrupt

    for signum in (signal.SIGTERM, signal.SIGHUP):
        handlers[signum] = signal.signal(signum, terminate)
    try:
        from display import RobotDisplay
        from picamera2 import Picamera2
        from imu_controller import IMUDevice
        from lane_center import load as load_center

        display = RobotDisplay(fullscreen=not args.windowed, fps=30, size=(800, 480), bg_color=(0, 0, 0))
        display.set_status("Starting camera and sensors...", position="top", bg_color=(0, 0, 0),
                           text_color=(255, 255, 255))
        infos = Picamera2.global_camera_info()
        camera = Picamera2(args.camera if args.camera is not None else choose_front(infos))
        camera.configure(camera.create_video_configuration(
            main={"size": (640, 480), "format": "XRGB8888"}, queue=False))
        camera.start()
        started = True
        imu = IMUDevice()
        time.sleep(1.5)
        metadata = camera.capture_metadata()
        if "ColourGains" in metadata:
            camera.set_controls({"AwbEnable": False, "ColourGains": metadata["ColourGains"]})
        print("Camera exposure:", json.dumps(configure_motion_exposure(camera, args.exposure_us), default=str))
        if args.execute:
            from sabertooth import Sabertooth
            saber = Sabertooth()
            if saber.ser is None or not saber.ser.is_open:
                raise RuntimeError("Motor serial port unavailable")
            saber.stop()
            saber.set_auto_stop(200)
            saber.set_ramping(8)
        nominal, _ = leg_timing(args.leg_ft, args.estimated_speed_cm_s)
        print(f"{'POWERED' if args.execute else 'MOTOR-FREE'}: use the touchscreen. START drives up to "
              f"{args.leg_ft:g} ft (~{nominal:.0f} s); STOP stops; QUIT exits. Ctrl+C also stops.")
        app = LaneApp(display, camera, imu, saber, settings, load_center(), args.output, args.leg_ft,
                      args.estimated_speed_cm_s, args.record_every, args.execute)
        runs = app.run()
        print(f"Quit after {runs} run(s); motors stopped.")
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted; stop requested.")
        return 2
    except Exception as exc:
        print(f"Lane app stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        actions = ([saber.stop, saber.close] if saber is not None else [])
        actions += ([imu.close] if imu is not None else [])
        if camera is not None:
            actions += ([camera.stop] if started else []) + [camera.close]
        actions += ([display.close] if display is not None else [])
        for action in actions:
            try:
                action()
            except Exception as exc:
                print(f"Cleanup warning: {exc}", file=sys.stderr)
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    return 0


if __name__ == "__main__":
    sys.exit(main())
