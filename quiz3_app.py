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

Sign Reading: the robot stays still. Every sign the sign service (separate process,
sign_service.py) reports steadily gets a box and label on the display and is
announced once; showing a sign again after taking it away announces it again.

Zones come from one running front position (PositionFuser): it advances by driving
time x speed (stops do not count), and while the front is more than 20 in out each
camera reading nudges it; misreads are ignored, and a run of agreeing readings
resets it (bad start, robot slowed or blocked). The speed is measured from those
readings (default 4.0 in/s). Inside 20 in only driving distance is used: there the
lab camera estimates misread or stuck at calibration-stop values.
--camera-zones uses the camera zone tracker for every zone instead.

Intersection run: before the pause zone the robot centers itself on the lane with
the lane-centering controller from quiz3_autonomous.py (it may be placed off-center
or angled). The lane detector and its calibration (lane_center.json) were made on
the old cropped camera view, which is exactly the middle of the full view
(LEGACY_CROP), so it reads that region enlarged - no recalibration. Where the lane
is not visible (cross road filling the view) and from the pause zone on, it holds
the track direction with the IMU. Every time the
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
from location_service import LABELS, PHASES, LocationWorker, PositionFuser, zone_index
from record_quiz3 import CIRCLE, CROSS, SQUARE, TRIANGLE, Pad, drive_command, hold_turn, open_writer
from sign_service import SPOKEN as SIGN_WORDS, SignWorker

OPTIONS = 9
# The old 640x480 front view used sensor area (1152, 432, 2304, 1728) of 4608x2592
# (fov_check report). In the full 640x360 frame that is x 160-480, y 60-300.
LEGACY_CROP = (160, 60, 320, 240)


def legacy_view(front):
    """Middle of the full view, enlarged to the old 640x480 lane-centering view."""
    import cv2
    x, y, w, h = LEGACY_CROP
    return cv2.resize(front[y:y + h, x:x + w], (640, 480), interpolation=cv2.INTER_LINEAR)
MODE_TEXT = {"idle": "Choose a mode: Square = Intersection, Triangle = Sign Reading",
             "intersection": "INTERSECTION DETECTION", "sign": "SIGN READING"}
SPOKEN_MODE = {"intersection": "Intersection detection mode", "sign": "Sign reading mode"}
BUTTONS = [
    {"handle": "intersection", "text": "INTERSECT", "bg_color": (30, 80, 200), "text_color": (255, 255, 255)},
    {"handle": "sign", "text": "SIGNS", "bg_color": (120, 60, 160), "text_color": (255, 255, 255)},
    {"handle": "start", "text": "START", "bg_color": (40, 150, 70), "text_color": (255, 255, 255)},
    {"handle": "stop", "text": "STOP", "bg_color": (200, 60, 60), "text_color": (255, 255, 255)},
]
EXTRA_PHRASES = ["Run complete", "Stopped", "Location lost. Stopping.", "Ready"]


class App:
    def __init__(self, args, display, pad, front, side, imu, saber, announcer, worker, recorder, signs):
        self.args, self.display, self.pad = args, display, pad
        self.front, self.side, self.imu, self.saber = front, side, imu, saber
        self.say, self.worker, self.recorder = announcer.say, worker, recorder
        self.signs = signs
        self.sign_result = None
        self.sign_announced = None
        # lane centering
        self.lane = None             # quiz3_autonomous.Controller during a run
        self.lane_calibration = None
        if not args.no_lane:
            from lane_center import load
            self.lane_calibration = load()
        self.track_zero = None       # app heading that points along the track
        self.lane_command = None
        self.lane_status = ""
        self.recent = []             # last few (front frame, heading) for the start measurement
        self.offset = getattr(args, "zone_offset", 0.0)
        self.drive_time = 0.0        # seconds the motors drove forward during this run
        self.last_loop = None
        self.fuser = None            # PositionFuser during a run (None with --camera-zones)
        self.speed = args.drive_speed
        self.odo_front = None
        self.mode, self.state = "idle", "idle"
        self.quit = False
        self.frame_id = 0
        self.frames = {"front": None, "side": None}
        self.result = None           # newest LocationResult
        self.result_time = 0.0
        self.familiar_time = 0.0
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
        if mode == "sign":
            self.signs.reset()
            self.sign_result, self.sign_announced = None, None
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
        self.start_lane(now)
        self.run_start = self.result_time = self.familiar_time = now
        self.drive_time, self.odo_front = 0.0, None
        self.fuser = None if self.args.camera_zones else PositionFuser(
            speed=self.args.drive_speed, measure_speed=self.args.measure_speed)
        self.speed = self.args.drive_speed
        self.state = "driving"
        self.message = "Driving"

    def start_lane(self, now):
        """Find the robot's angle to the lane from the last frames; set up the controller."""
        self.lane, self.lane_command = None, None
        if self.lane_calibration is None or self.heading is None:
            return
        import statistics
        from dataclasses import replace
        from lane_center import image_yaw, reference
        from quiz3_autonomous import Controller, Settings
        from track_vision import detect_lane
        settings = replace(Settings(), speed=self.args.speed, lost_limit=1e9)
        yaws = []
        for frame, heading in self.recent:
            detection = detect_lane(legacy_view(frame))
            if detection["status"] == "tracked" and detection["confidence"] >= settings.min_confidence:
                yaws.append(image_yaw(detection["fit"], self.lane_calibration, settings.rotation_error_per_deg)
                            - (heading - self.heading))
        yaw = statistics.median(yaws) if len(yaws) >= 2 else 0.0
        # Relative heading 0 = along the track; the robot starts at `yaw` to it.
        self.track_zero = self.heading - yaw
        self.held_heading = self.track_zero
        self.lane = Controller(reference(self.lane_calibration), now, settings, self.lane_calibration)
        self.lane_status = (f"start angle {yaw:+.1f} deg from lane" if yaws
                            else "lane not visible at start: holding heading")
        print("Lane:", self.lane_status)

    def lane_step(self, now):
        """One lane-centering decision on the newest front frame (driving states only)."""
        if self.lane is None or self.heading is None:
            self.lane_command = None
            return
        heading = (self.heading - self.track_zero + 180) % 360 - 180
        detection, summary = self.lane.observe(legacy_view(self.frames["front"]), heading, now)
        if self.announced >= PHASES.index("pause"):
            # Cross road from here on: lane fits are unreliable; hold the track direction.
            detection = dict(detection, status="line_lost", confidence=0.0)
        row = {}
        try:
            command = self.lane.decide(detection, summary, heading, now, row)
        except RuntimeError as exc:   # e.g. turned past the safety limit
            print("Lane controller stop:", exc)
            self.stop_run("Stopped")
            return
        self.lane_command = (command["speed"], command["raw_turn"])
        self.lane_status = (f"lane {row.get('state')}  offset {row.get('lateral_error', 0):+.3f}  "
                            f"target {row.get('target_heading_deg') or 0:+.0f} deg")

    def stop_run(self, reason):
        if self.state in ("driving", "judging", "finishing"):
            self.say(reason)
            self.message = reason
        self.state = "ready" if self.mode == "intersection" else self.state
        self.motors_stop()

    def update_signs(self):
        """Announce each new steady sign once; the robot never moves in this mode."""
        result = self.signs.latest()
        if result is None:
            return
        if "error" in result:
            print("Sign worker error:", result["error"])
            return
        self.sign_result = result
        label = result["label"]
        if label != self.sign_announced:
            self.sign_announced = label
            if label is not None and self.mode == "sign":
                self.say(SIGN_WORDS[label])
                self.message = f"Sign: {SIGN_WORDS[label]}"
                print(f"sign: {label}  box {result['box']}")

    def update_run(self, now):
        result = self.worker.latest()
        if result is not None:
            if "error" in result:
                print("Location worker error:", result["error"])
            elif result["frame_id"] >= self.run_frame0:
                self.result, self.result_time = result, now
                if result.get("familiar", True):
                    self.familiar_time = now
                if self.fuser is not None and self.state in ("driving", "judging", "finishing"):
                    reading = (result["raw_distance_in"] - self.offset
                               if result.get("familiar", True) and result.get("raw_distance_in") is not None
                               else None)
                    self.fuser.observe(reading, self.drive_time)
                    self.speed = self.fuser.speed
                if self.state in ("driving", "judging") and result["zone"] is not None and self.fuser is None:
                    index = PHASES.index(result["zone"])
                    if self.announced < 0:
                        # Start: announce only the zone the robot is in, not the ones before it.
                        self.announced = index - 1
                    # Report every phase in order, even if two arrive together.
                    while self.announced < index:
                        self.announced += 1
                        self.pending.append(self.announced)
            elif self.state not in ("driving", "judging", "finishing"):
                self.result = result  # show live location while idle too
        if self.fuser is not None and self.state in ("driving", "judging", "finishing"):
            self.odo_front = self.fuser.position(self.drive_time)
            if self.odo_front is not None:
                # Announce on the line: `zone_lead` covers the time to speak and stop.
                target = zone_index(self.odo_front - self.args.zone_lead)
                if self.announced < 0:
                    self.announced = max(0, target) - 1 if target > 0 else 0
                while self.announced < target:
                    self.announced += 1
                    self.pending.append(self.announced)
        if self.state == "driving":
            if self.pending:
                self.begin_phase(self.pending.pop(0), now)
            elif self.fuser is not None and self.odo_front is not None and self.odo_front < 20:
                # Close in, zones come from driving distance: no camera timeouts.
                if now - self.run_start > self.args.max_run_s:
                    self.stop_run("Stopped")
                elif self.heading_error() is not None and abs(self.heading_error()) > self.args.max_heading:
                    self.stop_run("Stopped")
            elif now - self.result_time > self.args.lost_s:
                self.stop_run("Location lost. Stopping.")
            elif now - self.familiar_time > self.args.unfamiliar_s:
                # Views unlike the reference for this long: the room needs calibrating.
                print("Stopping: camera views do not match the location reference "
                      "(new room? record a calibration run).")
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
        if self.phase == "approach":
            # Before the sign-reading area is not a judged zone: keep driving quietly.
            self.message = "Approaching"
            return
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
            if self.lane_command is not None:
                command = self.lane_command
            else:
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
        if self.heading is not None:
            self.recent = (self.recent + [(self.frames["front"], self.heading)])[-6:]
        if self.state in ("driving", "finishing"):
            self.lane_step(now)
        if self.mode == "intersection" and self.frames["side"] is not None:
            self.worker.submit(self.frame_id, now, self.frames["front"], self.frames["side"])
        if self.mode == "sign":
            self.signs.submit(self.frame_id, now, self.frames["front"])
        if self.recorder:
            self.recorder.write(self, now)

    def show(self):
        import cv2
        from annotations import draw, label
        front, side = self.frames["front"], self.frames["side"]
        canvas = np.zeros((400, 800, 3), dtype=np.uint8)
        if self.mode == "intersection":
            annotations = (self.result or {}).get("annotations", [])
        elif self.mode == "sign":
            annotations = (self.sign_result or {}).get("annotations", [])
        else:
            annotations = []
        if front is not None:
            canvas[0:281, 0:500] = cv2.resize(draw(front.copy(), annotations, "front"), (500, 281))
        if side is not None:
            canvas[0:225, 500:800] = cv2.resize(draw(side.copy(), annotations, "side"), (300, 225))
        label(canvas, "FRONT", (8, 275), (255, 255, 255), 0.5)
        label(canvas, "SIDE", (508, 220), (255, 255, 255), 0.5)
        mode = {"intersection": "INTERSECTION DETECTION", "sign": "SIGN READING"}.get(self.mode, "NO MODE")
        if self.mode != "sign":  # Sign Reading has its own large label below
            label(canvas, mode, (508, 255), (0, 255, 255), 0.6)
        if self.mode == "intersection":
            phase = ("Approaching" if self.phase == "approach"
                     else LABELS[self.phase] if self.phase else "-")
            label(canvas, f"PHASE: {phase}", (10, 330), (0, 255, 0), 1.1)
            if self.odo_front is not None:
                label(canvas, f"front {self.odo_front:+.1f} in (driving distance, {self.speed:.1f} in/s)",
                      (508, 345), (180, 255, 180), 0.4)
            if self.lane_status and self.lane is not None:
                label(canvas, self.lane_status, (508, 330), (200, 200, 255), 0.4)
            loc = self.result
            if loc and loc.get("distance_in") is not None:
                label(canvas, f"location {loc['label']}  {loc['distance_in']:+.0f} in", (508, 285),
                      (255, 255, 255), 0.5)
        if self.mode == "sign":
            sign = (self.sign_result or {}).get("label")
            label(canvas, f"SIGN: {SIGN_WORDS[sign].upper() if sign else '-'}", (10, 330),
                  (0, 255, 0) if sign else (160, 160, 160), 1.3)
            label(canvas, "SIGN READING", (508, 265), (0, 255, 255), 1.0)
        label(canvas, self.message, (10, 380), (255, 255, 255), 0.7)
        if self.heading is not None:
            label(canvas, f"heading {self.heading:+.1f}  motors {self.command[0]:+d}/{self.command[1]:+d}",
                  (508, 310), (200, 200, 200), 0.45)
        self.display.set_frame(canvas)

    def run(self):
        next_frame = time.monotonic()
        while not self.quit:
            now = time.monotonic()
            if (self.last_loop is not None and self.state in ("driving", "finishing")
                    and self.command[0] > 0):
                self.drive_time += now - self.last_loop
            self.last_loop = now
            self.handle_inputs(now)
            self.update_signs()
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
                           "distance_in", "heading_rel_deg", "speed", "turn", "odo_front_in",
                           "drive_time_s", "raw_distance_in"])
        self.start = time.monotonic()

    def write(self, app, now):
        self.front.write(app.frames["front"])
        side = app.frames["side"]
        self.side.write(side if side is not None else self.blank)
        loc = app.result or {}
        self.log.writerow([app.frame_id, f"{now - self.start:.3f}", app.mode, app.state, app.phase,
                           loc.get("zone"), loc.get("distance_in"), app.heading, *app.command,
                           None if app.odo_front is None else round(app.odo_front, 2),
                           round(app.drive_time, 3), loc.get("raw_distance_in")])

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
    parser.add_argument("--no-lane", action="store_true", help="heading hold only, no lane centering")
    parser.add_argument("--camera-zones", action="store_true",
                        help="use camera estimates for every zone (no driving-distance after Read the sign)")
    parser.add_argument("--drive-speed", type=float, default=4.0,
                        help="forward speed in in/s used for driving distance (Nov 15 lab: 4.0)")
    parser.add_argument("--measure-speed", action="store_true",
                        help="re-measure the speed from camera readings on the approach")
    parser.add_argument("--zone-lead", type=float, default=0.3,
                        help="announce this many inches before each line (speaking/stopping delay)")
    parser.add_argument("--zone-ref", choices=("center", "front"), default="front",
                        help="which point of the robot the zones are judged by (default front)")
    # Body 36 cm (wheels stick out 2 cm more at the back; the front is what matters).
    # The calibration recordings marked the middle of the body, 18 cm behind the front.
    parser.add_argument("--robot-length-in", type=float, default=36 / 2.54,
                        help="body length front to back, inches (default 36 cm)")
    parser.add_argument("--front-offset-in", type=float, default=None,
                        help="measured distance from the center tape mark (used in the calibration "
                             "recordings) to the front edge, inches; overrides half the body length")
    parser.add_argument("--pause-s", type=float, default=1.5, help="judging pause at each phase")
    parser.add_argument("--finish-s", type=float, default=1.5, help="drive after Continue travel")
    parser.add_argument("--lost-s", type=float, default=2.0, help="stop if no location for this long")
    parser.add_argument("--unfamiliar-s", type=float, default=10.0,
                        help="stop if the views match no reference frame for this long (~40 in)")
    parser.add_argument("--max-run-s", type=float, default=60.0)
    parser.add_argument("--max-heading", type=float, default=30.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--volume", type=float, default=1.0)
    parser.add_argument("--record", action="store_true", help="save video + log to quiz3_runs/")
    parser.add_argument("--windowed", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.speed <= 40 or not 0 <= args.max_turn <= 40:
        parser.error("--speed and --max-turn must be 0-40")
    if args.zone_ref == "front" and not (args.robot_length_in and 4 <= args.robot_length_in <= 30):
        parser.error("--zone-ref front needs --robot-length-in (front bumper to back, 4-30 in)")
    if args.front_offset_in is not None and not 0 <= args.front_offset_in <= 20:
        parser.error("--front-offset-in must be 0-20 inches")

    front = side = imu = saber = pad = display = announcer = worker = recorder = signs = None
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

        if args.zone_ref == "center":
            offset = 0.0
        elif args.front_offset_in is not None:
            offset = args.front_offset_in
        else:
            offset = args.robot_length_in / 2
        print(f"Zones judged at the robot's {args.zone_ref}"
              + (f" ({offset:.1f} in ahead of center)" if offset else ""))
        worker = LocationWorker(offset=offset)  # start early: loading overlaps setup
        args.zone_offset = offset
        signs = SignWorker()
        display = RobotDisplay(fullscreen=not args.windowed, fps=30, size=(800, 480), bg_color=(0, 0, 0))
        display.set_status("Starting...", position="top", bg_color=(0, 0, 0), text_color=(255, 255, 255))
        display.set_buttons(BUTTONS, position="bottom")
        announcer = Announcer(list(LABELS.values()) + list(SPOKEN_MODE.values()) + EXTRA_PHRASES
                              + list(SIGN_WORDS.values()),
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
        App(args, display, pad, front, side, imu, saber, announcer, worker, recorder, signs).run()
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
        for closer in (worker, signs, display):
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
