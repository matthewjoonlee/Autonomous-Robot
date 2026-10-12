"""Lane centering: one program for every test. Place the robot on the lane; it measures
where it is and drives up to 15 ft, then waits to be moved. Motor-free unless --execute."""

import argparse
import csv
from dataclasses import dataclass, replace
from datetime import datetime
import json
import math
from pathlib import Path
import select
import signal
import statistics
import sys
import time

from calibrate_motors import bounded_int, fresh_heading
from calibrate_track import choose_front
from test_imu import positive_number, signed_delta


@dataclass(frozen=True)
class Settings:
    speed: int = 25
    # Below speed 25, so mixed mode never reverses the inner wheel. The heading
    # loop only asks for more than 10 when ~3+ degrees off, i.e. during recovery.
    max_turn: int = 20
    position_gain: float = 90.0  # Degrees of target heading per normalized lateral error; provisional.
    heading_gain: float = 3.0    # Physical turn units per degree; provisional.
    max_target_heading: float = 12.0  # Approach angle when far off, e.g. a half-off-road start.
    abort_heading: float = 45.0  # Runaway guard. A nudge rotated the robot ~11 deg and was recoverable.
    min_confidence: float = 0.35
    smoothing: float = 0.5       # Weight of each new lane measurement; 1 disables filtering.
    # Near-row image error per degree of yaw: capture 005 moved the lane -272 px at +13.1 deg.
    rotation_error_per_deg: float = 0.0323
    # Per-frame weight pulling the IMU heading toward the image heading (~0.5 s time
    # constant); during edge trials the IMU drifted 2-4.5 deg from the image. 0 = IMU only.
    heading_correction: float = 0.1
    turn_deadzone: float = 4.0   # Turn commands below ~5 barely rotate the robot (course-run data).
    turn_rate: float = 40.0      # Max change in physical turn units per second.
    turn_slowdown: float = 0.0   # Speed fraction shed at full turn; 0 until lower speeds are validated.
    lost_limit: float = 3.0      # Seconds searching for an unseen lane before stopping (~42 cm).
    # End of the track: lane seen only in the near rows (none above this image row) with no
    # confident full-lane frame for `end_seconds`. Normal driving reaches row <= 0.67 (99%).
    end_top_row: float = 0.68
    end_seconds: float = 0.5
    prior_seconds: float = 0.3   # Search near the last lane fit only while it is this recent.
    max_sample_age: float = 0.15


TUNABLE = ("position_gain", "heading_gain", "max_target_heading", "min_confidence", "smoothing",
           "rotation_error_per_deg", "heading_correction", "turn_deadzone", "turn_rate", "turn_slowdown",
           "prior_seconds")


def tuned(settings, overrides):
    """Apply KEY=VALUE tuning overrides. Speed, turn, and safety limits are not tunable here."""
    changes = {}
    for item in overrides:
        key, separator, text = item.partition("=")
        if not separator or key not in TUNABLE:
            raise ValueError(f"--set takes KEY=VALUE with KEY one of: {', '.join(TUNABLE)}")
        kind = type(getattr(settings, key))
        try:
            value = kind(text)
        except ValueError:
            raise ValueError(f"--set {key} needs a {kind.__name__}") from None
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--set {key} must be a nonnegative number")
        changes[key] = value
    result = replace(settings, **changes)
    if not (0 < result.smoothing <= 1 and 0 <= result.turn_slowdown < 1 and result.heading_correction <= 1
            and 0 < result.min_confidence <= 1 and result.max_target_heading < result.abort_heading
            and result.turn_deadzone < result.max_turn and result.turn_rate > 0):
        raise ValueError("--set values out of range (check smoothing, max_target_heading, turn_deadzone)")
    return result


def configure_motion_exposure(camera, exposure_us):
    """Short shutter, compensated gain within hardware limits; log actual values."""
    metadata = camera.capture_metadata()
    if not exposure_us:
        return {"mode": "auto", "observed": {key: metadata.get(key) for key in
                ("ExposureTime", "AnalogueGain", "DigitalGain")}}
    controls = camera.camera_controls
    if not all(name in controls for name in ("ExposureTime", "AnalogueGain", "AeEnable")):
        raise RuntimeError("Camera lacks manual exposure controls; use --exposure-us 0 for auto")
    low, high = controls["ExposureTime"][:2]
    exposure = max(low, min(exposure_us, high)) if high > 0 else max(low, exposure_us)
    previous_exposure = metadata.get("ExposureTime", exposure)
    previous_gain = metadata.get("AnalogueGain", 1.0) * metadata.get("DigitalGain", 1.0)
    desired_gain = previous_gain * previous_exposure / exposure
    gain_low, gain_high = controls["AnalogueGain"][:2]
    gain = max(gain_low, min(desired_gain, gain_high, 8.0))
    requested = {"AeEnable": False, "ExposureTime": int(exposure), "AnalogueGain": gain}
    # Newer libcamera versions expose explicit manual mode controls (Manual=1).
    for mode in ("ExposureTimeMode", "AnalogueGainMode"):
        if mode in controls:
            requested[mode] = 1
    camera.set_controls(requested)
    time.sleep(0.3)
    observed = camera.capture_metadata()
    return {"mode": "manual", "requested": requested,
            "gain_limited": gain < desired_gain,
            "observed": {key: observed.get(key) for key in
                         ("ExposureTime", "AnalogueGain", "DigitalGain")}}


def clamp(value, limit):
    return max(-limit, min(limit, value))


def choose_command(lateral, heading, lost_for, previous_turn, dt, settings=Settings()):
    """Positive lateral error requests physical right, hence negative raw turn.

    `lateral` is the filtered near-row image error with the IMU-measured
    rotation removed, relative to the centered reference (image units, not a
    distance). It sets a bounded heading toward the lane; relative IMU heading
    regulates the turn toward that heading, so steering keeps working while the
    lane is out of view. The turn changes at a bounded rate. Commands are then
    lifted past the motor dead zone so small corrections actually rotate the
    robot, and are rounded only for the motor command.
    """
    values = (lateral, heading, lost_for, previous_turn, dt)
    if any(value is None or not math.isfinite(value) for value in values):
        raise RuntimeError("Invalid camera/IMU reading")
    if abs(heading) >= settings.abort_heading:
        raise RuntimeError("Heading exceeded bounded test limit")
    if lost_for >= settings.lost_limit:
        raise RuntimeError("Lane not found; run stopped")
    target = clamp(settings.position_gain * lateral, settings.max_target_heading)
    desired = clamp(settings.heading_gain * (target - heading), settings.max_turn)
    turn = previous_turn + clamp(desired - previous_turn, settings.turn_rate * dt)
    command = 0.0
    if abs(turn) >= 0.5:
        span = settings.max_turn - settings.turn_deadzone
        command = math.copysign(settings.turn_deadzone + abs(turn) * span / settings.max_turn, turn)
    speed = int(round(settings.speed * (1 - settings.turn_slowdown * abs(turn) / settings.max_turn)))
    physical_turn = int(round(command))
    return {"speed": speed, "raw_turn": -physical_turn, "physical_turn": physical_turn,
            "turn": round(turn, 3), "target_heading_deg": round(target, 3)}


class LaneTracker:
    """Smoothed lateral lane estimate that survives frames without the lane.

    Rotation shifts the whole lane in the image (about 21 px per degree), so
    the IMU-measured part is removed to estimate sideways offset. While the
    lane is unseen the estimate is held, so the robot keeps steering toward the
    side where the lane was. The search window for the next frame follows the
    robot's rotation so fast turns do not outrun it.
    """

    def __init__(self, reference, now, settings):
        self.settings = settings
        # No search prior until the lane is first found: the robot may have been
        # moved (e.g. to a half-off-road start) after the reference was captured.
        self.fit = None
        self.lateral = 0.0
        self.seen = now
        self.seen_heading = 0.0

    def prior(self, now, heading):
        if self.fit is None or now - self.seen > self.settings.prior_seconds:
            return None
        slope, intercept = self.fit
        return slope, intercept - self.settings.rotation_error_per_deg * (heading - self.seen_heading)

    def trusted(self, detection, error):
        return (detection["status"] == "tracked" and detection["confidence"] >= self.settings.min_confidence
                and error is not None and math.isfinite(error))

    def update(self, detection, error, heading, now):
        settings = self.settings
        if self.trusted(detection, error):
            measured = error + settings.rotation_error_per_deg * heading
            self.lateral += settings.smoothing * (measured - self.lateral)
            self.fit, self.seen, self.seen_heading = detection["fit"], now, heading
            return True
        return False


class Controller:
    """Per-frame lane estimate and drive decision, shared by the robot and replay.

    `observe` measures the lane; `decide` turns that into a command and state.
    The caller sends the command and owns timing and hardware safety checks, so
    a replay of recorded frames makes exactly the decisions the robot made.
    """

    def __init__(self, reference, now, settings, calibration=None):
        self.reference, self.settings, self.calibration = reference, settings, calibration
        self.tracker = LaneTracker(reference, now, settings)
        self.turn = 0.0
        self.last = now
        # Correction added to the IMU heading so it follows the image heading
        # (lane_center calibration) whenever the lane is seen; held while unseen.
        self.heading_bias = 0.0
        self.end_since = None  # When the lane started ending short of the far rows.

    def observe(self, frame, imu_heading, capture_time):
        from track_vision import detect_lane, summarize
        heading = imu_heading + self.heading_bias
        detection = detect_lane(frame, self.tracker.prior(capture_time, heading))
        return detection, summarize(detection, self.reference, imu_heading)

    def decide(self, detection, summary, imu_heading, now, row):
        """Fill `row` with the decision; faults raise after naming the state."""
        tracker, settings = self.tracker, self.settings
        error = summary["near_error_normalized"]
        if self.calibration is not None and settings.heading_correction and tracker.trusted(detection, error):
            from lane_center import image_yaw
            yaw = image_yaw(detection["fit"], self.calibration, settings.rotation_error_per_deg)
            innovation = clamp(yaw - (imu_heading + self.heading_bias), 5.0)  # Limit one bad fit.
            self.heading_bias += settings.heading_correction * innovation
            row["image_yaw_deg"] = round(yaw, 2)
        heading = imu_heading + self.heading_bias
        points = detection["points"]
        short = len(points) >= 4 and points[:, 1].min() > settings.end_top_row
        if tracker.trusted(detection, error) and not short:
            self.end_since = None
        elif short and self.end_since is None:
            self.end_since = now
        seen = tracker.update(detection, error, heading, now)
        lost_for = now - tracker.seen
        row.update(heading_deg=round(heading, 2), lateral_error=round(tracker.lateral, 5),
                   lost_s=round(lost_for, 3))
        try:
            if self.end_since is not None and now - self.end_since >= settings.end_seconds:
                raise RuntimeError("End of lane ahead")
            command = choose_command(tracker.lateral, heading, lost_for, self.turn, now - self.last,
                                     settings)
        except RuntimeError as exc:
            row["state"] = str(exc)
            raise
        self.last = now
        self.turn = command["turn"]
        row.update(command, state="driving" if seen else "searching")
        return command


def new_row(elapsed, frame_id, processing, summary):
    return {"elapsed_s": round(elapsed, 4), "frame_id": frame_id,
            "processing_s": round(processing, 4), **summary,
            "image_yaw_deg": None, "heading_deg": None,
            "lateral_error": None, "lost_s": None, "distance_cm": None,
            "speed": 0, "raw_turn": 0, "physical_turn": 0, "turn": 0.0,
            "target_heading_deg": None, "state": "checking"}


def control_loop(camera, imu, saber, reference, zero, duration, settings, records, recorder=None,
                 clock=time.monotonic, sleep=time.sleep, finish_requested=None,
                 max_distance_cm=None, speed_cm_s=14.0, calibration=None):
    """Track the lane; search for it on IMU heading when unseen; stop on faults.

    No disk writes or console printing while motors are being commanded. Keep
    diagnostic records in memory and write them only after requesting a stop.
    Every frame goes to the recorder (encoded off-thread) for later replay.
    Without wheel encoders, distance is estimated from commanded speed and the
    measured `speed_cm_s` at `settings.speed`; the run ends at `max_distance_cm`.
    """
    import cv2

    start = clock()
    deadline = start + duration
    last_command = start
    count = 0
    distance = 0.0
    last_speed = 0
    previous = 0.0  # Decision times since start; the gap between them includes motor I/O.
    controller = Controller(reference, 0.0, settings, calibration)  # Times are seconds since start.
    try:
        while clock() < deadline:
            if finish_requested is not None and finish_requested():
                return "finish_marker"
            capture_start = clock()
            frame = cv2.cvtColor(camera.capture_array(), cv2.COLOR_BGRA2BGR)
            if recorder is not None:
                recorder.add(count, frame)
            heading = fresh_heading(imu, clock)
            delta = signed_delta(heading, zero, imu.SIGN)
            detection, summary = controller.observe(frame, delta, capture_start - start)
            now = clock()
            row = new_row(now - start, count, now - capture_start, summary)
            records.append(row)
            distance += speed_cm_s * last_speed / settings.speed * (now - start - previous)
            previous = now - start
            row["distance_cm"] = round(distance, 1)
            if max_distance_cm is not None and distance >= max_distance_cm:
                row["state"] = "estimated_distance"
                return "estimated_distance"
            if now >= deadline:
                row["state"] = "duration_complete"
                break
            if finish_requested is not None and finish_requested():
                row["state"] = "finish_marker"
                return "finish_marker"
            if now - last_command > settings.max_sample_age:
                row["state"] = "loop_delayed"
                raise RuntimeError("Control loop delayed; refusing to resume motion")
            if now - capture_start > settings.max_sample_age:
                raise RuntimeError("Camera/processing data stale")
            command = controller.decide(detection, summary, delta, now - start, row)
            if saber is not None:
                if command["speed"] == 0:
                    saber.stop()
                else:
                    saber.drive(command["speed"], command["raw_turn"])
            last_command = clock()
            last_speed = command["speed"]
            count += 1
            sleep(max(0, min(0.04 - (clock() - capture_start), deadline - clock())))
        return "duration_complete"
    finally:
        if saber is not None:
            saber.stop()


def measure_start(camera, imu, calibration, settings, samples=10, clock=time.monotonic, sleep=time.sleep):
    """Stationary start-up: find the lane and the robot's heading from the image.

    There is no centered reference capture. The calibrated center
    (`lane_center.json`) gives the lane position, and the image gives the
    heading relative to the track, which sets the IMU zero. Refuses to start
    when the lane is not seen in most start-up frames.
    """
    import cv2
    from lane_center import image_yaw
    from track_vision import NEAR_Y, detect_lane

    first = None
    yaws, offsets, nears, frame = [], [], [], None
    for _ in range(samples):
        frame = cv2.cvtColor(camera.capture_array(), cv2.COLOR_BGRA2BGR)
        heading = fresh_heading(imu, clock)
        first = heading if first is None else first
        detection = detect_lane(frame)
        if detection["status"] == "tracked" and detection["confidence"] >= settings.min_confidence:
            yaws.append(image_yaw(detection["fit"], calibration, settings.rotation_error_per_deg))
            offsets.append(signed_delta(heading, first, 1))
            nears.append(detection["fit"][0] * NEAR_Y + detection["fit"][1])
        sleep(0.04)
    if len(yaws) < max(3, samples // 2):
        raise RuntimeError("Lane not visible at start; place the robot where the camera sees the lane")
    yaw = statistics.median(yaws)
    heading = (first + statistics.median(offsets)) % 360.0
    zero = (heading - imu.SIGN * yaw) % 360.0  # So the relative heading starts at `yaw`.
    near_error = statistics.median(nears) - calibration["near_x"]
    lateral_px = (near_error + settings.rotation_error_per_deg * yaw) * calibration["width"]
    return zero, {"frame": frame, "yaw_deg": round(yaw, 2), "yaw_spread_deg": round(statistics.pstdev(yaws), 2),
                  "lateral_px": round(lateral_px, 1), "lane_frames": len(yaws), "frames": samples}


def save_results(output, records, recorder, outcome):
    from run_metrics import score
    if records:
        with (output / "readings.csv").open("x", newline="") as log:
            writer = csv.DictWriter(log, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    if recorder is not None:
        outcome["recording"] = recorder.save(output / "frames")
    metrics = score(records, outcome["settings"]["min_confidence"])
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "outcome.json").write_text(json.dumps(outcome, indent=2) + "\n")
    return metrics


def read_command():
    """Nonblocking terminal line: None if nothing typed. EOF stops safely."""
    if select.select([sys.stdin], [], [], 0)[0]:
        line = sys.stdin.readline()
        if not line:
            raise RuntimeError("Remote input closed; run stopped")
        return line.strip().lower()
    return None


def leg_timing(leg_ft, estimated_speed_cm_s):
    nominal = leg_ft * 30.48 / estimated_speed_cm_s
    # A leg stops at the estimated distance or the operator's Enter; this is
    # only a bounded time fallback if the estimate is badly wrong.
    return nominal, min(90.0, nominal * 2.0)


# Faults that end one leg but not the session: the robot stops and waits to be placed.
RECOVERABLE = ("Lane not found", "Heading exceeded", "Lane not visible at start", "End of lane ahead")


def wait_for_placement(camera, settings, stopped_near=None, settle=1.5, moved_px=40, steady_px=10,
                       commands=None, clock=time.monotonic, sleep=time.sleep, notify=print):
    """Stopped: return "go" once the robot sits on the lane, "quit" on q.

    After a leg (`stopped_near` = lane position where it stopped) the robot
    must first be moved: picked up (lane unseen for 0.5 s) or slid (lane
    shifts more than `moved_px`). It then goes once the lane has been seen
    steadily for `settle` seconds, so hands are clear. Enter starts sooner.
    """
    import cv2
    from track_vision import NEAR_Y, detect_lane

    commands = commands or read_command
    moved = stopped_near is None
    unseen_since = steady_since = steady_near = None
    announced = False
    while True:
        command = commands()
        if command == "q":
            return "quit"
        frame = cv2.cvtColor(camera.capture_array(), cv2.COLOR_BGRA2BGR)
        detection = detect_lane(frame)
        now = clock()
        if detection["status"] == "tracked" and detection["confidence"] >= settings.min_confidence:
            near = (detection["fit"][0] * NEAR_Y + detection["fit"][1]) * frame.shape[1]
            unseen_since = None
            if stopped_near is not None and abs(near - stopped_near) > moved_px:
                moved = True
            if steady_near is None or abs(near - steady_near) > steady_px:
                steady_near, steady_since = near, now
            if command == "" or (moved and now - steady_since >= settle):
                return "go"
            if moved and not announced:
                notify(f"Lane found; starting in {settle:g} s unless moved...")
                announced = True
        else:
            steady_near = steady_since = None
            announced = False
            unseen_since = now if unseen_since is None else unseen_since
            if now - unseen_since >= 0.5:
                moved = True
        sleep(0.04)


def run_leg(number, camera, imu, saber, settings, calibration, output_root, leg_ft, speed_cm_s,
            record_every, execute, commands, clock=time.monotonic, sleep=time.sleep):
    """One leg: stationary heading measurement, then drive up to `leg_ft`.

    Always returns with motors stopped and results saved; returns
    (reason, lane position where it stopped or None).
    """
    from lane_center import reference as center_reference
    from run_metrics import describe
    from track_vision import NEAR_Y, save_frame

    reference = center_reference(calibration)
    nominal, duration = leg_timing(leg_ft, speed_cm_s)
    output = output_root / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    records, recorder = [], None
    outcome = {"execute": execute, "trial": f"leg {number}", "leg": number, "leg_ft": leg_ft,
               "nominal_time_estimate_s": round(nominal, 1), "duration_s": duration, "distance_measured": False,
               "estimated_speed_cm_s": speed_cm_s, "settings": settings.__dict__, "reason": "start_failed"}
    try:
        zero, start = measure_start(camera, imu, calibration, settings, clock=clock, sleep=sleep)
        save_frame(output / "start_raw.png", start.pop("frame"))
        outcome["start"] = start
        (output / "reference.json").write_text(json.dumps({
            "fit": reference["fit"], "width": reference["width"], "zero_heading_deg": zero,
            "source": "lane_center.json calibration; IMU zero from start-up image heading", **start,
        }, indent=2) + "\n")
        print(f"Leg {number}: {start['yaw_deg']:+.1f} deg from the track direction, lane offset "
              f"{start['lateral_px']:+.0f} px. Driving up to {leg_ft:g} ft; Enter stops the leg.")
        if record_every:
            from run_recording import FrameRecorder
            recorder = FrameRecorder(every=record_every)
        outcome["reason"] = control_loop(
            camera, imu, saber, reference, zero, duration, settings, records, recorder, clock, sleep,
            finish_requested=lambda: commands() is not None, max_distance_cm=leg_ft * 30.48,
            speed_cm_s=speed_cm_s, calibration=calibration)
    except RuntimeError as exc:
        outcome["reason"] = str(exc)
        if not str(exc).startswith(RECOVERABLE):
            raise
    finally:
        if saber is not None:
            saber.stop()
        metrics = save_results(output, records, recorder, outcome)
        print(f"Leg {number} ended: {outcome['reason']}. Saved {output}")
        print("  Score:", describe(metrics))
    last = next((row for row in reversed(records) if row.get("near_error_normalized") is not None), None)
    stopped_near = None
    if last is not None:
        slope, intercept = reference["fit"]
        stopped_near = (slope * NEAR_Y + intercept + last["near_error_normalized"]) * reference["width"]
    return outcome["reason"], stopped_near


def run_session(camera, imu, saber, settings, calibration, output_root, leg_ft=15.0, speed_cm_s=14.0,
                record_every=3, execute=False, max_legs=None, commands=None,
                clock=time.monotonic, sleep=time.sleep):
    """One program for every test: wait to be placed, drive a leg, stop, repeat.

    Place the robot anywhere on the lane (centered, half off at an edge, ...);
    it measures where it is and drives. After each leg it stops and waits to be
    picked up or slid to a new start. `q` + Enter (or Ctrl+C) quits.
    """
    commands = commands or read_command
    legs = 0
    stopped_near = None
    while max_legs is None or legs < max_legs:
        print("Waiting: place the robot on the lane (q + Enter quits, Enter starts now).")
        if wait_for_placement(camera, settings, stopped_near, commands=commands, clock=clock,
                              sleep=sleep) == "quit":
            break
        legs += 1
        _, stopped_near = run_leg(legs, camera, imu, saber, settings, calibration, output_root, leg_ft,
                                  speed_cm_s, record_every, execute, commands, clock, sleep)
    return legs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="enable physical motor commands")
    parser.add_argument("--leg-ft", type=positive_number, default=15.0,
                        help="distance per leg, maximum 20 ft; default 15 (three mats)")
    parser.add_argument("--legs", type=bounded_int(1, 100), help="stop after this many legs (default: until q)")
    parser.add_argument("--estimated-speed-cm-s", type=positive_number, default=14.0,
                        help="speed at command 25 for distance estimates; default 14 from course runs")
    parser.add_argument("--speed", type=bounded_int(1, 25), default=25)
    parser.add_argument("--max-turn", type=bounded_int(1, 20), default=20,
                        help="steering command limit (of 127); default 20")
    parser.add_argument("--camera", type=int)
    parser.add_argument("--exposure-us", type=bounded_int(0, 20000), default=5000,
                        help="short exposure in microseconds; 0 keeps auto exposure")
    parser.add_argument("--record-every", type=bounded_int(0, 10), default=3,
                        help="save every Nth camera frame as JPEG for replay (default 3, ~10 MB per "
                             "15 ft leg); 1 keeps all; 0 disables")
    parser.add_argument("--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
                        help=f"tuning override, e.g. --set position_gain=30 turn_rate=30; keys: {', '.join(TUNABLE)}")
    parser.add_argument("--output", type=Path, default=Path("centering_results"))
    args = parser.parse_args(argv)
    if args.leg_ft > 20:
        parser.error("--leg-ft must be <= 20")
    if args.speed != 25:
        parser.error("distance estimates currently support only calibrated speed command 25")
    if args.camera is not None and args.camera < 0:
        parser.error("--camera must be nonnegative")
    try:
        settings = tuned(Settings(speed=args.speed, max_turn=args.max_turn), args.set)
    except ValueError as exc:
        parser.error(str(exc))
    camera = imu = saber = None
    started = False
    handlers = {}

    def terminate(_signal, _frame):
        raise KeyboardInterrupt

    for signum in (signal.SIGTERM, signal.SIGHUP):
        handlers[signum] = signal.signal(signum, terminate)
    try:
        from picamera2 import Picamera2
        from imu_controller import IMUDevice
        from lane_center import load as load_center

        infos = Picamera2.global_camera_info()
        for index, info in enumerate(infos):
            print(f"Camera {index}: {info}")
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
        exposure = configure_motion_exposure(camera, args.exposure_us)
        print("Camera exposure:", json.dumps(exposure, default=str))
        if args.execute:
            from sabertooth import Sabertooth
            saber = Sabertooth()
            if saber.ser is None or not saber.ser.is_open:
                raise RuntimeError("Motor serial port unavailable")
            saber.stop()
            saber.set_auto_stop(200)
            saber.set_ramping(8)
        nominal, _ = leg_timing(args.leg_ft, args.estimated_speed_cm_s)
        print(f"{'POWERED' if args.execute else 'MOTOR-FREE'} session: legs of up to {args.leg_ft:g} ft "
              f"(~{nominal:.0f} s, estimated from speed).")
        print("Stop other control programs. Keep a physical stop accessible; Ctrl+C stops.")
        print("Remote disconnect is not a guaranteed stop; hardware timeout is not yet verified.")
        print(f"Each leg: it holds still ~0.5 s to find the lane and its heading, then drives. Lane unseen for "
              f"{settings.lost_limit:g} s or {settings.abort_heading:g} deg of rotation ends the leg.")
        legs = run_session(camera, imu, saber, settings, load_center(), args.output, args.leg_ft,
                           args.estimated_speed_cm_s, args.record_every, args.execute, args.legs)
        print(f"Session finished after {legs} leg(s); motors stopped.")
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted; stop requested.")
        return 2
    except Exception as exc:
        print(f"Centering session stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        # Stop first; never delay motor stopping for image encoding/disk writes.
        actions = ([saber.stop, saber.close] if saber is not None else [])
        actions += ([imu.close] if imu is not None else [])
        if camera is not None:
            actions += ([camera.stop] if started else []) + [camera.close]
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
