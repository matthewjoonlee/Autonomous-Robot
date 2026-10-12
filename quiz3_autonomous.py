"""Bounded straight-track centering test. Motor-free diagnostics by default."""

import argparse
import csv
from dataclasses import dataclass, replace
from datetime import datetime
import json
import math
from pathlib import Path
import select
import signal
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
    position_gain: float = 60.0  # Degrees of target heading per normalized lateral error; provisional.
    heading_gain: float = 3.0    # Physical turn units per degree; provisional.
    max_target_heading: float = 6.0
    abort_heading: float = 45.0  # Runaway guard. A nudge rotated the robot ~11 deg and was recoverable.
    min_confidence: float = 0.35
    smoothing: float = 0.5       # Weight of each new lane measurement; 1 disables filtering.
    # Near-row image error per degree of yaw: capture 005 moved the lane -272 px at +13.1 deg.
    rotation_error_per_deg: float = 0.0323
    turn_deadzone: float = 4.0   # Turn commands below ~5 barely rotate the robot (course-run data).
    turn_rate: float = 40.0      # Max change in physical turn units per second.
    turn_slowdown: float = 0.0   # Speed fraction shed at full turn; 0 until lower speeds are validated.
    lost_limit: float = 3.0      # Seconds searching for an unseen lane before stopping (~42 cm).
    prior_seconds: float = 0.3   # Search near the last lane fit only while it is this recent.
    max_sample_age: float = 0.15


TUNABLE = ("position_gain", "heading_gain", "max_target_heading", "min_confidence", "smoothing",
           "rotation_error_per_deg", "turn_deadzone", "turn_rate", "turn_slowdown", "prior_seconds")


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
    if not (0 < result.smoothing <= 1 and 0 <= result.turn_slowdown < 1
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
        self.fit = reference["fit"]
        self.lateral = 0.0
        self.seen = now
        self.seen_heading = 0.0

    def prior(self, now, heading):
        if now - self.seen > self.settings.prior_seconds:
            return None
        slope, intercept = self.fit
        return slope, intercept - self.settings.rotation_error_per_deg * (heading - self.seen_heading)

    def update(self, detection, error, heading, now):
        settings = self.settings
        if (detection["status"] == "tracked" and detection["confidence"] >= settings.min_confidence
                and error is not None and math.isfinite(error)):
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

    def __init__(self, reference, now, settings):
        self.reference, self.settings = reference, settings
        self.tracker = LaneTracker(reference, now, settings)
        self.turn = 0.0
        self.last = now

    def observe(self, frame, heading, capture_time):
        from track_vision import detect_lane, summarize
        detection = detect_lane(frame, self.tracker.prior(capture_time, heading))
        return detection, summarize(detection, self.reference, heading)

    def decide(self, detection, summary, heading, now, row):
        """Fill `row` with the decision; faults raise after naming the state."""
        tracker = self.tracker
        seen = tracker.update(detection, summary["near_error_normalized"], heading, now)
        lost_for = now - tracker.seen
        row.update(lateral_error=round(tracker.lateral, 5), lost_s=round(lost_for, 3))
        try:
            command = choose_command(tracker.lateral, heading, lost_for, self.turn, now - self.last,
                                     self.settings)
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
            "lateral_error": None, "lost_s": None, "distance_cm": None,
            "speed": 0, "raw_turn": 0, "physical_turn": 0, "turn": 0.0,
            "target_heading_deg": None, "state": "checking"}


def control_loop(camera, imu, saber, reference, zero, duration, settings, records, recorder=None,
                 clock=time.monotonic, sleep=time.sleep, finish_requested=None,
                 max_distance_cm=None, speed_cm_s=14.0):
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
    controller = Controller(reference, 0.0, settings)  # Times are seconds since start.
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
            distance += speed_cm_s * last_speed / settings.speed * (now - last_command)
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


def finish_entered():
    """Nonblocking terminal stop at the marked finish; EOF also stops safely."""
    if select.select([sys.stdin], [], [], 0)[0]:
        if not sys.stdin.readline():
            raise RuntimeError("Remote input closed; run stopped")
        return True
    return False


def course_timing(distance_ft, estimated_speed_cm_s):
    nominal = distance_ft * 30.48 / estimated_speed_cm_s
    # The run stops at the estimated distance or the operator's Enter; this is
    # only a bounded time fallback if the estimate is badly wrong.
    return nominal, min(30.0, nominal * 2.0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="enable physical motor commands")
    parser.add_argument("--duration", type=positive_number, help="seconds, maximum 10; default 3")
    parser.add_argument("--trial", choices=["straight", "left", "right", "nudge"], default="straight")
    parser.add_argument("--distance-ft", type=positive_number,
                        help="course length, maximum 5 ft (one mat); stops at the estimated distance")
    parser.add_argument("--estimated-speed-cm-s", type=positive_number, default=14.0,
                        help="speed at command 25 for distance estimates; default 14 from course runs")
    parser.add_argument("--speed", type=bounded_int(1, 25), default=25)
    parser.add_argument("--max-turn", type=bounded_int(1, 20), default=20,
                        help="steering command limit (of 127); default 20")
    parser.add_argument("--camera", type=int)
    parser.add_argument("--exposure-us", type=bounded_int(0, 20000), default=5000,
                        help="short exposure in microseconds; 0 keeps auto exposure")
    parser.add_argument("--record-every", type=bounded_int(0, 10), default=1,
                        help="save every Nth camera frame as JPEG for replay; 0 disables")
    parser.add_argument("--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
                        help=f"tuning override, e.g. --set position_gain=30 turn_rate=30; keys: {', '.join(TUNABLE)}")
    parser.add_argument("--output", type=Path, default=Path("centering_results"))
    args = parser.parse_args(argv)
    if args.duration is not None and args.duration > 10:
        parser.error("--duration must be <= 10 seconds for initial testing")
    nominal = None
    if args.distance_ft is not None:
        if args.distance_ft > 5 or args.duration is not None:
            parser.error("--distance-ft must be <= 5 and cannot be combined with --duration")
        if args.speed != 25:
            parser.error("course timing currently supports only calibrated speed command 25")
        nominal, args.duration = course_timing(args.distance_ft, args.estimated_speed_cm_s)
    else:
        args.duration = args.duration or 3.0
    if args.camera is not None and args.camera < 0:
        parser.error("--camera must be nonnegative")
    try:
        settings = tuned(Settings(speed=args.speed, max_turn=args.max_turn), args.set)
    except ValueError as exc:
        parser.error(str(exc))
    camera = imu = saber = None
    started = False
    output = None
    records, recorder = [], None
    outcome = {"execute": args.execute, "duration_s": args.duration,
               "trial": args.trial, "requested_distance_ft": args.distance_ft,
               "nominal_time_estimate_s": nominal, "distance_measured": False,
               "estimated_speed_cm_s": args.estimated_speed_cm_s if nominal is not None else None,
               "settings": settings.__dict__, "reason": "initialization_failed"}
    handlers = {}

    def terminate(_signal, _frame):
        raise KeyboardInterrupt

    for signum in (signal.SIGTERM, signal.SIGHUP):
        handlers[signum] = signal.signal(signum, terminate)
    try:
        import cv2
        from picamera2 import Picamera2
        from imu_controller import IMUDevice
        from track_vision import detect_lane, save_frame

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
        outcome["camera_exposure"] = configure_motion_exposure(camera, args.exposure_us)
        print("Camera exposure:", json.dumps(outcome["camera_exposure"], default=str))
        if args.execute:
            from sabertooth import Sabertooth
            saber = Sabertooth()
            if saber.ser is None or not saber.ser.is_open:
                raise RuntimeError("Motor serial port unavailable")
            saber.stop()
            saber.set_auto_stop(200)
            saber.set_ramping(8)
        print(f"{'POWERED' if args.execute else 'MOTOR-FREE'} bounded run: {args.duration}s.")
        print("Stop other control programs. Align centered/straight on continuous track.")
        print("Keep a physical stop accessible; Ctrl+C stops.")
        if nominal is not None:
            print(f"Mark start/finish {args.distance_ft:g} ft apart. Nominal time ~{nominal:.1f}s.")
            print(f"Stops automatically at an estimated {args.distance_ft:g} ft (from speed, not measured);")
            print(f"press Enter at the finish to stop sooner. Time fallback {args.duration:.1f}s.")
        print("Remote disconnect is not a guaranteed stop; hardware timeout is not yet verified.")
        input("Press Enter while centered and straight to capture the reference: ")
        output = args.output / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        output.mkdir(parents=True, exist_ok=False)
        frame = cv2.cvtColor(camera.capture_array(), cv2.COLOR_BGRA2BGR)
        reference = detect_lane(frame)
        save_frame(output / "reference_raw.png", frame)
        if reference["status"] != "tracked" or reference["confidence"] < settings.min_confidence:
            raise RuntimeError("Centered reference not reliably tracked")
        reference["width"] = frame.shape[1]
        zero = fresh_heading(imu)
        (output / "reference.json").write_text(json.dumps({
            "fit": reference["fit"], "width": reference["width"], "zero_heading_deg": zero,
        }, indent=2) + "\n")
        print(f"Results: {output}")
        instructions = {
            "straight": "Keep centered and straight; no pushes during this trial.",
            "left": "Now slide approximately 3 cm LEFT without rotating; keep this centered reference.",
            "right": "Now slide approximately 3 cm RIGHT without rotating; keep this centered reference.",
            "nudge": "Start centered. After forward motion begins, apply one small sideways foot nudge.",
        }
        print(instructions[args.trial])
        if args.trial != "straight":
            print("Recovery is under test. If the lane leaves view, the robot keeps driving and steers back")
            print(f"toward it on IMU heading; {settings.lost_limit:g}s without the lane or "
                  f"{settings.abort_heading:g} deg of rotation ends the run.")
        answer = input("Type RUN to begin the bounded test (anything else cancels): ")
        if answer.strip() != "RUN":
            outcome["reason"] = "cancelled"
            return 2
        if args.record_every:
            from run_recording import FrameRecorder
            recorder = FrameRecorder(every=args.record_every)
        reason = control_loop(camera, imu, saber, reference, zero, args.duration, settings, records, recorder,
                              finish_requested=finish_entered if nominal is not None else None,
                              max_distance_cm=args.distance_ft * 30.48 if nominal is not None else None,
                              speed_cm_s=args.estimated_speed_cm_s)
        outcome["reason"] = reason
        if nominal is not None and reason == "duration_complete":
            print("Fallback time limit reached. Five-foot completion is unconfirmed.")
        print("Run finished; motors stopped." if args.execute else "Motor-free run finished.")
        note = input("Movement/stopping observations (Enter to skip): ")
        (output / "operator_note.txt").write_text(note + "\n")
    except (KeyboardInterrupt, EOFError):
        outcome["reason"] = "interrupted"
        print("\nInterrupted; stop requested.")
        return 2
    except Exception as exc:
        outcome["reason"] = str(exc)
        print(f"Centering test stopped: {exc}", file=sys.stderr)
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
        if output is not None:
            try:
                from run_metrics import describe
                metrics = save_results(output, records, recorder, outcome)
                print(f"Saved results: {output}")
                print("Run score:", describe(metrics))
            except Exception as exc:
                print(f"Could not save all results: {exc}", file=sys.stderr)
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    return 0


if __name__ == "__main__":
    sys.exit(main())
