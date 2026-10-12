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
    max_turn: int = 10
    position_gain: float = 40.0  # Degrees per normalized image error; provisional.
    heading_gain: float = 3.0    # Physical turn units per degree; provisional.
    max_target_heading: float = 4.0
    abort_heading: float = 10.0
    min_confidence: float = 0.5
    smoothing: float = 0.5       # Weight of each new lane measurement; 1 disables filtering.
    turn_rate: float = 40.0      # Max change in physical turn units per second.
    turn_slowdown: float = 0.0   # Speed fraction shed at full turn; 0 until lower speeds are validated.
    coast_full: float = 0.3      # Seconds of lane loss at full speed, holding IMU heading.
    coast_stop: float = 1.0      # Seconds of lane loss by which speed has ramped to zero.
    coast_decay: float = 0.5     # Seconds; held lane offset relaxes toward straight while coasting.
    lost_limit: float = 3.0      # Seconds of lane loss (stopped after coast_stop) before a fault.
    resume_frames: int = 2       # Consecutive lane frames needed to restart after a full stop.
    prior_seconds: float = 0.3   # Search near the last lane fit only while it is this recent.
    max_sample_age: float = 0.15


TUNABLE = ("position_gain", "heading_gain", "max_target_heading", "min_confidence", "smoothing",
           "turn_rate", "turn_slowdown", "coast_full", "coast_stop", "coast_decay",
           "resume_frames", "prior_seconds")


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
            and result.coast_full < result.coast_stop < result.lost_limit
            and result.turn_rate > 0 and result.coast_decay > 0 and result.resume_frames >= 1):
        raise ValueError("--set values out of range (check smoothing, coast times, turn_rate)")
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


def choose_command(offset, heading, lost_for, previous_turn, dt, settings=Settings()):
    """Positive image error requests physical right, hence negative raw turn.

    `offset` is the filtered image error relative to the centered reference; it
    includes both translation and perspective changes from rotation, so it is
    feedback, not a distance estimate. Relative IMU heading tracks the requested
    heading, so steering stays smooth and keeps working while the lane is
    briefly unseen. Speed ramps to zero as lane loss persists; the turn command
    changes at a bounded rate and is rounded only for the motor command.
    """
    values = (offset, heading, lost_for, previous_turn, dt)
    if any(value is None or not math.isfinite(value) for value in values):
        raise RuntimeError("Invalid camera/IMU reading")
    if abs(heading) >= settings.abort_heading:
        raise RuntimeError("Heading exceeded bounded test limit")
    if lost_for >= settings.lost_limit:
        raise RuntimeError("Lane not found; run stopped")
    if lost_for <= settings.coast_full:
        scale = 1.0
    else:
        scale = max(0.0, (settings.coast_stop - lost_for) / (settings.coast_stop - settings.coast_full))
    target = clamp(settings.position_gain * offset, settings.max_target_heading)
    desired = clamp(settings.heading_gain * (target - heading), settings.max_turn)
    turn = previous_turn + clamp(desired - previous_turn, settings.turn_rate * dt) if scale else 0.0
    speed = int(round(settings.speed * scale
                      * (1 - settings.turn_slowdown * abs(turn) / settings.max_turn)))
    physical_turn = int(round(turn))
    return {"speed": speed, "raw_turn": -physical_turn, "physical_turn": physical_turn,
            "turn": round(turn, 3), "target_heading_deg": round(target, 3)}


class LaneTracker:
    """Smoothed lane offset that survives unusable frames.

    Trusted detections blend into the estimate. Unusable frames leave it in
    place, relaxing toward zero (straight along the track) while coasting.
    """

    def __init__(self, reference, now, settings):
        self.settings = settings
        self.fit = reference["fit"]
        self.offset = 0.0
        self.seen = self.updated = now
        self.streak = 0

    def prior(self, now):
        return self.fit if now - self.seen <= self.settings.prior_seconds else None

    def update(self, detection, error, now):
        settings = self.settings
        elapsed, self.updated = now - self.updated, now
        if (detection["status"] == "tracked" and detection["confidence"] >= settings.min_confidence
                and error is not None and math.isfinite(error)):
            self.offset += settings.smoothing * (error - self.offset)
            self.fit, self.seen = detection["fit"], now
            self.streak += 1
            return True
        self.offset *= math.exp(-elapsed / settings.coast_decay)
        self.streak = 0
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
        self.stopped = False
        self.last = now

    def observe(self, frame, heading, capture_time):
        from track_vision import detect_lane, summarize
        detection = detect_lane(frame, self.tracker.prior(capture_time))
        return detection, summarize(detection, self.reference, heading)

    def decide(self, detection, summary, heading, now, row):
        """Fill `row` with the decision; faults raise after naming the state."""
        tracker, settings = self.tracker, self.settings
        seen = tracker.update(detection, summary["near_error_normalized"], now)
        lost_for = now - tracker.seen
        row.update(filtered_error=round(tracker.offset, 5), lost_s=round(lost_for, 3))
        try:
            command = choose_command(tracker.offset, heading, lost_for, self.turn, now - self.last, settings)
        except RuntimeError as exc:
            row["state"] = str(exc)
            raise
        self.last = now
        if self.stopped and tracker.streak < settings.resume_frames:
            command.update(speed=0, raw_turn=0, physical_turn=0, turn=0.0)
        self.stopped = command["speed"] == 0
        self.turn = command["turn"]
        row.update(command, state="waiting" if self.stopped else "driving" if seen else "coasting")
        return command


def new_row(elapsed, frame_id, processing, summary):
    return {"elapsed_s": round(elapsed, 4), "frame_id": frame_id,
            "processing_s": round(processing, 4), **summary,
            "filtered_error": None, "lost_s": None,
            "speed": 0, "raw_turn": 0, "physical_turn": 0, "turn": 0.0,
            "target_heading_deg": None, "state": "checking"}


def control_loop(camera, imu, saber, reference, zero, duration, settings, records, recorder=None,
                 clock=time.monotonic, sleep=time.sleep, finish_requested=None):
    """Drive through brief lane loss on IMU heading; slow, stop, then fault if it persists.

    No disk writes or console printing while motors are being commanded. Keep
    diagnostic records in memory and write them only after requesting a stop.
    Every frame goes to the recorder (encoded off-thread) for later replay.
    """
    import cv2

    start = clock()
    deadline = start + duration
    last_command = start
    count = 0
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
    # An operator marks actual completion; this is only a bounded fallback.
    return nominal, min(30.0, nominal * 2.0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="enable physical motor commands")
    parser.add_argument("--duration", type=positive_number, help="seconds, maximum 10; default 3")
    parser.add_argument("--trial", choices=["straight", "left", "right", "nudge"], default="straight")
    parser.add_argument("--distance-ft", type=positive_number,
                        help="marked course length, maximum 5 ft; press Enter at finish")
    parser.add_argument("--estimated-speed-cm-s", type=positive_number, default=11.0,
                        help="provisional time estimate only; default 11 from forward trial")
    parser.add_argument("--speed", type=bounded_int(1, 25), default=25)
    parser.add_argument("--max-turn", type=bounded_int(1, 10), default=10)
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
            print(f"Press Enter at the finish. Fallback stop at {args.duration:.1f}s; distance is not measured.")
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
            print("Recovery is under test. Brief lane loss coasts on IMU heading; longer loss slows,")
            print(f"stops, and waits; {settings.lost_limit:g}s without the lane, or large rotation, ends the run.")
        answer = input("Type RUN to begin the bounded test (anything else cancels): ")
        if answer.strip() != "RUN":
            outcome["reason"] = "cancelled"
            return 2
        if args.record_every:
            from run_recording import FrameRecorder
            recorder = FrameRecorder(every=args.record_every)
        reason = control_loop(camera, imu, saber, reference, zero, args.duration, settings, records, recorder,
                              finish_requested=finish_entered if nominal is not None else None)
        outcome["reason"] = reason
        if nominal is not None and reason != "finish_marker":
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
