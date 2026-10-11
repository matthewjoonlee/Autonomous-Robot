"""Bounded straight-track centering test. Motor-free diagnostics by default."""

import argparse
import csv
from dataclasses import dataclass
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
    min_confidence: float = 0.65
    image_deadband: float = 0.0125  # 8 px at 640 px; provisional.
    max_sample_age: float = 0.15
    reacquire_seconds: float = 1.0
    reacquire_frames: int = 3
    max_reacquisitions: int = 5


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


def choose_command(error, heading, confidence, status, sample_age, settings=Settings()):
    """Positive image error requests physical right, hence negative raw turn.

    Image error includes both translation and perspective changes from rotation.
    It is feedback relative to the centered reference, not a distance estimate.
    """
    if status != "tracked":
        raise RuntimeError(f"Tracking fault: {status}")
    values = (error, heading, confidence, sample_age)
    if any(value is None or not math.isfinite(value) for value in values):
        raise RuntimeError("Invalid camera/IMU reading")
    if confidence < settings.min_confidence:
        raise RuntimeError("Tracking confidence too low")
    if sample_age < 0 or sample_age > settings.max_sample_age:
        raise RuntimeError("Camera/processing data stale")
    if abs(heading) >= settings.abort_heading:
        raise RuntimeError("Heading exceeded bounded test limit")
    effective_error = 0 if abs(error) <= settings.image_deadband else error
    target = clamp(settings.position_gain * effective_error, settings.max_target_heading)
    physical_turn = int(round(clamp(settings.heading_gain * (target - heading), settings.max_turn)))
    return {"speed": settings.speed, "raw_turn": -physical_turn,
            "physical_turn": physical_turn, "target_heading_deg": target}


def control_loop(camera, imu, saber, reference, zero, duration, settings, records, frames,
                 clock=time.monotonic, sleep=time.sleep, finish_requested=None, frame_interval=0.25):
    """Sample before each command; pause on brief tracking loss, stop on faults.

    No disk writes or console printing while motors are being commanded. Keep
    diagnostic records in memory and write them only after requesting a stop.
    """
    import cv2
    from track_vision import detect_line, summarize

    start = clock()
    deadline = start + duration
    last_command = start
    last_saved = -1.0
    last_frame = None
    last_frame_id = None
    count = 0
    lost_since = None
    good_frames = 0
    recoveries = 0
    try:
        while clock() < deadline:
            if finish_requested is not None and finish_requested():
                return "finish_marker"
            capture_start = clock()
            frame = cv2.cvtColor(camera.capture_array(), cv2.COLOR_BGRA2BGR)
            last_frame = frame
            last_frame_id = count
            heading = fresh_heading(imu, clock)
            delta = signed_delta(heading, zero, imu.SIGN)
            detection = detect_line(frame)
            summary = summarize(detection, reference, delta)
            now = clock()
            row = {"elapsed_s": round(now - start, 4), "frame_id": count,
                   "processing_s": round(now - capture_start, 4), **summary,
                   "speed": 0, "raw_turn": 0, "physical_turn": 0,
                   "target_heading_deg": None, "state": "checking"}
            records.append(row)
            if now - start - last_saved >= frame_interval:
                frames.append((count, frame.copy()))
                last_saved = now - start
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
            if abs(delta) >= settings.abort_heading:
                raise RuntimeError("Heading exceeded bounded test limit")
            trustworthy = (detection["status"] == "tracked"
                           and detection["confidence"] >= settings.min_confidence)
            if not trustworthy or lost_since is not None:
                # A blur frame requests a stop immediately. Reacquisition uses
                # fresh camera samples while stopped, not blind forward motion.
                if saber is not None:
                    saber.stop()
                last_command = clock()
                if lost_since is None:
                    recoveries += 1
                    if recoveries > settings.max_reacquisitions:
                        raise RuntimeError("Repeated tracking interruptions; run stopped")
                    lost_since = now
                    good_frames = 0
                if trustworthy:
                    good_frames += 1
                else:
                    good_frames = 0
                row["state"] = "reacquiring"
                if now - lost_since >= settings.reacquire_seconds:
                    raise RuntimeError("Tracking did not recover while stopped")
                if good_frames < settings.reacquire_frames:
                    # Preserve interruption frames even between routine snapshots.
                    if not frames or frames[-1][0] != count:
                        frames.append((count, frame.copy()))
                    count += 1
                    sleep(max(0, min(0.04 - (clock() - capture_start), deadline - clock())))
                    continue
                lost_since = None
                good_frames = 0
            try:
                command = choose_command(summary["near_error_normalized"], delta,
                                         detection["confidence"], detection["status"],
                                         now - capture_start, settings)
            except RuntimeError as exc:
                row["state"] = str(exc)
                raise
            if saber is not None:
                saber.drive(command["speed"], command["raw_turn"])
            row.update(command, state="driving" if saber is not None else "diagnostic")
            last_command = clock()
            count += 1
            sleep(max(0, min(0.04 - (clock() - capture_start), deadline - clock())))
        if lost_since is not None:
            raise RuntimeError("Tracking not recovered before time limit")
        return "duration_complete"
    finally:
        if saber is not None:
            saber.stop()
        if last_frame is not None and (not frames or frames[-1][0] != last_frame_id):
            frames.append((last_frame_id, last_frame.copy()))


def save_results(output, records, frames, outcome):
    from track_vision import save_frame
    if records:
        with (output / "readings.csv").open("x", newline="") as log:
            writer = csv.DictWriter(log, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    raw = output / "raw"
    raw.mkdir(exist_ok=True)
    for index, frame in frames:
        save_frame(raw / f"frame_{index:05d}.png", frame)
    (output / "outcome.json").write_text(json.dumps(outcome, indent=2) + "\n")


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
    settings = Settings(speed=args.speed, max_turn=args.max_turn)
    camera = imu = saber = None
    started = False
    output = None
    records, frames = [], []
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
        from track_vision import detect_line, save_frame

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
        reference = detect_line(frame)
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
            print("Recovery is under test. Large rotation/line loss still stops the program.")
        answer = input("Type RUN to begin the bounded test (anything else cancels): ")
        if answer.strip() != "RUN":
            outcome["reason"] = "cancelled"
            return 2
        reason = control_loop(camera, imu, saber, reference, zero, args.duration, settings, records, frames,
                              finish_requested=finish_entered if nominal is not None else None,
                              frame_interval=1.0 if nominal is not None else 0.25)
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
                save_results(output, records, frames, outcome)
                print(f"Saved results: {output}")
            except Exception as exc:
                print(f"Could not save all results: {exc}", file=sys.stderr)
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    return 0


if __name__ == "__main__":
    sys.exit(main())
