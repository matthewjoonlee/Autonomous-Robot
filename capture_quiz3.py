"""One photo session for Quiz 3: lane calibration, intersection positions, and signs.

Run on the Pi with motor power off (no motor commands are ever sent):
    python capture_quiz3.py            # full field of view (camera_setup.py)
    python capture_quiz3.py --legacy-fov   # old cropped 640x480 front view

Every capture saves the front AND side camera (full view) plus the IMU heading
relative to the `z` reference. Move the robot by hand, hold still, then type a command:

  z          zero the IMU heading (robot centered and straight on the lane)
  c l r a d x  lane calibration poses: centered straight, left offset, right offset,
             centered rotated left, centered rotated right, combined
  i          intersection position: asks the robot-center distance from the
             cross-road center line in inches (+ before it, - past it)
  m          set the mat label used by `i` (e.g. four_way, right_turn)
  s          sign photo: asks which sign (left/right/straight/pause/none) and where
  n          any other labeled photo
  v          reconnect the side camera after it drops out
  q          finish

If a camera stops responding (e.g. the USB side camera disconnects), the capture gives
up after a few seconds instead of freezing; the front-only photos keep saving.

Results: quiz3_captures/session_*/ with NNN_front_*.png, NNN_side_*.png, captures.csv.
"""

import argparse
import csv
from datetime import datetime
import math
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time

LANE_POSES = {
    "c": "centered_straight",
    "l": "left_offset_straight",
    "r": "right_offset_straight",
    "a": "centered_rotated_left",
    "d": "centered_rotated_right",
    "x": "combined_offset_rotation",
}
SIGNS = ("left", "right", "straight", "pause", "none")
FIELDS = ["timestamp", "number", "kind", "label", "mat", "distance_in", "sign", "note",
          "front_image", "side_image", "heading_rel_deg", "front_exposure_us",
          "front_gain", "view"]
CHECKLIST = """Suggested session (about 20 minutes):
  1. Lane: put the robot centered and straight on a straight mat, z, then
     c c c (move the robot a little between each), l, r, a, d, x (twice each).
  2. Intersection (each mat): put a tape measure beside the road from the cross-road
     center line. m to name the mat, then i at robot-center distances
     36 30 24 20 16 12 10 8 6 4 2 0 -2 -4 -6 -8 -12 inches, robot centered.
     Repeat a few distances slightly left and right of center (put it in the note).
  3. Signs: robot still. s for each sign where the judges will put it, then at a few
     other distances/sides/angles, plus s none with nothing in view."""


def slug(text):
    return re.sub(r"[^a-z0-9.+-]+", "_", text.lower()).strip("_")[:50] or "photo"


def read_heading(imu, tries=10):
    """Heading in degrees, retrying briefly: the BNO055 sometimes returns no reading."""
    if imu is None:
        return None
    for attempt in range(tries):
        try:
            value = imu.heading()
        except Exception:
            value = None
        if value is not None and math.isfinite(value):
            return value
        if attempt < tries - 1:
            time.sleep(0.05)
    return None


def imu_diagnosis(imu, init_error):
    """Why the heading is unavailable, in plain words."""
    if imu is None:
        return (f"The IMU did not start: {init_error}. Check its wiring and power, then run "
                "`i2cdetect -y 1` (the BNO055 should show as 28) or `python test_imu.py`.")
    try:
        euler = imu._sensor.euler  # bypass the wrapper, which hides errors
        return (f"The IMU started but returns no heading (raw euler {euler}). It may still be "
                "settling; wait a few seconds and try z again, or restart the program.")
    except Exception as exc:
        return (f"The IMU started but reading it fails: {exc!r}. Check the I2C wiring, then "
                "run `python test_imu.py`.")


def ask(prompt, allowed=None):
    while True:
        answer = input(prompt).strip()
        if allowed is None or answer.lower() in allowed:
            return answer.lower() if allowed else answer
        print(f"Use one of: {', '.join(allowed)}")


def ask_inches(prompt):
    while True:
        text = input(prompt).strip()
        try:
            return float(text)
        except ValueError:
            print("Enter a number of inches, e.g. 24 or -6.")


HUNG = []  # devices that stopped responding; never touched again (cleanup would hang too)


def with_timeout(work, seconds):
    """Run work() in a helper thread; raise TimeoutError if it takes too long.

    A camera that disconnects mid-capture blocks forever inside libcamera, which
    froze this tool; the stuck helper thread is abandoned instead.
    """
    box = {}

    def run():
        try:
            box["value"] = work()
        except BaseException as exc:
            box["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        raise TimeoutError(f"no response in {seconds:g} s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def open_side(Picamera2, side_config, choose_side, index=None):
    camera = Picamera2(index if index is not None else choose_side(Picamera2.global_camera_info()))
    camera.configure(side_config(camera))
    camera.start()
    return camera


def grab(camera, convert):
    request = camera.capture_request()
    try:
        frame = convert(request.make_array("main"))
        metadata = request.get_metadata()
    finally:
        request.release()
    return frame, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--legacy-fov", action="store_true", help="old cropped 640x480 front view")
    parser.add_argument("--front", type=int, help="front camera index override")
    parser.add_argument("--side", type=int, help="side camera index override")
    parser.add_argument("--output", type=Path, default=Path("quiz3_captures"))
    args = parser.parse_args()

    front = side = imu = None
    imu_error = None
    started = []
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    session = None
    count = 0
    try:
        import cv2
        from picamera2 import Picamera2
        from calibrate_track import choose_front
        from camera_setup import front_bgr, front_config, side_bgr, side_config
        from capture_side_camera import choose_side

        infos = Picamera2.global_camera_info()
        for index, info in enumerate(infos):
            print(f"Camera {index}: {info}")
        front = Picamera2(args.front if args.front is not None else choose_front(infos))
        front.configure(front_config(front, legacy=args.legacy_fov))
        front.start()
        started.append(front)
        try:
            side = with_timeout(lambda: open_side(Picamera2, side_config, choose_side, args.side), 10)
            started.append(side)
        except Exception as exc:
            print(f"WARNING: side camera unavailable ({exc}); saving front only.")
            side = None
        try:
            from imu_controller import IMUDevice
            imu = IMUDevice()
        except Exception as exc:
            imu_error = repr(exc)
            imu = None
        time.sleep(2)
        metadata = front.capture_metadata()
        if "ColourGains" in metadata:  # same white-balance lock as the driving programs
            front.set_controls({"AwbEnable": False, "ColourGains": metadata["ColourGains"]})

        view = "legacy_640x480" if args.legacy_fov else "full"
        session = args.output / datetime.now().strftime("session_%Y%m%d_%H%M%S")
        session.mkdir(parents=True, exist_ok=False)
        print(f"Saving to {session} ({view} front view). No motor commands are sent.")
        print(CHECKLIST)
        print("Commands:" + __doc__.split("type a command:")[1].split("Results:")[0])
        if read_heading(imu) is None:
            print("WARNING: no IMU heading. " + imu_diagnosis(imu, imu_error))
            print("Photos still save; for a, d, x write the approximate angle in the note.")
        reference = None
        mat = ""
        with (session / "captures.csv").open("x", newline="") as log:
            writer = csv.DictWriter(log, fieldnames=FIELDS)
            writer.writeheader()
            while True:
                command = ask("command (z c l r a d x i m s n v q)> ")
                if command == "q":
                    break
                if command == "z":
                    reference = read_heading(imu)
                    print("Heading unavailable. " + imu_diagnosis(imu, imu_error) if reference is None
                          else f"Reference {reference:.2f} deg.")
                    continue
                if command == "v":
                    if side is not None:
                        print("Side camera is already working.")
                        continue
                    try:
                        side = with_timeout(lambda: open_side(Picamera2, side_config, choose_side), 10)
                        started.append(side)
                        print("Side camera reconnected.")
                    except Exception as exc:
                        print(f"Side camera still unavailable ({exc!r}). Check its USB cable; "
                              "if this keeps failing, quit with q and restart the program.")
                    continue
                if command == "m":
                    mat = slug(ask("Mat label (e.g. four_way, right_turn)> "))
                    continue
                row = {"mat": "", "distance_in": "", "sign": ""}
                if command in LANE_POSES:
                    kind, label = "lane", LANE_POSES[command]
                    if reference is None:
                        print("Note: no z reference yet; headings will be blank.")
                elif command == "i":
                    while not mat:
                        mat = slug(ask("Mat label (e.g. four_way, right_turn)> "))
                    distance = ask_inches("Robot-center distance from cross-road center, inches "
                                          "(+ before, - past)> ")
                    kind, label = "intersection", f"{mat}_{distance:+g}in"
                    row.update(mat=mat, distance_in=distance)
                elif command == "s":
                    sign = ask("Sign (left/right/straight/pause/none)> ", SIGNS)
                    kind, label = "sign", sign
                    row["sign"] = sign
                elif command == "n":
                    kind, label = "other", ask("Label> ")
                else:
                    print("Unknown command.")
                    continue
                note = ask("Note (offset, sign position/distance, lighting; Enter to skip)> ")
                time.sleep(0.3)
                print("Capturing...", flush=True)
                try:
                    frame, meta = with_timeout(lambda: grab(front, front_bgr), 5)
                except TimeoutError:
                    HUNG.append(front)
                    print("The FRONT camera stopped responding. Photos so far are saved. "
                          "Check its ribbon cable, then restart the program.")
                    break
                side_frame = None
                if side is not None:
                    try:
                        side_frame = with_timeout(lambda: grab(side, side_bgr)[0], 5)
                    except Exception as exc:
                        if isinstance(exc, TimeoutError):
                            HUNG.append(side)
                        print(f"The SIDE camera stopped responding ({exc}). Check its USB cable. "
                              "Saving this photo front-only; type v to reconnect the side camera.")
                        side = None
                try:
                    heading = with_timeout(lambda: read_heading(imu), 3)
                except TimeoutError:
                    HUNG.append(imu)
                    print("The IMU stopped responding; continuing without headings.")
                    imu, heading = None, None
                count += 1
                stem = f"{count:03d}_{{}}_{slug(label)}.png"
                front_name = stem.format("front")
                if not cv2.imwrite(str(session / front_name), frame):
                    raise RuntimeError(f"Could not save {front_name}")
                side_name = ""
                if side_frame is not None:
                    side_name = stem.format("side")
                    cv2.imwrite(str(session / side_name), side_frame)
                rel = None
                if heading is not None and reference is not None:
                    from test_imu import signed_delta
                    rel = signed_delta(heading, reference, imu.SIGN)  # right positive
                row.update(timestamp=datetime.now().astimezone().isoformat(), number=count,
                           kind=kind, label=label, note=note, front_image=front_name,
                           side_image=side_name, heading_rel_deg=rel,
                           front_exposure_us=meta.get("ExposureTime"),
                           front_gain=meta.get("AnalogueGain"), view=view)
                writer.writerow(row)
                log.flush()
                print(f"Saved #{count} {label}" + ("" if rel is None else f" (heading {rel:+.1f} deg)"))
    except (KeyboardInterrupt, EOFError):
        print("\nStopped; saved photos are kept.")
    except Exception as exc:
        print(f"Capture failed: {exc}", file=sys.stderr)
        return 1
    finally:
        for camera in started:
            if any(camera is hung for hung in HUNG):
                continue
            try:
                with_timeout(camera.stop, 5)
            except Exception:
                pass
        closed = set()
        for device in started + [imu]:
            if device is not None and id(device) not in closed and not any(device is h for h in HUNG):
                closed.add(id(device))
                try:
                    with_timeout(device.close, 5)
                except Exception:
                    pass
        signal.signal(signal.SIGTERM, previous)
    if session is not None:
        print(f"Saved {count} captures in {session}")
    return 0


if __name__ == "__main__":
    code = main()
    if HUNG:  # a stuck camera thread would keep Python from exiting
        sys.stdout.flush()
        os._exit(code)
    sys.exit(code)
