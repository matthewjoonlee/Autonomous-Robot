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
  q          finish

Results: quiz3_captures/session_*/ with NNN_front_*.png, NNN_side_*.png, captures.csv.
"""

import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import re
import signal
import sys
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


def read_heading(imu):
    if imu is None:
        return None
    try:
        value = imu.heading()
    except Exception:
        return None
    return value if value is not None and math.isfinite(value) else None


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
            side = Picamera2(args.side if args.side is not None else choose_side(infos))
            side.configure(side_config(side))
            side.start()
            started.append(side)
        except Exception as exc:
            print(f"WARNING: side camera unavailable ({exc}); saving front only.")
            side = None
        try:
            from imu_controller import IMUDevice
            imu = IMUDevice()
        except Exception as exc:
            print(f"WARNING: IMU unavailable ({exc}); headings will be blank.")
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
        reference = None
        mat = ""
        with (session / "captures.csv").open("x", newline="") as log:
            writer = csv.DictWriter(log, fieldnames=FIELDS)
            writer.writeheader()
            while True:
                command = ask("command (z c l r a d x i m s n q)> ")
                if command == "q":
                    break
                if command == "z":
                    reference = read_heading(imu)
                    print("Heading unavailable." if reference is None
                          else f"Reference {reference:.2f} deg.")
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
                frame, meta = grab(front, front_bgr)
                side_frame = grab(side, side_bgr)[0] if side is not None else None
                heading = read_heading(imu)
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
            try:
                camera.stop()
            except Exception:
                pass
        for device in (front, side, imu):
            if device is not None:
                try:
                    device.close()
                except Exception:
                    pass
        signal.signal(signal.SIGTERM, previous)
    if session is not None:
        print(f"Saved {count} captures in {session}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
