"""Save labeled side-camera photos for comparing mount angles and track poses.

Run on the Pi: python capture_side_camera.py
No display, IMU, or motor controller is required or opened.
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import re
import shutil
import signal
import sys
import time


def choose_side(cameras):
    """Use the USB/UVC side camera, requiring an override if ambiguous."""
    matches = [i for i, info in enumerate(cameras)
               if any(word in str(info).lower() for word in ("usb", "uvc"))]
    if len(matches) == 1:
        return matches[0]
    raise RuntimeError("Cannot uniquely identify the USB side camera. "
                       "Rerun with --camera INDEX using the list above.")


def slug(label):
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:60] or "position"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, help="Picamera2 camera index override")
    parser.add_argument("--output", type=Path, default=Path("side_camera_positions"))
    args = parser.parse_args()
    if args.camera is not None and args.camera < 0:
        parser.error("--camera must be nonnegative")

    camera = session = None
    started = False
    count = 0
    result = 0
    previous_handler = signal.getsignal(signal.SIGTERM)

    def terminate(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        import cv2
        from picamera2 import Picamera2
        from camera_setup import side_bgr, side_config

        cameras = Picamera2.global_camera_info()
        for index, info in enumerate(cameras):
            print(f"Camera {index}: {info}")
        index = args.camera if args.camera is not None else choose_side(cameras)
        if index >= len(cameras):
            raise RuntimeError(f"Camera index {index} is not in the discovered camera list")
        print(f"Using side camera {index}")
        camera = Picamera2(index)
        camera.configure(side_config(camera))
        camera.start()
        started = True
        time.sleep(2)

        session = args.output / datetime.now().strftime("session_%Y%m%d_%H%M%S_%f")
        session.mkdir(parents=True, exist_ok=False)
        print(f"Saving to {session}")
        print("Stop other camera/motor programs first. Move the robot by hand.")
        print("Use the same track locations for each mount angle.")
        print("Commands: Enter/c = capture, m = change mount label, q = finish.")
        print("Suggested poses: centered straight, left/right offset, before turn,")
        print("quarter turn, halfway through turn, aligned after turn.")
        mount = ""
        fields = ["timestamp", "image", "mount", "pose", "note", "camera_index",
                  "sensor_timestamp_ns", "exposure_time_us", "analogue_gain"]
        with (session / "captures.csv").open("x", newline="") as log:
            writer = csv.DictWriter(log, fieldnames=fields)
            writer.writeheader()
            log.flush()
            while True:
                if not mount:
                    mount = input("Mount label (e.g. straight_down or tilted_out_15deg)> ").strip()
                    if not mount:
                        continue
                command = input(f"[{mount}] Enter to capture / m / q> ").strip().lower()
                if command == "q":
                    break
                if command == "m":
                    mount = ""
                    continue
                if command not in ("", "c"):
                    print("Use Enter/c, m, or q.")
                    continue
                pose = input("Robot pose (e.g. centered_straight, halfway_turn)> ").strip()
                if not pose:
                    print("Please label the pose so we can compare it later.")
                    continue
                note = input("Notes: height, tilt, distance from line (Enter to skip)> ").strip()
                input("Hold still; press Enter when ready to take the photo> ")
                time.sleep(0.5)
                request = camera.capture_request()
                try:
                    frame = request.make_array("main")
                    metadata = request.get_metadata()
                    frame_bgr = side_bgr(frame)  # full view, shrunk to 640x480
                finally:
                    request.release()
                name = f"{count + 1:03d}_{slug(mount)}_{slug(pose)}.png"
                if not cv2.imwrite(str(session / name), frame_bgr):
                    raise RuntimeError(f"Could not save {name}")
                writer.writerow(dict(zip(fields, [
                    datetime.now().astimezone().isoformat(), name, mount, pose, note,
                    index, metadata.get("SensorTimestamp"), metadata.get("ExposureTime"),
                    metadata.get("AnalogueGain"),
                ])))
                log.flush()
                count += 1
                print(f"Saved {name}")
    except (KeyboardInterrupt, EOFError):
        print("\nCapture stopped; completed photos are saved.")
    except Exception as exc:
        print(f"Capture failed: {exc}", file=sys.stderr)
        result = 1
    finally:
        if camera is not None:
            for action in ([camera.stop] if started else []) + [camera.close]:
                try:
                    action()
                except Exception as exc:
                    print(f"Cleanup warning: {exc}", file=sys.stderr)
        signal.signal(signal.SIGTERM, previous_handler)

    if session is not None:
        print(f"Saved {count} photos in {session.resolve()}")
        if count:
            try:
                archive = shutil.make_archive(str(session), "zip", root_dir=session)
                print(f"Share this ZIP for review: {Path(archive).resolve()}")
            except Exception as exc:
                print(f"ZIP creation failed: {exc}. Share the session folder instead.", file=sys.stderr)
                result = 1
    return result


if __name__ == "__main__":
    sys.exit(main())
