"""Capture labeled stationary front-camera poses for Quiz 3; no motor commands."""

import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import signal
import sys
import time

from test_imu import signed_delta


POSES = {
    "c": "centered_straight",
    "l": "left_offset_straight",
    "r": "right_offset_straight",
    "a": "centered_rotated_left",
    "d": "centered_rotated_right",
    "x": "combined_offset_rotation",
}


def choose_front(cameras):
    """Match the existing display tool's non-USB front-camera convention."""
    for index, info in enumerate(cameras):
        description = " ".join(str(value).lower() for value in info.values())
        if "usb" not in description and "uvc" not in description:
            return index
    raise RuntimeError("No non-USB front camera found. Use --camera with its listed index.")


def read_heading(imu):
    value = imu.heading()
    return value if value is not None and math.isfinite(value) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, help="override the front camera index")
    parser.add_argument("--output", type=Path, default=Path("track_calibration"))
    args = parser.parse_args()
    if args.camera is not None and args.camera < 0:
        parser.error("--camera must be nonnegative")

    camera = imu = None
    started = False
    previous_handler = signal.getsignal(signal.SIGTERM)

    def terminate(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        import cv2
        from picamera2 import Picamera2
        from imu_controller import IMUDevice

        cameras = Picamera2.global_camera_info()
        for index, info in enumerate(cameras):
            print(f"Camera {index}: {info}")
        camera_index = args.camera if args.camera is not None else choose_front(cameras)
        camera = Picamera2(camera_index)
        camera.configure(camera.create_video_configuration(
            main={"size": (640, 480), "format": "XRGB8888"}, queue=False))
        camera.start()
        started = True
        imu = IMUDevice()
        time.sleep(1.5)
        metadata = camera.capture_metadata()
        if "ColourGains" in metadata:
            camera.set_controls({"AwbEnable": False, "ColourGains": metadata["ColourGains"]})

        session = args.output / datetime.now().strftime("session_%Y%m%d_%H%M%S_%f")
        session.mkdir(parents=True, exist_ok=False)
        print(f"Saving to {session}")
        print("No motor commands. Stop other camera/control programs before using this tool.")
        print("Keep the mount fixed and hold each pose still before capturing.")
        print("First align centered and straight; enter z to set the IMU reference.")
        print("Capture: c=centered straight, l=left offset straight, r=right offset straight,")
        print("a=centered rotated left, d=centered rotated right, x=combined pose.")
        print("Use q or Ctrl+C to quit. Enter commands after positioning the robot.")
        reference = None
        count = 0
        with (session / "captures.csv").open("x", newline="") as log:
            fields = ["timestamp", "image", "pose", "note", "camera_index",
                      "reference_deg", "heading_before_deg", "heading_after_deg",
                      "delta_after_deg", "capture_start_monotonic_s",
                      "capture_end_monotonic_s", "sensor_timestamp_ns",
                      "exposure_time_us", "imu_status"]
            writer = csv.DictWriter(log, fieldnames=fields)
            writer.writeheader()
            log.flush()
            while True:
                command = input("Pose command> ").strip().lower()
                if command == "q":
                    break
                if command == "z":
                    heading = read_heading(imu)
                    if heading is None:
                        print("Heading unavailable; zero not changed. Try again.")
                    else:
                        reference = heading
                        print(f"Reference: {reference:.2f} degrees. Keep this reference for all poses.")
                    continue
                if command not in POSES:
                    print("Use z, c, l, r, a, d, x, or q.")
                    continue
                if reference is None:
                    print("Align centered and straight and enter z first.")
                    continue
                note = input("Approximate offset/angle or other note (Enter to skip)> ").strip()
                heading_before = read_heading(imu)
                capture_start = time.monotonic()
                request = camera.capture_request()
                try:
                    frame = request.make_array("main")
                    metadata = request.get_metadata()
                finally:
                    request.release()
                heading_after = read_heading(imu)
                capture_end = time.monotonic()
                # Same four-channel front-camera conversion as test_display.py.
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                name = f"{count + 1:03d}_{POSES[command]}.jpg"
                if not cv2.imwrite(str(session / name), frame_bgr):
                    raise RuntimeError(f"Could not save {name}")
                delta = (signed_delta(heading_after, reference, imu.SIGN)
                         if heading_after is not None else None)
                writer.writerow(dict(zip(fields, [
                    datetime.now().astimezone().isoformat(), name, POSES[command], note,
                    camera_index, reference, heading_before, heading_after, delta,
                    capture_start, capture_end, metadata.get("SensorTimestamp"),
                    metadata.get("ExposureTime"),
                    "ok" if heading_before is not None and heading_after is not None else "unavailable",
                ])))
                log.flush()
                count += 1
                delta_text = "unavailable" if delta is None else f"{delta:+.2f} degrees"
                print(f"Saved {name}; relative heading {delta_text}")
                if heading_before is None or heading_after is None:
                    print("IMU reading missing; repeat this pose when readings recover.")
            print(f"Saved {count} captures. Upload the whole folder: {session}")
    except (KeyboardInterrupt, EOFError):
        print("\nCalibration stopped; completed captures are saved.")
    except Exception as exc:
        print(f"Calibration failed: {exc}", file=sys.stderr)
        return 1
    finally:
        # Attempt every cleanup even if one device reports an error.
        for action in ([imu.close] if imu is not None else []) + (
                ([camera.stop] if started else []) + [camera.close] if camera is not None else []):
            try:
                action()
            except Exception as exc:
                print(f"Cleanup warning: {exc}", file=sys.stderr)
        signal.signal(signal.SIGTERM, previous_handler)
    return 0


if __name__ == "__main__":
    sys.exit(main())
