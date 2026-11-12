"""Check whether each camera's view is cropped, and capture a full-field-of-view comparison.

Run on the Pi with motor power off and no other camera program running:
    python camera_fov_check.py

For every camera it prints the sensor modes and how much of the sensor's area the
current setup (640x480 output, sensor mode left to Picamera2) uses, then repeats
the capture with the full-sensor mode at the matching aspect ratio. Each camera gets
a side-by-side image: current setup on the left, full FOV on the right.
Results go to fov_check/session_*/ (images + report.txt). No motor controller is opened.
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

CURRENT_SIZE = (640, 480)
FULL_WIDTH = 640


def rect(value):
    """Return (x, y, w, h) from a libcamera rectangle tuple, or None."""
    try:
        x, y, w, h = (int(v) for v in value)
        return x, y, w, h
    except (TypeError, ValueError):
        return None


def full_area(camera):
    """Largest crop rectangle the sensor allows, in sensor pixel coordinates."""
    for key in ("ScalerCropMaximum",):
        area = rect(camera.camera_properties.get(key))
        if area:
            return area
    limits = camera.camera_controls.get("ScalerCrop")
    if limits:
        area = rect(limits[1])
        if area:
            return area
    size = camera.camera_properties.get("PixelArraySize")
    return (0, 0, int(size[0]), int(size[1])) if size else None


def widest_mode(modes):
    """Sensor mode covering the most sensor area; among ties, the smallest (fastest) output."""
    def coverage(mode):
        crop = rect(mode.get("crop_limits"))
        return crop[2] * crop[3] if crop else mode["size"][0] * mode["size"][1]
    best = max(coverage(m) for m in modes)
    return min((m for m in modes if coverage(m) >= 0.98 * best),
               key=lambda m: m["size"][0] * m["size"][1])


def capture(camera, config, settle=2.0):
    camera.configure(config)
    camera.start()
    try:
        time.sleep(settle)
        frame = camera.capture_array("main")
        metadata = camera.capture_metadata()
        return frame, metadata, camera.camera_configuration()
    finally:
        camera.stop()


def describe(name, metadata, configuration, area, lines):
    sensor = configuration.get("sensor") or {}
    raw = configuration.get("raw") or {}
    lines.append(f"  {name}: main {configuration['main']['size']}, "
                 f"sensor mode {sensor.get('output_size', raw.get('size'))}")
    crop = rect(metadata.get("ScalerCrop"))
    if crop and area:
        lines.append(f"    ScalerCrop {crop} of full {area}: "
                     f"{100 * crop[2] / area[2]:.0f}% width, {100 * crop[3] / area[3]:.0f}% height")
    else:
        lines.append(f"    ScalerCrop not reported (USB/UVC cameras usually do not); compare the images")


def check(index, info, session, cv2, np, Picamera2, lines):
    lines.append(f"Camera {index}: {info}")
    camera = Picamera2(index)
    try:
        area = full_area(camera)
        modes = camera.sensor_modes
        lines.append(f"  Full sensor area: {area}")
        for mode in modes:
            lines.append(f"  mode {mode['size']} {mode.get('format')} fps {mode.get('fps', 0):.0f} "
                         f"crop_limits {mode.get('crop_limits')}")

        current = camera.create_video_configuration(
            main={"size": CURRENT_SIZE, "format": "XRGB8888"}, queue=False)
        frame_a, meta_a, conf_a = capture(camera, current)
        describe("current setup", meta_a, conf_a, area, lines)

        images = [frame_a]
        if modes:
            mode = widest_mode(modes)
            w, h = mode["size"]
            size = (FULL_WIDTH, round(FULL_WIDTH * h / w / 2) * 2)
            try:
                full = camera.create_video_configuration(
                    main={"size": size, "format": "XRGB8888"},
                    sensor={"output_size": mode["size"], "bit_depth": mode.get("bit_depth", 10)},
                    queue=False)
            except TypeError:  # older Picamera2 without the sensor= argument
                full = camera.create_video_configuration(
                    main={"size": size, "format": "XRGB8888"},
                    raw={"size": mode["size"], "format": mode.get("unpacked", mode["format"])},
                    queue=False)
            frame_b, meta_b, conf_b = capture(camera, full)
            describe("full FOV", meta_b, conf_b, area, lines)
            images.append(frame_b)
        else:
            lines.append("  No sensor modes listed; only the current setup was captured")

        bgr = [cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) if img.shape[2] == 4 else img for img in images]
        height = max(img.shape[0] for img in bgr)
        padded = [cv2.copyMakeBorder(img, 0, height - img.shape[0], 0, 8, cv2.BORDER_CONSTANT)
                  for img in bgr]
        for i, img in enumerate(bgr):
            cv2.imwrite(str(session / f"camera{index}_{'current' if i == 0 else 'full_fov'}.png"), img)
        cv2.imwrite(str(session / f"camera{index}_compare.png"), np.hstack(padded))
    finally:
        camera.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("fov_check"))
    args = parser.parse_args()

    import cv2
    import numpy as np
    from picamera2 import Picamera2

    session = args.output / datetime.now().strftime("session_%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=False)
    lines = []
    cameras = Picamera2.global_camera_info()
    if not cameras:
        print("No cameras found.")
        return 1
    for index, info in enumerate(cameras):
        try:
            check(index, info, session, cv2, np, Picamera2, lines)
        except Exception as exc:  # keep checking the other camera
            lines.append(f"  Camera {index} failed: {exc!r}")
        lines.append("")
    report = "\n".join(lines)
    print(report)
    (session / "report.txt").write_text(report + "\n")
    print(f"Saved to {session}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
