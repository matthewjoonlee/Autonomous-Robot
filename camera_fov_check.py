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


def usable(area):
    return area if area and area[2] > 0 and area[3] > 0 else None


def full_area(camera, modes):
    """Largest crop rectangle the sensor allows, in sensor pixel coordinates."""
    candidates = [rect(camera.camera_properties.get("ScalerCropMaximum"))]
    limits = camera.camera_controls.get("ScalerCrop")
    if limits:
        candidates.append(rect(limits[1]))
    size = camera.camera_properties.get("PixelArraySize")
    if size:
        candidates.append((0, 0, int(size[0]), int(size[1])))
    candidates += [rect(m.get("crop_limits")) for m in modes]
    candidates = [a for a in candidates if usable(a)]
    return max(candidates, key=lambda a: a[2] * a[3]) if candidates else None


def widest_mode(modes):
    """Sensor mode covering the most sensor area; among ties, the smallest (fastest) output."""
    def coverage(mode):
        crop = rect(mode.get("crop_limits"))
        return crop[2] * crop[3] if crop else mode["size"][0] * mode["size"][1]
    modes = [m for m in modes if m.get("size")]
    if not modes:
        return None
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


def describe(name, metadata, configuration, area, lines, modes=()):
    sensor = configuration.get("sensor") or {}
    raw = configuration.get("raw") or {}
    mode_size = sensor.get("output_size", raw.get("size"))
    lines.append(f"  {name}: main {configuration['main']['size']}, sensor mode {mode_size}")
    crop = usable(rect(metadata.get("ScalerCrop")))
    if not crop:  # fall back to the chosen mode's crop limits, then the output aspect cut
        mode = next((m for m in modes if tuple(m.get("size", ())) == tuple(mode_size or ())), None)
        limits = usable(rect(mode.get("crop_limits"))) if mode else None
        if limits:
            out_w, out_h = configuration["main"]["size"]
            w, h = limits[2], limits[3]
            if out_w / out_h < w / h:
                w = h * out_w / out_h
            else:
                h = w * out_h / out_w
            crop = (0, 0, round(w), round(h))
    if crop and area:
        lines.append(f"    ScalerCrop {crop} of full {area}: "
                     f"{100 * crop[2] / area[2]:.0f}% width, {100 * crop[3] / area[3]:.0f}% height")
    else:
        lines.append(f"    ScalerCrop not reported (USB/UVC cameras usually do not); compare the images")


def check(index, info, session, cv2, np, Picamera2, lines):
    lines.append(f"Camera {index}: {info}")
    camera = Picamera2(index)
    try:
        try:
            modes = list(camera.sensor_modes)
        except Exception as exc:
            modes = []
            lines.append(f"  sensor_modes unavailable: {exc!r}")
        area = full_area(camera, modes)
        lines.append(f"  Full sensor area: {area}")
        for mode in modes:
            lines.append(f"  mode {mode}")

        current = camera.create_video_configuration(
            main={"size": CURRENT_SIZE, "format": "XRGB8888"}, queue=False)
        frame_a, meta_a, conf_a = capture(camera, current)
        describe("current setup", meta_a, conf_a, area, lines, modes)

        images = [frame_a]
        mode = widest_mode(modes) if modes else None
        if mode:
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
            describe("full FOV", meta_b, conf_b, area, lines, modes)
            images.append(frame_b)
        elif area:  # USB/UVC: capture at the camera's largest size, shrink to 640 wide
            full = camera.create_video_configuration(
                main={"size": (area[2], area[3]), "format": "XRGB8888"}, queue=False)
            frame_b, meta_b, conf_b = capture(camera, full)
            describe("largest size", meta_b, conf_b, area, lines)
            h = round(FULL_WIDTH * frame_b.shape[0] / frame_b.shape[1])
            images.append(cv2.resize(frame_b, (FULL_WIDTH, h), interpolation=cv2.INTER_AREA))
        else:
            lines.append("  No usable sensor modes; only the current setup was captured")

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
