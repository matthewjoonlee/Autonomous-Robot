"""Calibrated centered-and-straight lane position, so runs need no reference capture.

The camera is fixed to the robot, so where the lane appears when the robot is
centered and pointing along the track is a property of the robot, not of each
run. `lane_center.json` stores it, plus how sideways offset and rotation move
the near and far rows differently, which lets the start-up frames measure the
robot's heading relative to the track from the image alone.

    python lane_center.py IMAGE [IMAGE ...]

recomputes the centered position as the median over images of the robot
centered and straight (rerun this if the camera mount moves).
"""

import argparse
import json
from pathlib import Path
import statistics
import sys

from track_vision import FAR_Y, NEAR_Y

CALIBRATION = Path(__file__).resolve().parent / "lane_center.json"


def load(path=CALIBRATION):
    return json.loads(Path(path).read_text())


def reference(calibration):
    """Reference in the same form as a detection, for `summarize`."""
    near, far = calibration["near_x"], calibration["far_x"]
    slope = (near - far) / (NEAR_Y - FAR_Y)
    return {"fit": (slope, near - slope * NEAR_Y), "width": calibration["width"]}


def image_yaw(fit, calibration, rotation_error_per_deg):
    """Robot heading relative to the track (degrees, positive right) from one lane fit.

    Turning right by one degree moves the near row by -rotation_error_per_deg
    (the controller's setting). Offset moves the far row `offset_far_ratio` as
    much as the near row while rotation moves it `rotation_far_ratio` as much,
    so far-minus-scaled-near depends only on rotation.
    """
    near = fit[0] * NEAR_Y + fit[1] - calibration["near_x"]
    far = fit[0] * FAR_Y + fit[1] - calibration["far_x"]
    ratio = calibration["offset_far_ratio"]
    per_deg = -rotation_error_per_deg * (calibration["rotation_far_ratio"] - ratio)
    return (far - ratio * near) / per_deg


def main(argv=None):
    from track_vision import detect_lane, read_image
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", type=Path, help="images with the robot centered and straight")
    parser.add_argument("--output", type=Path, default=CALIBRATION)
    args = parser.parse_args(argv)
    nears, fars, used = [], [], []
    for path in args.images:
        frame = read_image(path)
        detection = detect_lane(frame)
        if detection["fit"] is None:
            print(f"skipped (no lane): {path}", file=sys.stderr)
            continue
        slope, intercept = detection["fit"]
        nears.append(slope * NEAR_Y + intercept)
        fars.append(slope * FAR_Y + intercept)
        used.append(str(path))
    if len(nears) < 3:
        parser.error("need at least 3 images with a detected lane")
    calibration = load(args.output) if args.output.exists() else {
        "offset_far_ratio": 0.60, "rotation_far_ratio": 0.91}
    calibration.update(width=frame.shape[1], near_x=round(statistics.median(nears), 4),
                       far_x=round(statistics.median(fars), 4), center_images=used)
    args.output.write_text(json.dumps(calibration, indent=2) + "\n")
    spread = 640 * statistics.pstdev(nears)
    print(f"near_x {calibration['near_x']} far_x {calibration['far_x']} from {len(nears)} images "
          f"(near spread {spread:.1f} px) -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
