"""Calibrated lane center and image-based heading, replacing the reference capture."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from lane_center import CALIBRATION, image_yaw, load, main, reference
from quiz3_autonomous import Settings
from track_vision import DEFAULT_SESSION, FAR_Y, NEAR_Y, detect_lane, read_image

ROTATION = Settings().rotation_error_per_deg


def fit(name):
    return detect_lane(read_image(DEFAULT_SESSION / f"{name}.jpg"))["fit"]


def lateral_px(lane_fit, calibration):
    yaw = image_yaw(lane_fit, calibration, ROTATION)
    near = lane_fit[0] * NEAR_Y + lane_fit[1] - calibration["near_x"]
    return (near + ROTATION * yaw) * calibration["width"]


class LaneCenterTests(unittest.TestCase):
    def test_image_heading_matches_imu_on_labeled_captures(self):
        calibration = load()
        for name, imu in (("001_centered_straight", 0.0), ("002_left_offset_straight", -0.875),
                          ("003_right_offset_straight", -0.0625), ("005_centered_rotated_right", 13.125)):
            with self.subTest(name=name):
                self.assertAlmostEqual(image_yaw(fit(name), calibration, ROTATION), imu, delta=0.6)

    def test_rotation_is_not_mistaken_for_offset(self):
        calibration = load()
        self.assertLess(abs(lateral_px(fit("005_centered_rotated_right"), calibration)), 10)
        self.assertGreater(lateral_px(fit("002_left_offset_straight"), calibration), 30)
        self.assertLess(lateral_px(fit("003_right_offset_straight"), calibration), -30)

    def test_reference_reproduces_calibrated_rows(self):
        calibration = load()
        slope, intercept = reference(calibration)["fit"]
        self.assertAlmostEqual(slope * NEAR_Y + intercept, calibration["near_x"])
        self.assertAlmostEqual(slope * FAR_Y + intercept, calibration["far_x"])

    def test_cli_recomputes_center_and_keeps_ratios(self):
        images = [DEFAULT_SESSION / f"{name}.jpg" for name in
                  ("001_centered_straight", "002_left_offset_straight", "003_right_offset_straight")]
        nears = sorted(f[0] * NEAR_Y + f[1] for f in (fit(path.stem) for path in images))
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "center.json"
            shutil.copy(CALIBRATION, output)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([*map(str, images), "--output", str(output)]), 0)
            saved = json.loads(output.read_text())
        self.assertAlmostEqual(saved["near_x"], nears[1], places=4)
        self.assertEqual(saved["offset_far_ratio"], load()["offset_far_ratio"])
        self.assertEqual(len(saved["center_images"]), 3)


if __name__ == "__main__":
    unittest.main()
