"""Saved-pose and adverse-image checks for the motor-free vision detector."""

import csv
import itertools
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from track_vision import DEFAULT_SESSION, detect_line, live, read_image, summarize


class TrackVisionTests(unittest.TestCase):
    def test_original_centered_images_remain_detectable(self):
        root = Path(__file__).resolve().parents[1]
        for index in (1, 2, 3):
            with self.subTest(index=index):
                frame = read_image(root / f"snapshot_{index:03d}_front.jpg")
                self.assertEqual(detect_line(frame)["status"], "tracked")

    def test_labeled_physical_captures(self):
        reference = detect_line(read_image(DEFAULT_SESSION / "001_centered_straight.jpg"))
        reference["width"] = 640
        with (DEFAULT_SESSION / "captures.csv").open(newline="") as log:
            captures = list(csv.DictReader(log))
        results = {}
        for capture in captures:
            detection = detect_line(read_image(DEFAULT_SESSION / capture["image"]))
            results[capture["pose"]] = summarize(
                detection, reference, float(capture["delta_after_deg"]))
        self.assertEqual(results["centered_straight"]["diagnostic"], "NEAR REFERENCE")
        self.assertGreater(results["left_offset_straight"]["near_error_px"], 30)
        self.assertLess(results["right_offset_straight"]["near_error_px"], -30)
        self.assertEqual(results["centered_rotated_left"]["status"], "line_lost")
        self.assertIsNone(results["centered_rotated_left"]["near_error_px"])
        self.assertEqual(results["centered_rotated_right"]["diagnostic"], "ROTATED RIGHT")

    def test_blank_image_has_no_estimate(self):
        result = detect_line(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual(result["status"], "line_lost")
        self.assertIsNone(result["fit"])

    def test_horizontal_crossing_is_not_a_forward_line(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for x in range(40, 620, 90):
            cv2.rectangle(frame, (x, 400), (x + 45, 407), (0, 255, 255), -1)
        self.assertEqual(detect_line(frame)["status"], "line_lost")

    def test_background_yellow_is_excluded(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for y in range(30, 250, 30):
            cv2.rectangle(frame, (300, y), (309, y + 10), (0, 255, 255), -1)
        self.assertEqual(detect_line(frame)["status"], "line_lost")

    def test_two_competing_paths_are_ambiguous(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for x in (200, 450):
            for y in range(280, 460, 25):
                cv2.rectangle(frame, (x, y), (x + 9, y + 10), (0, 255, 255), -1)
        result = detect_line(frame)
        self.assertEqual(result["status"], "ambiguous")
        self.assertIsNone(result["fit"])

    def test_yellow_speckles_on_green_cannot_join_one_nearby_dash(self):
        hsv = np.full((480, 640, 3), (35, 105, 155), dtype=np.uint8)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        frame[420:] = 0
        # Deliberately collinear speckles with colors inside the yellow mask.
        for y in range(280, 420, 25):
            cv2.rectangle(frame, (300, y), (309, y + 10), (0, 255, 255), -1)
        cv2.rectangle(frame, (300, 440), (312, 455), (0, 255, 255), -1)
        self.assertEqual(detect_line(frame)["status"], "line_lost")

    def test_yellow_marks_on_bright_floor_are_rejected(self):
        frame = np.full((480, 640, 3), 190, dtype=np.uint8)
        for y in range(280, 460, 25):
            cv2.rectangle(frame, (300, y), (309, y + 10), (0, 255, 255), -1)
        self.assertEqual(detect_line(frame)["status"], "line_lost")

    def test_floor_gap_breaks_track_support(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for y in range(280, 460, 25):
            cv2.rectangle(frame, (300, y), (309, y + 10), (0, 255, 255), -1)
        frame[350:395] = 190
        self.assertEqual(detect_line(frame)["status"], "line_lost")

    def test_live_saves_exact_unannotated_pixels_and_local_reference(self):
        centered = read_image(DEFAULT_SESSION / "001_centered_straight.jpg")
        offset = read_image(DEFAULT_SESSION / "002_left_offset_straight.jpg")

        class FakeCamera:
            closed = stopped = False

            @staticmethod
            def global_camera_info():
                return [{"Model": "imx219"}]

            def __init__(self, index):
                self.frames = iter([centered, offset])

            def create_video_configuration(self, **kwargs):
                return kwargs

            def configure(self, config):
                pass

            def start(self):
                pass

            def capture_metadata(self):
                return {}

            def capture_array(self):
                return cv2.cvtColor(next(self.frames), cv2.COLOR_BGR2BGRA)

            def stop(self):
                FakeCamera.stopped = True

            def close(self):
                FakeCamera.closed = True

        class FakeIMU:
            SIGN = 1
            closed = False

            def heading(self):
                return 10.0

            def close(self):
                FakeIMU.closed = True

        camera_module = types.ModuleType("picamera2")
        camera_module.Picamera2 = FakeCamera
        imu_module = types.ModuleType("imu_controller")
        imu_module.IMUDevice = FakeIMU
        args = types.SimpleNamespace(camera=None, reference=None, duration=0.05, interval=0.001)
        clock = itertools.count(0, 0.01)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            with patch.dict(sys.modules, {"picamera2": camera_module, "imu_controller": imu_module}), \
                    patch("builtins.input", return_value=""), \
                    patch("track_vision.time.sleep"), \
                    patch("track_vision.time.monotonic", side_effect=lambda: next(clock)):
                live(args, None, output)
            np.testing.assert_array_equal(read_image(output / "reference_raw.png"), centered)
            with (output / "results.csv").open(newline="") as log:
                rows = list(csv.DictReader(log))
            self.assertEqual(len(rows), 1)
            np.testing.assert_array_equal(read_image(output / rows[0]["raw_image"]), offset)
            self.assertGreater(float(rows[0]["near_error_px"]), 30)
            self.assertTrue((output / rows[0]["image"]).is_file())
        self.assertTrue(FakeCamera.stopped and FakeCamera.closed and FakeIMU.closed)


if __name__ == "__main__":
    unittest.main()
