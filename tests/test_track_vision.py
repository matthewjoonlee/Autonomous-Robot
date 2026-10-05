"""Saved-pose and adverse-image checks for the motor-free vision detector."""

import csv
from pathlib import Path
import unittest

import cv2
import numpy as np

from track_vision import DEFAULT_SESSION, detect_line, read_image, summarize


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


if __name__ == "__main__":
    unittest.main()
