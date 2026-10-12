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

from track_vision import DEFAULT_SESSION, detect_lane, detect_line, live, read_image, summarize


class TrackVisionTests(unittest.TestCase):
    def test_blurred_course_failure_frame_is_tracked(self):
        root = Path(__file__).resolve().parents[1]
        run = root / "centering_results/run_20261011_184441_221716"
        reference = detect_line(read_image(run / "reference_raw.png"))
        reference["width"] = 640
        detection = detect_line(read_image(run / "raw/frame_00010.png"))
        self.assertEqual(detection["status"], "tracked")
        self.assertGreaterEqual(detection["confidence"], 0.65)
        self.assertLess(abs(summarize(detection, reference)["near_error_px"]), 8)

    def test_powered_failure_frame_retains_line_at_control_confidence(self):
        root = Path(__file__).resolve().parents[1]
        run = root / "centering_results/run_20261011_183304_621915"
        reference = detect_line(read_image(run / "reference_raw.png"))
        reference["width"] = 640
        detection = detect_line(read_image(run / "raw/frame_00015.png"))
        summary = summarize(detection, reference, 0)
        self.assertEqual(detection["status"], "tracked")
        self.assertGreaterEqual(detection["confidence"], 0.65)
        self.assertLess(abs(summary["near_error_px"]), 8)

    def test_compact_dash_cluster_without_vertical_coverage_is_rejected(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for y in range(390, 439, 7):
            cv2.rectangle(frame, (300, y), (309, y + 3), (0, 255, 255), -1)
        self.assertEqual(detect_line(frame)["status"], "line_lost")

    def test_sparse_compact_path_is_rejected(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for y in (350, 370, 390, 410, 430):
            cv2.rectangle(frame, (300, y), (309, y + 4), (0, 255, 255), -1)
        self.assertEqual(detect_line(frame)["status"], "line_lost")

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

    @staticmethod
    def synthetic_track(lane=(280, 400), dashes=None):
        """Green surround, dark lane, optional yellow dashes at one column."""
        frame = np.full((480, 640, 3), (40, 170, 60), dtype=np.uint8)
        frame[:, lane[0]:lane[1]] = 35
        if dashes is not None:
            for y in range(260, 480, 25):
                cv2.rectangle(frame, (dashes - 4, y), (dashes + 4, y + 10), (0, 255, 255), -1)
        return frame

    def test_lane_tracks_blurred_powered_frames(self):
        root = Path(__file__).resolve().parents[1]
        for run, frame_name in [("run_20261011_184441_221716", "frame_00010.png"),
                                ("run_20261011_183304_621915", "frame_00015.png")]:
            with self.subTest(run=run):
                reference = detect_lane(read_image(root / "centering_results" / run / "reference_raw.png"))
                reference["width"] = 640
                detection = detect_lane(read_image(root / "centering_results" / run / "raw" / frame_name),
                                        reference["fit"])
                self.assertEqual(detection["status"], "tracked")
                self.assertGreaterEqual(detection["confidence"], 0.5)
                self.assertLess(abs(summarize(detection, reference)["near_error_px"]), 8)

    def test_lane_tracks_heavy_motion_blur_rejected_by_dash_detector(self):
        frame = read_image(Path(__file__).resolve().parents[1]
                           / "vision_results/run_20261005_174624_462341/raw/frame_00022.png")
        self.assertEqual(detect_line(frame)["status"], "line_lost")
        detection = detect_lane(frame)
        self.assertEqual(detection["status"], "tracked")
        self.assertAlmostEqual(detection["fit"][0] * 0.85 + detection["fit"][1], 0.71, delta=0.02)

    def test_lane_matches_labeled_captures(self):
        reference = detect_lane(read_image(DEFAULT_SESSION / "001_centered_straight.jpg"))
        reference["width"] = 640
        errors = {}
        for name in ("002_left_offset_straight", "003_right_offset_straight", "005_centered_rotated_right"):
            detection = detect_lane(read_image(DEFAULT_SESSION / f"{name}.jpg"))
            self.assertEqual(detection["status"], "tracked", name)
            errors[name] = summarize(detection, reference)["near_error_px"]
        self.assertGreater(errors["002_left_offset_straight"], 30)
        self.assertLess(errors["003_right_offset_straight"], -30)
        # Rotated right: lane cut by the image border; dashes keep the center.
        self.assertLess(errors["005_centered_rotated_right"], -200)

    def test_lane_agrees_with_dash_fit_on_saved_runs(self):
        root = Path(__file__).resolve().parents[1]
        frames = sorted(root.glob("centering_results/run_*/raw/*.png"))
        self.assertGreater(len(frames), 20)
        for path in frames:
            frame = read_image(path)
            lane, line = detect_lane(frame), detect_line(frame)
            with self.subTest(frame=str(path.relative_to(root))):
                self.assertEqual(lane["status"], "tracked")
                if line["fit"] is not None:
                    difference = ((lane["fit"][0] - line["fit"][0]) * 0.85
                                  + lane["fit"][1] - line["fit"][1]) * 640
                    self.assertLess(abs(difference), 5)

    def test_lane_rejects_frames_without_a_lane(self):
        bright = np.full((480, 640, 3), 190, dtype=np.uint8)
        for name, frame in [("black", np.zeros((480, 640, 3), dtype=np.uint8)),
                            ("bright floor", bright), ("green only", self.synthetic_track((0, 0)))]:
            with self.subTest(name=name):
                self.assertEqual(detect_lane(frame)["status"], "line_lost")
                self.assertIsNone(detect_lane(frame)["fit"])

    def test_lane_ignores_crossing_band_and_distractor(self):
        frame = self.synthetic_track(dashes=340)
        frame[395:405] = 35                    # Full-width dark crossing band.
        frame[:, 40:110] = 35                  # Dark distractor strip far from the lane.
        for prior in (None, (0.0, 340 / 640)):
            with self.subTest(prior=prior):
                detection = detect_lane(frame, prior)
                self.assertEqual(detection["status"], "tracked")
                self.assertAlmostEqual(detection["fit"][1], 340 / 640, delta=0.005)
                self.assertAlmostEqual(detection["fit"][0], 0, delta=0.01)

    def test_lane_prior_rejects_jump_to_other_strip(self):
        frame = self.synthetic_track(lane=(40, 110))
        self.assertEqual(detect_lane(frame, (0.0, 340 / 640))["status"], "line_lost")
        self.assertEqual(detect_lane(frame)["status"], "tracked")

    def test_half_visible_lane_uses_dashes(self):
        detection = detect_lane(self.synthetic_track(lane=(0, 260), dashes=100))
        self.assertEqual(detection["status"], "tracked")
        self.assertAlmostEqual(detection["fit"][1] * 640, 100, delta=3)
        self.assertEqual(detect_lane(self.synthetic_track(lane=(0, 260)))["status"], "line_lost")

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
