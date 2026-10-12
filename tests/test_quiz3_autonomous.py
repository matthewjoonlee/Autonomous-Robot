"""Feedback-direction and bounded-loop checks before physical centering tests."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import cv2
import numpy as np

from quiz3_autonomous import (LaneTracker, Settings, choose_command, configure_motion_exposure,
                              control_loop, measure_start)
from test_imu import signed_delta
from track_vision import DEFAULT_SESSION, detect_lane, read_image


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Camera:
    def __init__(self, frames, clock, delay=0):
        self.frames = iter(frames)
        self.clock, self.delay = clock, delay

    def capture_array(self):
        self.clock.sleep(self.delay)
        return cv2.cvtColor(next(self.frames), cv2.COLOR_BGR2BGRA)


class IMU:
    SIGN = 1

    def __init__(self, values):
        self.values = iter(values)

    def heading(self):
        value = next(self.values)
        if isinstance(value, BaseException):
            raise value
        return value


class Recorder:
    def __init__(self):
        self.ids = []

    def add(self, frame_id, frame):
        self.ids.append(frame_id)


class Motors:
    def __init__(self, clock):
        self.clock = clock
        self.commands, self.stops = [], []

    def drive(self, speed, turn):
        self.commands.append((self.clock(), speed, turn))

    def stop(self):
        self.stops.append(self.clock())


def command(lateral=0.0, heading=0.0, lost_for=0.0, previous_turn=0.0, dt=1.0, settings=Settings()):
    return choose_command(lateral, heading, lost_for, previous_turn, dt, settings)


class CenteringTests(unittest.TestCase):
    def test_physical_right_maps_to_negative_motor_command(self):
        result = command(0.1)
        self.assertGreater(result["physical_turn"], 0)
        self.assertLess(result["raw_turn"], 0)

    def test_physical_left_maps_to_positive_motor_command(self):
        result = command(-0.1)
        self.assertLess(result["physical_turn"], 0)
        self.assertGreater(result["raw_turn"], 0)

    def test_heading_feedback_straightens_at_reference(self):
        self.assertGreater(command(heading=3)["raw_turn"], 0)
        self.assertLess(command(heading=-3)["raw_turn"], 0)
        self.assertEqual(command(0.001)["raw_turn"], 0)

    def test_commands_are_limited(self):
        result = command(0.8, -4)
        self.assertEqual(result["target_heading_deg"], 12)
        self.assertEqual(result["raw_turn"], -20)
        self.assertEqual(result["speed"], 25)

    def test_small_corrections_are_lifted_past_motor_dead_zone(self):
        small = command(0.004)
        self.assertTrue(0.5 <= small["turn"] < 1.5)
        self.assertEqual(small["physical_turn"], 5)
        self.assertEqual(command(0.004, settings=replace(Settings(), turn_deadzone=0))["physical_turn"], 1)
        self.assertEqual(command(-0.004)["physical_turn"], -5)
        self.assertEqual(command(0.8, heading=-1)["physical_turn"], 20)

    def test_turn_changes_at_bounded_rate(self):
        self.assertAlmostEqual(command(0.8, dt=0.04)["turn"], 1.6)
        self.assertAlmostEqual(command(0.8, previous_turn=9.0, dt=0.04)["turn"], 10.6)
        self.assertAlmostEqual(command(0.8, heading=-4, previous_turn=19.0, dt=0.04)["turn"], 20)
        self.assertAlmostEqual(command(-0.8, previous_turn=5.0, dt=0.04)["turn"], 3.4)

    def test_keeps_driving_and_steering_back_while_lane_is_unseen(self):
        # Nudged 12 deg left with the lane last seen to the right: full speed, steer right.
        result = command(0.03, heading=-12, lost_for=2.5)
        self.assertEqual(result["speed"], 25)
        self.assertEqual(result["physical_turn"], 20)
        with self.assertRaisesRegex(RuntimeError, "Lane not found"):
            command(lost_for=3.0)

    def test_nudge_sized_rotation_does_not_abort(self):
        self.assertEqual(command(heading=-20)["physical_turn"], 20)
        self.assertEqual(command(heading=30)["physical_turn"], -20)
        # Normal tracking stays gentle: a 1 degree heading error uses well under 10.
        self.assertLess(abs(command(heading=1)["physical_turn"]), 10)

    def test_optional_slowdown_during_large_turns(self):
        settings = replace(Settings(), turn_slowdown=0.2)
        self.assertEqual(command(0.8, heading=-4, settings=settings)["speed"], 20)
        self.assertEqual(command(settings=settings)["speed"], 25)

    def test_faults_do_not_produce_commands(self):
        for lateral, heading, lost_for in [
            (None, 0, 0), (float("nan"), 0, 0), (0, None, 0), (0, 45, 0), (0, -45, 0), (0, 0, 5),
        ]:
            with self.subTest(lateral=lateral, heading=heading, lost_for=lost_for):
                with self.assertRaises(RuntimeError):
                    command(lateral, heading, lost_for)

    def test_tracker_removes_rotation_and_holds_estimate_while_unseen(self):
        settings = Settings()
        tracker = LaneTracker({"fit": (0.1, 0.6)}, 0.0, settings)
        tracked = {"status": "tracked", "confidence": 0.9, "fit": (0.1, 0.65)}
        lost = {"status": "line_lost", "confidence": 0.0, "fit": None}
        # Pure rotation: 10 deg left shifts the lane right but is not a sideways offset.
        self.assertTrue(tracker.update(tracked, 10 * settings.rotation_error_per_deg, -10, 0.04))
        self.assertAlmostEqual(tracker.lateral, 0)
        self.assertTrue(tracker.update(tracked, 0.04, 0, 0.08))
        self.assertTrue(tracker.update(tracked, 0.04, 0, 0.12))
        self.assertAlmostEqual(tracker.lateral, 0.03)
        self.assertFalse(tracker.update(lost, None, -5, 0.16))
        self.assertAlmostEqual(tracker.lateral, 0.03)  # Held, so steering keeps toward the lane.
        self.assertFalse(tracker.update(dict(tracked, confidence=0.3), 0.5, 0, 0.2))

    def test_search_window_follows_rotation(self):
        settings = Settings()
        tracker = LaneTracker({"fit": (0.1, 0.6)}, 0.0, settings)
        self.assertIsNone(tracker.prior(0.0, 0))  # Robot may have moved after the reference.
        tracker.update({"status": "tracked", "confidence": 0.9, "fit": (0.1, 0.65)}, 0.0, 0, 0.04)
        slope, intercept = tracker.prior(0.08, -2)
        self.assertEqual(slope, 0.1)
        self.assertAlmostEqual(intercept, 0.65 + 2 * settings.rotation_error_per_deg)
        self.assertIsNone(tracker.prior(0.5, 0))

    def setup_loop(self, frames, headings, delay=0):
        clock = Clock()
        reference_frame = read_image(DEFAULT_SESSION / "001_centered_straight.jpg")
        reference = detect_lane(reference_frame)
        reference["width"] = 640
        return (Camera(frames, clock, delay), IMU(headings), Motors(clock),
                reference, clock, [], Recorder())

    def test_transient_blur_keeps_driving(self):
        good = read_image(DEFAULT_SESSION / "001_centered_straight.jpg")
        bad = np.zeros_like(good)
        camera, imu, motors, reference, clock, rows, frames = self.setup_loop(
            [good, bad, bad, good, good, good], [0] * 6)
        control_loop(camera, imu, motors, reference, 0, 0.18, Settings(), rows, frames, clock, clock.sleep)
        self.assertEqual([row["state"] for row in rows[:5]],
                         ["driving", "searching", "searching", "driving", "driving"])
        self.assertEqual([speed for _, speed, _ in motors.commands], [25] * 5)
        self.assertEqual(len(motors.stops), 1)  # Only the final stop.
        self.assertEqual(frames.ids, [row["frame_id"] for row in rows])  # Every frame recorded.

    def test_nudge_off_the_lane_steers_back_then_reacquires(self):
        good = read_image(DEFAULT_SESSION / "001_centered_straight.jpg")
        bad = np.zeros_like(good)
        sequence = [good] * 3 + [bad] * 10 + [good] * 3
        headings = [0, 0, 0] + [-12] * 10 + [-2] * 3
        camera, imu, motors, reference, clock, rows, frames = self.setup_loop(sequence, headings)
        control_loop(camera, imu, motors, reference, 0, 0.04 * len(sequence) - 0.02, Settings(),
                     rows, frames, clock, clock.sleep)
        searching = [row for row in rows if row["state"] == "searching"]
        self.assertEqual(len(searching), 10)
        self.assertTrue(all(row["speed"] == 25 for row in searching))
        self.assertGreater(searching[-1]["physical_turn"], 4)  # Turning right, back toward the lane.
        self.assertEqual(rows[-1]["state"], "driving")
        self.assertEqual(len(motors.stops), 1)

    def test_half_off_road_start_finds_lane_immediately_and_steers_back(self):
        good = read_image(DEFAULT_SESSION / "001_centered_straight.jpg")
        # Robot moved left after the centered reference: the lane appears ~150 px right.
        shifted = cv2.warpAffine(good, np.float32([[1, 0, 150], [0, 1, 0]]), (640, 480),
                                 borderMode=cv2.BORDER_REPLICATE)
        camera, imu, motors, reference, clock, rows, frames = self.setup_loop([shifted] * 6, [0] * 6)
        control_loop(camera, imu, motors, reference, 0, 0.22, Settings(), rows, frames, clock, clock.sleep)
        self.assertTrue(all(row["state"] == "driving" for row in rows[:5]))
        self.assertGreater(rows[0]["near_error_px"], 120)
        turns = [row["physical_turn"] for row in rows[:5]]
        self.assertTrue(all(turn >= 0 for turn in turns) and max(turns) > 4)  # Right, toward the lane.
        self.assertEqual(max(row["target_heading_deg"] for row in rows[:5]), 12)

    def test_start_up_sets_imu_zero_from_image_heading(self):
        from lane_center import load
        rotated = read_image(DEFAULT_SESSION / "005_centered_rotated_right.jpg")
        for sign in (1, -1):
            with self.subTest(sign=sign):
                clock = Clock()
                imu = IMU([350.0 + 0.0625 * (i % 2) for i in range(10)])
                imu.SIGN = sign
                zero, start = measure_start(Camera([rotated] * 10, clock), imu, load(), Settings(),
                                            clock=clock, sleep=clock.sleep)
                # Placed ~13 deg right of the track: relative heading must start there.
                self.assertAlmostEqual(signed_delta(350.0, zero, sign), start["yaw_deg"], delta=0.1)
                self.assertAlmostEqual(start["yaw_deg"], 13.1, delta=0.6)
                self.assertLess(abs(start["lateral_px"]), 10)
                self.assertEqual(start["lane_frames"], 10)

    def test_heading_follows_image_when_imu_drifts(self):
        from lane_center import load
        good = read_image(DEFAULT_SESSION / "001_centered_straight.jpg")
        # IMU heading jumps 5 deg wrong (real drift was ~0.5 deg/s); the camera sees it straight.
        drift = [0.0] * 5 + [5.0] * 36
        camera, imu, motors, reference, clock, rows, frames = self.setup_loop([good] * 41, drift)
        control_loop(camera, imu, None, reference, 0, 0.04 * 41 - 0.02, Settings(), rows, frames,
                     clock, clock.sleep, calibration=load())
        last = rows[-1]
        self.assertEqual(float(last["imu_delta_deg"]), 5.0)
        self.assertAlmostEqual(last["heading_deg"], last["image_yaw_deg"], delta=1.0)
        self.assertLess(abs(last["lateral_error"] * 640), 15)  # Drift is not mistaken for offset.
        # Without the calibration the controller trusts the drifting IMU.
        camera, imu, motors, reference, clock, rows, frames = self.setup_loop([good] * 41, drift)
        control_loop(camera, imu, None, reference, 0, 0.04 * 41 - 0.02, Settings(), rows, frames,
                     clock, clock.sleep)
        self.assertEqual(rows[-1]["heading_deg"], 5.0)

    def test_start_up_refuses_without_lane(self):
        from lane_center import load
        good = read_image(DEFAULT_SESSION / "001_centered_straight.jpg")
        clock = Clock()
        frames = [np.zeros_like(good)] * 7 + [good] * 3
        with self.assertRaisesRegex(RuntimeError, "Lane not visible at start"):
            measure_start(Camera(frames, clock), IMU([0.0] * 10), load(), Settings(),
                          clock=clock, sleep=clock.sleep)

    def test_keep_searching_never_ends_leg_for_lost_lane(self):
        good = read_image(DEFAULT_SESSION / "001_centered_straight.jpg")
        bad = np.zeros_like(good)
        camera, imu, motors, reference, clock, rows, frames = self.setup_loop([good] + [bad] * 150, [0] * 151)
        settings = replace(Settings(), lost_limit=float("inf"))
        reason = control_loop(camera, imu, motors, reference, 0, 5.5, settings, rows, frames, clock, clock.sleep)
        self.assertEqual(reason, "duration_complete")
        self.assertGreater(rows[-2]["lost_s"], 5)
        self.assertTrue(all(speed == 25 for _, speed, _ in motors.commands))

    def test_persistent_loss_searches_then_faults(self):
        good = read_image(DEFAULT_SESSION / "001_centered_straight.jpg")
        bad = np.zeros_like(good)
        camera, imu, motors, reference, clock, rows, frames = self.setup_loop(
            [good] + [bad] * 90, [0] * 91)
        with self.assertRaisesRegex(RuntimeError, "Lane not found"):
            control_loop(camera, imu, motors, reference, 0, 5, Settings(), rows, frames, clock, clock.sleep)
        self.assertTrue(all(speed == 25 for _, speed, _ in motors.commands))
        self.assertTrue(all(t < 3.0 for t, _, _ in motors.commands))
        self.assertTrue(motors.stops)
        self.assertEqual(rows[-1]["state"], "Lane not found; run stopped")

    def test_estimated_distance_ends_run(self):
        good = read_image(DEFAULT_SESSION / "001_centered_straight.jpg")
        camera, imu, motors, reference, clock, rows, frames = self.setup_loop([good] * 20, [0] * 20)
        reason = control_loop(camera, imu, motors, reference, 0, 5, Settings(), rows, frames, clock,
                              clock.sleep, max_distance_cm=5, speed_cm_s=14)
        self.assertEqual(reason, "estimated_distance")
        self.assertEqual(rows[-1]["state"], "estimated_distance")
        self.assertGreaterEqual(rows[-1]["distance_cm"], 5)
        self.assertLess(rows[-2]["distance_cm"], 5)
        self.assertEqual(len(motors.commands), len(rows) - 1)
        self.assertTrue(motors.stops)

    def test_distance_estimate_includes_motor_command_time(self):
        class SlowMotors(Motors):
            def drive(self, speed, turn):
                super().drive(speed, turn)
                self.clock.sleep(0.015)  # Serial writes took ~1/3 of each cycle on the Pi.

        good = read_image(DEFAULT_SESSION / "001_centered_straight.jpg")
        camera, imu, _, reference, clock, rows, frames = self.setup_loop([good] * 30, [0] * 30)
        control_loop(camera, imu, SlowMotors(clock), reference, 0, 1.0, Settings(), rows, frames, clock,
                     clock.sleep, max_distance_cm=1000, speed_cm_s=14)
        driven = rows[-1]["elapsed_s"] - rows[0]["elapsed_s"]
        self.assertAlmostEqual(rows[-1]["distance_cm"], 14 * driven, delta=0.5)

    def test_manual_exposure_gain_is_limited_and_logged(self):
        class ExposureCamera:
            camera_controls = {"AeEnable": (False, True, None), "ExposureTime": (100, 40000, 10000),
                               "AnalogueGain": (1.0, 16.0, 1.0)}

            def __init__(self):
                self.requested = None

            def capture_metadata(self):
                return {"ExposureTime": 15000, "AnalogueGain": 4.0, "DigitalGain": 1.0}

            def set_controls(self, requested):
                self.requested = requested

        camera = ExposureCamera()
        with patch("quiz3_autonomous.time.sleep"):
            result = configure_motion_exposure(camera, 5000)
        self.assertEqual(camera.requested["ExposureTime"], 5000)
        self.assertEqual(camera.requested["AnalogueGain"], 8)
        self.assertFalse(camera.requested["AeEnable"])
        self.assertTrue(result["gain_limited"])
        self.assertEqual(result["observed"]["ExposureTime"], 15000)

    def test_auto_exposure_override_does_not_set_controls(self):
        class AutoCamera:
            def capture_metadata(self):
                return {"ExposureTime": 33000}

            def set_controls(self, requested):
                raise AssertionError("auto mode should not set manual controls")

        self.assertEqual(configure_motion_exposure(AutoCamera(), 0)["mode"], "auto")

    def test_missing_imu_and_interrupt_stop(self):
        good = read_image(DEFAULT_SESSION / "001_centered_straight.jpg")
        for value, expected in [(None, RuntimeError), (KeyboardInterrupt(), KeyboardInterrupt)]:
            with self.subTest(value=value):
                camera, imu, motors, reference, clock, rows, frames = self.setup_loop([good, good], [0, value])
                with self.assertRaises(expected):
                    control_loop(camera, imu, motors, reference, 0, 1, Settings(), rows, frames, clock, clock.sleep)
                self.assertEqual(len(motors.commands), 1)
                self.assertTrue(motors.stops)

    def test_stale_capture_cannot_start_motion(self):
        good = read_image(DEFAULT_SESSION / "001_centered_straight.jpg")
        camera, imu, motors, reference, clock, rows, frames = self.setup_loop([good], [0], delay=0.16)
        with self.assertRaisesRegex(RuntimeError, "delayed"):
            control_loop(camera, imu, motors, reference, 0, 1, Settings(), rows, frames, clock, clock.sleep)
        self.assertFalse(motors.commands)
        self.assertTrue(motors.stops)

    def test_deadline_stops_and_default_motor_free_loop(self):
        good = read_image(DEFAULT_SESSION / "001_centered_straight.jpg")
        for powered in (False, True):
            with self.subTest(powered=powered):
                camera, imu, motors, reference, clock, rows, frames = self.setup_loop([good] * 10, [0] * 10)
                control_loop(camera, imu, motors if powered else None, reference, 0, 0.12,
                             Settings(), rows, frames, clock, clock.sleep)
                if powered:
                    self.assertTrue(motors.stops)
                    self.assertTrue(all(t < 0.12 for t, _, _ in motors.commands))
                else:
                    self.assertFalse(motors.commands)
                    self.assertFalse(motors.stops)
                    self.assertTrue(all(row["state"] == "driving" for row in rows))


class MainFlowTests(unittest.TestCase):
    def test_motor_free_run_needs_no_reference_capture(self):
        import itertools
        import json
        import sys
        import tempfile
        import types
        from pathlib import Path
        from quiz3_autonomous import main

        frame = read_image(DEFAULT_SESSION / "002_left_offset_straight.jpg")

        class FakeCamera:
            @staticmethod
            def global_camera_info():
                return [{"Model": "imx219"}]

            def __init__(self, index):
                pass

            def create_video_configuration(self, **kwargs):
                return kwargs

            def configure(self, config):
                pass

            def start(self):
                pass

            def capture_metadata(self):
                return {}

            def capture_array(self):
                return cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

            def stop(self):
                pass

            def close(self):
                pass

        class FakeIMU:
            SIGN = 1

            def heading(self):
                return 200.0

            def close(self):
                pass

        camera_module = types.ModuleType("picamera2")
        camera_module.Picamera2 = FakeCamera
        imu_module = types.ModuleType("imu_controller")
        imu_module.IMUDevice = FakeIMU
        prompts = []
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict(sys.modules, {"picamera2": camera_module, "imu_controller": imu_module}), \
                    patch("builtins.input", side_effect=lambda prompt="": prompts.append(prompt) or ""), \
                    patch("quiz3_autonomous.read_command", return_value=None), \
                    patch("builtins.print"), patch("quiz3_autonomous.time.sleep"):
                result = main(["--legs", "1", "--leg-ft", "0.25", "--exposure-us", "0", "--output", folder])
            run = next(Path(folder).glob("run_*"))
            saved = json.loads((run / "reference.json").read_text())
            outcome = json.loads((run / "outcome.json").read_text())
            names = {path.name for path in run.iterdir()}
        self.assertEqual(result, 0)
        self.assertEqual(prompts, [])  # No reference capture and no per-trial prompts.
        self.assertNotIn("reference_raw.png", names)
        self.assertTrue({"start_raw.png", "readings.csv", "metrics.json", "frames"} <= names)
        self.assertEqual((outcome["reason"], outcome["leg"]), ("estimated_distance", 1))
        self.assertGreater(outcome["start"]["lateral_px"], 30)  # Placed left of center.
        self.assertAlmostEqual(saved["zero_heading_deg"], (200.0 - saved["yaw_deg"]) % 360, delta=0.01)


if __name__ == "__main__":
    unittest.main()
