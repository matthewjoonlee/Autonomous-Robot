"""Feedback-direction and bounded-loop checks before physical centering tests."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import cv2
import numpy as np

from quiz3_autonomous import (LaneTracker, Settings, choose_command, configure_motion_exposure,
                              control_loop)
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


def command(offset=0.0, heading=0.0, lost_for=0.0, previous_turn=0.0, dt=1.0, settings=Settings()):
    return choose_command(offset, heading, lost_for, previous_turn, dt, settings)


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
        self.assertEqual(command(0.002)["raw_turn"], 0)

    def test_commands_are_limited(self):
        result = command(0.8, -4)
        self.assertEqual(result["target_heading_deg"], 4)
        self.assertEqual(result["raw_turn"], -10)
        self.assertEqual(result["speed"], 25)

    def test_small_errors_steer_proportionally_without_deadband(self):
        self.assertGreater(command(0.004)["turn"], 0)
        self.assertLess(command(-0.004)["turn"], 0)

    def test_turn_changes_at_bounded_rate(self):
        self.assertAlmostEqual(command(0.8, dt=0.04)["turn"], 1.6)
        self.assertAlmostEqual(command(0.8, previous_turn=9.0, dt=0.04)["turn"], 10)
        self.assertAlmostEqual(command(-0.8, previous_turn=5.0, dt=0.04)["turn"], 3.4)

    def test_speed_ramps_down_while_lane_is_lost(self):
        speeds = [command(lost_for=t)["speed"] for t in (0, 0.3, 0.65, 0.9, 1.0, 2.0)]
        self.assertEqual(speeds[:2], [25, 25])
        self.assertTrue(0 < speeds[3] < speeds[2] < 25)
        self.assertEqual(speeds[4:], [0, 0])
        self.assertEqual(command(0.8, lost_for=1.5)["turn"], 0)
        with self.assertRaisesRegex(RuntimeError, "Lane not found"):
            command(lost_for=3.0)

    def test_optional_slowdown_during_large_turns(self):
        settings = replace(Settings(), turn_slowdown=0.2)
        self.assertEqual(command(0.8, settings=settings)["speed"], 20)
        self.assertEqual(command(settings=settings)["speed"], 25)

    def test_faults_do_not_produce_commands(self):
        for offset, heading, lost_for in [
            (None, 0, 0), (float("nan"), 0, 0), (0, None, 0), (0, 10, 0), (0, -10, 0), (0, 0, 5),
        ]:
            with self.subTest(offset=offset, heading=heading, lost_for=lost_for):
                with self.assertRaises(RuntimeError):
                    command(offset, heading, lost_for)

    def test_tracker_holds_and_relaxes_estimate_on_unusable_frames(self):
        reference = {"fit": (0.1, 0.6)}
        tracker = LaneTracker(reference, 0.0, Settings())
        tracked = {"status": "tracked", "confidence": 0.9, "fit": (0.1, 0.65)}
        lost = {"status": "line_lost", "confidence": 0.0, "fit": None}
        self.assertTrue(tracker.update(tracked, 0.04, 0.04))
        self.assertTrue(tracker.update(tracked, 0.04, 0.08))
        self.assertAlmostEqual(tracker.offset, 0.03)
        self.assertEqual(tracker.prior(0.08), (0.1, 0.65))
        self.assertFalse(tracker.update(lost, None, 0.12))
        self.assertTrue(0 < tracker.offset < 0.03)
        self.assertEqual(tracker.streak, 0)
        self.assertIsNone(tracker.prior(0.5))
        weak = dict(tracked, confidence=0.3)
        self.assertFalse(tracker.update(weak, 0.5, 0.16))

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
                         ["driving", "coasting", "coasting", "driving", "driving"])
        self.assertEqual([speed for _, speed, _ in motors.commands], [25] * 5)
        self.assertEqual(len(motors.stops), 1)  # Only the final stop.
        self.assertEqual(frames.ids, [row["frame_id"] for row in rows])  # Every frame recorded.

    def test_persistent_loss_slows_waits_then_faults(self):
        good = read_image(DEFAULT_SESSION / "001_centered_straight.jpg")
        bad = np.zeros_like(good)
        camera, imu, motors, reference, clock, rows, frames = self.setup_loop(
            [good] + [bad] * 90, [0] * 91)
        with self.assertRaisesRegex(RuntimeError, "Lane not found"):
            control_loop(camera, imu, motors, reference, 0, 5, Settings(), rows, frames, clock, clock.sleep)
        speeds = [speed for _, speed, _ in motors.commands]
        self.assertEqual(speeds[:8], [25] * 8)
        self.assertEqual(speeds, sorted(speeds, reverse=True))
        self.assertTrue(all(t < 1.0 for t, _, _ in motors.commands))
        self.assertIn("waiting", [row["state"] for row in rows])
        self.assertTrue(motors.stops)
        self.assertEqual(rows[-1]["speed"], 0)

    def test_resume_after_full_stop_needs_consecutive_lane_frames(self):
        good = read_image(DEFAULT_SESSION / "001_centered_straight.jpg")
        bad = np.zeros_like(good)
        sequence = [good] + [bad] * 30 + [good, bad, good, good, good]
        camera, imu, motors, reference, clock, rows, frames = self.setup_loop(sequence, [0] * len(sequence))
        control_loop(camera, imu, motors, reference, 0, 0.04 * 35 - 0.02, Settings(),
                     rows, frames, clock, clock.sleep)
        states = [row["state"] for row in rows[30:35]]
        self.assertEqual(states, ["waiting", "waiting", "waiting", "waiting", "driving"])
        self.assertEqual(rows[34]["speed"], 25)
        self.assertGreater(motors.commands[-1][0], 1.3)

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


if __name__ == "__main__":
    unittest.main()
