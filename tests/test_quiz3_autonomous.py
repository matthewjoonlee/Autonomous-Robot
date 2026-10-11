"""Feedback-direction and bounded-loop checks before physical centering tests."""

import unittest

import cv2
import numpy as np

from quiz3_autonomous import Settings, choose_command, control_loop
from track_vision import DEFAULT_SESSION, detect_line, read_image


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


class Motors:
    def __init__(self, clock):
        self.clock = clock
        self.commands, self.stops = [], []

    def drive(self, speed, turn):
        self.commands.append((self.clock(), speed, turn))

    def stop(self):
        self.stops.append(self.clock())


class CenteringTests(unittest.TestCase):
    def test_physical_right_maps_to_negative_motor_command(self):
        command = choose_command(0.1, 0, 0.95, "tracked", 0.05)
        self.assertGreater(command["physical_turn"], 0)
        self.assertLess(command["raw_turn"], 0)

    def test_physical_left_maps_to_positive_motor_command(self):
        command = choose_command(-0.1, 0, 0.95, "tracked", 0.05)
        self.assertLess(command["physical_turn"], 0)
        self.assertGreater(command["raw_turn"], 0)

    def test_heading_feedback_straightens_at_reference(self):
        self.assertGreater(choose_command(0, 3, 0.95, "tracked", 0.05)["raw_turn"], 0)
        self.assertLess(choose_command(0, -3, 0.95, "tracked", 0.05)["raw_turn"], 0)
        self.assertEqual(choose_command(0.005, 0, 0.95, "tracked", 0.05)["raw_turn"], 0)

    def test_commands_are_limited(self):
        command = choose_command(0.8, -4, 0.95, "tracked", 0.05)
        self.assertEqual(command["target_heading_deg"], 4)
        self.assertEqual(command["raw_turn"], -10)
        self.assertEqual(command["speed"], 25)

    def test_faults_do_not_produce_commands(self):
        for error, heading, confidence, status, age in [
            (None, 0, 0, "line_lost", 0.05), (0, 0, 0, "ambiguous", 0.05),
            (0, 0, 0.5, "tracked", 0.05), (0, 0, 0.9, "tracked", 0.2),
            (0, 10, 0.9, "tracked", 0.05), (0, None, 0.9, "tracked", 0.05),
            (float("nan"), 0, 0.9, "tracked", 0.05),
        ]:
            with self.subTest(status=status, age=age, heading=heading):
                with self.assertRaises(RuntimeError):
                    choose_command(error, heading, confidence, status, age)

    def setup_loop(self, frames, headings, delay=0):
        clock = Clock()
        reference_frame = read_image(DEFAULT_SESSION / "001_centered_straight.jpg")
        reference = detect_line(reference_frame)
        reference["width"] = 640
        return (Camera(frames, clock, delay), IMU(headings), Motors(clock),
                reference, clock, [], [])

    def test_line_loss_latches_stop_after_first_command(self):
        good = read_image(DEFAULT_SESSION / "001_centered_straight.jpg")
        bad = np.zeros_like(good)
        camera, imu, motors, reference, clock, rows, frames = self.setup_loop([good, bad, good], [0, 0, 0])
        with self.assertRaisesRegex(RuntimeError, "Tracking fault"):
            control_loop(camera, imu, motors, reference, 0, 1, Settings(), rows, frames, clock, clock.sleep)
        self.assertEqual(len(motors.commands), 1)
        self.assertTrue(motors.stops)
        self.assertEqual(rows[-1]["speed"], 0)
        self.assertEqual(frames[-1][0], 1)

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
                    self.assertTrue(all(row["state"] == "diagnostic" for row in rows))


if __name__ == "__main__":
    unittest.main()
