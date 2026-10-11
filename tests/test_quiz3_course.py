"""Course orchestration and operator finish-stop checks."""

import unittest

from quiz3_autonomous import Settings, control_loop, course_timing
from quiz3_course import run_suite
import test_quiz3_autonomous as centering_helpers
from track_vision import DEFAULT_SESSION, read_image


class CourseTests(unittest.TestCase):
    def test_distance_is_only_a_time_estimate_with_fallback_cap(self):
        nominal, cap = course_timing(5, 11)
        self.assertAlmostEqual(nominal, 152.4 / 11)
        self.assertAlmostEqual(cap, nominal * 2)
        self.assertEqual(course_timing(5, 1)[1], 30)

    def test_suite_runs_four_labeled_trials(self):
        calls = []

        def runner(arguments):
            calls.append(arguments)
            return 0

        self.assertEqual(run_suite(True, runner), 0)
        self.assertEqual([args[1] for args in calls], ["straight", "left", "right", "nudge"])
        self.assertTrue(all("--execute" in args and args[-2] == "5" for args in calls))

    def test_suite_stops_after_fault_or_cancellation(self):
        calls = []

        def runner(arguments):
            calls.append(arguments)
            return 2 if len(calls) == 2 else 0

        self.assertEqual(run_suite(False, runner), 2)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all("--execute" not in args for args in calls))

    def test_finish_marker_stops_before_another_command(self):
        good = read_image(DEFAULT_SESSION / "001_centered_straight.jpg")
        camera, imu, motors, reference, clock, rows, frames = centering_helpers.CenteringTests().setup_loop(
            [good] * 4, [0] * 4)
        checks = iter([False, False, True])
        reason = control_loop(camera, imu, motors, reference, 0, 20, Settings(), rows, frames,
                              clock, clock.sleep, finish_requested=lambda: next(checks))
        self.assertEqual(reason, "finish_marker")
        self.assertEqual(len(motors.commands), 1)
        self.assertTrue(motors.stops)

    def test_finish_arriving_during_capture_prevents_drive(self):
        good = read_image(DEFAULT_SESSION / "001_centered_straight.jpg")
        camera, imu, motors, reference, clock, rows, frames = centering_helpers.CenteringTests().setup_loop([good], [0])
        checks = iter([False, True])
        reason = control_loop(camera, imu, motors, reference, 0, 20, Settings(), rows, frames,
                              clock, clock.sleep, finish_requested=lambda: next(checks))
        self.assertEqual(reason, "finish_marker")
        self.assertFalse(motors.commands)
        self.assertTrue(motors.stops)
        self.assertEqual(rows[-1]["state"], "finish_marker")


if __name__ == "__main__":
    unittest.main()
