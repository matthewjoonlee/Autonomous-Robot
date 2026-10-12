"""One-program session: wait to be placed, drive a leg, stop, repeat."""

import contextlib
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from lane_center import load
from quiz3_autonomous import Settings, control_loop, leg_timing, run_session, wait_for_placement
import test_quiz3_autonomous as centering_helpers
from track_vision import DEFAULT_SESSION, read_image

GOOD = read_image(DEFAULT_SESSION / "001_centered_straight.jpg")
BLACK = np.zeros_like(GOOD)
# Robot slid left of center (real capture; a whole-image shift would read as rotation).
EDGE = read_image(DEFAULT_SESSION / "002_left_offset_straight.jpg")
CENTER_NEAR = 0.6574 * 640  # Lane near-row position in GOOD.


class Camera:
    """Plays frames in order, then keeps repeating the last one."""

    def __init__(self, frames, clock):
        self.frames, self.clock = list(frames), clock
        self.index = 0

    def capture_array(self):
        frame = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        return cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)


def wait(frames, stopped_near=None, commands=None):
    clock = centering_helpers.Clock()
    with contextlib.redirect_stdout(io.StringIO()):
        result = wait_for_placement(Camera(frames, clock), Settings(), stopped_near,
                                    commands=commands or (lambda: None), clock=clock, sleep=clock.sleep)
    return result, clock.now


class CourseTests(unittest.TestCase):
    def test_leg_time_fallback_is_bounded(self):
        nominal, cap = leg_timing(15, 14)
        self.assertAlmostEqual(nominal, 15 * 30.48 / 14)
        self.assertAlmostEqual(cap, nominal * 2)
        self.assertEqual(leg_timing(15, 1)[1], 90)

    def test_first_placement_goes_once_lane_is_steady(self):
        result, elapsed = wait([GOOD] * 80)
        self.assertEqual(result, "go")
        self.assertGreaterEqual(elapsed, 1.5)

    def test_stopped_robot_does_not_restart_until_moved(self):
        calls = itertools.count()
        result, elapsed = wait([GOOD] * 200, stopped_near=CENTER_NEAR,
                               commands=lambda: "q" if next(calls) >= 150 else None)
        self.assertEqual(result, "quit")
        self.assertGreater(elapsed, 5)

    def test_picked_up_and_set_down_restarts(self):
        result, _ = wait([GOOD] * 10 + [BLACK] * 15 + [GOOD] * 60, stopped_near=CENTER_NEAR)
        self.assertEqual(result, "go")

    def test_slid_to_the_edge_restarts(self):
        result, _ = wait([GOOD] * 10 + [EDGE] * 60, stopped_near=CENTER_NEAR)
        self.assertEqual(result, "go")

    def test_enter_starts_now_and_q_quits(self):
        self.assertEqual(wait([GOOD] * 5, stopped_near=CENTER_NEAR, commands=lambda: "")[0], "go")
        self.assertEqual(wait([BLACK] * 5, commands=lambda: "q")[0], "quit")

    def test_one_session_runs_legs_from_different_starts(self):
        clock = centering_helpers.Clock()
        frames = ([GOOD] * 40 + [GOOD] * 10 + [GOOD] * 25      # placed centered, start, leg 1
                  + [BLACK] * 15 + [EDGE] * 45 + [EDGE] * 10 + [EDGE] * 25)  # picked up, edge, leg 2
        motors = centering_helpers.Motors(clock)
        imu = centering_helpers.IMU(itertools.repeat(0.0))
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            legs = run_session(Camera(frames, clock), imu, motors, Settings(), load(), Path(folder),
                               leg_ft=0.25, max_legs=2, commands=lambda: None, clock=clock, sleep=clock.sleep)
            outcomes = [json.loads((run / "outcome.json").read_text()) for run in sorted(Path(folder).glob("run_*"))]
        self.assertEqual(legs, 2)
        self.assertEqual([o["leg"] for o in outcomes], [1, 2])
        self.assertTrue(all(o["reason"] == "estimated_distance" for o in outcomes))
        self.assertLess(abs(outcomes[0]["start"]["lateral_px"]), 40)
        self.assertGreater(outcomes[1]["start"]["lateral_px"], 30)  # Knew it started left of center.
        self.assertTrue(motors.commands and motors.stops)

    def test_lane_fault_ends_leg_but_not_session(self):
        clock = centering_helpers.Clock()
        # Leg 1 loses the lane for good (searching ends it); then the robot is placed again.
        frames = [GOOD] * 40 + [GOOD] * 10 + [GOOD] * 3 + [BLACK] * 120 + [GOOD] * 45 + [GOOD] * 10 + [GOOD] * 25
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            # Searching keeps driving, so the leg must be long enough for the 3 s limit to end it.
            legs = run_session(Camera(frames, clock), centering_helpers.IMU(itertools.repeat(0.0)), None,
                               Settings(), load(), Path(folder), leg_ft=5, max_legs=2,
                               commands=lambda: None, clock=clock, sleep=clock.sleep)
            reasons = [json.loads((run / "outcome.json").read_text())["reason"]
                       for run in sorted(Path(folder).glob("run_*"))]
        self.assertEqual(legs, 2)
        self.assertEqual(reasons, ["Lane not found; run stopped", "estimated_distance"])

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
