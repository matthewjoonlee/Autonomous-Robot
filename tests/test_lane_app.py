"""Touchscreen app: START drives, STOP stops, QUIT exits, camera view shown, no-motion warning."""

import contextlib
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lane_app import DRIVING_BUTTONS, IDLE_BUTTONS, LaneApp, MotionWatch, render
from lane_center import load, reference
from quiz3_autonomous import Settings
import test_quiz3_autonomous as helpers
from test_quiz3_course import Camera, GOOD
from track_vision import detect_lane


class FakeDisplay:
    """Records what the app shows; delivers button presses at scripted clock times."""

    def __init__(self, clock, presses):
        self.clock, self.presses = clock, sorted(presses)
        self.frames, self.statuses, self.button_sets = [], [], []

    def poll_events(self):
        due = [name for when, name in self.presses if when <= self.clock.now]
        self.presses = [(when, name) for when, name in self.presses if when > self.clock.now]
        return due

    def set_frame(self, frame):
        self.frames.append(frame)

    def set_status(self, text, **_):
        self.statuses.append(text)

    def set_buttons(self, buttons, **_):
        self.button_sets.append([b["handle"] for b in buttons])


def app_run(presses, frames=None, imu_values=None):
    clock = helpers.Clock()
    display = FakeDisplay(clock, presses)
    motors = helpers.Motors(clock)
    imu = helpers.IMU(imu_values or itertools.repeat(0.0))
    with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
        app = LaneApp(display, Camera(frames or [GOOD], clock), imu, motors, Settings(), load(), Path(folder),
                      leg_ft=15, clock=clock, sleep=clock.sleep)
        runs = app.run()
        outcomes = [json.loads((run / "outcome.json").read_text()) for run in sorted(Path(folder).glob("run_*"))]
    return runs, display, motors, outcomes, app


class LaneAppTests(unittest.TestCase):
    def test_start_drives_until_stop_then_quit(self):
        runs, display, motors, outcomes, app = app_run([(0.1, "start"), (2.0, "stop"), (2.5, "quit")])
        self.assertEqual(runs, 1)
        self.assertEqual([o["reason"] for o in outcomes], ["finish_marker"])
        self.assertTrue(motors.commands)
        self.assertTrue(all(t < 2.1 for t, _, _ in motors.commands))  # Nothing driven after STOP.
        self.assertTrue(motors.stops)
        idle, driving = [b["handle"] for b in IDLE_BUTTONS], [b["handle"] for b in DRIVING_BUTTONS]
        self.assertEqual(display.button_sets[:3], [idle, driving, idle])
        self.assertGreater(len(display.frames), 10)  # Camera view shown while idle and driving.
        self.assertIn("stopped", app.message)

    def test_quit_while_idle_never_drives(self):
        runs, display, motors, outcomes, _ = app_run([(0.5, "quit")])
        self.assertEqual((runs, outcomes, motors.commands), (0, [], []))
        self.assertTrue(display.frames)

    def test_no_lane_at_start_reports_and_stays_stopped(self):
        black = np.zeros_like(GOOD)
        runs, _, motors, outcomes, app = app_run([(0.1, "start"), (1.5, "quit")], frames=[black])
        self.assertEqual(outcomes[0]["reason"], "Lane not visible at start; place the robot where the camera sees the lane")
        self.assertFalse(motors.commands)

    def test_motion_watch_flags_robot_that_does_not_move(self):
        watch = MotionWatch()
        frame = GOOD.copy()
        flags = [watch.update(frame, 0.0, True, t * 0.1) for t in range(25)]
        self.assertFalse(any(flags[:19]))
        self.assertTrue(flags[-1])
        watch = MotionWatch()
        moving = [watch.update(np.roll(GOOD, 4 * t, axis=1), 0.2 * t, True, t * 0.1) for t in range(25)]
        self.assertFalse(any(moving))
        self.assertFalse(MotionWatch().update(frame, 0.0, False, 0.0))

    def test_render_draws_lane_and_reference(self):
        detection = detect_lane(GOOD)
        view = render(GOOD, detection, reference(load()), ["hello"], "NOT MOVING")
        self.assertEqual(view.shape, GOOD.shape)
        self.assertFalse(np.array_equal(view, GOOD))


if __name__ == "__main__":
    unittest.main()


class MainTests(unittest.TestCase):
    def test_main_opens_display_and_quits_cleanly(self):
        import sys
        import time
        import types
        from unittest.mock import patch
        import cv2
        import lane_app

        closed = []

        class FakeRobotDisplay:
            def __init__(self, **kwargs):
                self.start = time.monotonic()
                self.frames = 0

            def set_status(self, *args, **kwargs):
                pass

            def set_buttons(self, *args, **kwargs):
                pass

            def set_frame(self, frame):
                self.frames += 1

            def poll_events(self):
                return ["quit"] if self.frames >= 3 else []

            def close(self):
                closed.append("display")

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
                return cv2.cvtColor(GOOD, cv2.COLOR_BGR2BGRA)

            def stop(self):
                pass

            def close(self):
                closed.append("camera")

        class FakeIMU:
            SIGN = 1

            def heading(self):
                return 0.0

            def close(self):
                closed.append("imu")

        modules = {"display": types.SimpleNamespace(RobotDisplay=FakeRobotDisplay),
                   "picamera2": types.SimpleNamespace(Picamera2=FakeCamera),
                   "imu_controller": types.SimpleNamespace(IMUDevice=FakeIMU)}
        with patch.dict(sys.modules, modules), patch("lane_app.time.sleep"), patch("builtins.print"):
            self.assertEqual(lane_app.main(["--exposure-us", "0"]), 0)
        self.assertEqual(sorted(closed), ["camera", "display", "imu"])
