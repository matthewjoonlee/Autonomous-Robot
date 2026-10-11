"""Bounded command timing and interruption checks with simulated hardware."""

import types
import unittest

from calibrate_motors import run_trial


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


class Motors:
    def __init__(self, clock):
        self.clock = clock
        self.commands = []
        self.stops = []

    def drive(self, speed, turn):
        self.commands.append((self.clock(), speed, turn))

    def stop(self):
        self.stops.append(self.clock())


class IMU:
    SIGN = 1

    def __init__(self, values):
        self.values = iter(values)

    def heading(self):
        value = next(self.values, 0.0)
        if isinstance(value, BaseException):
            raise value
        return value


class Writer:
    def __init__(self):
        self.rows = []

    def writerow(self, row):
        self.rows.append(row)


class MotorCalibrationTests(unittest.TestCase):
    def setup_trial(self, trial="forward", readings=()):
        clock = Clock()
        motors, imu, writer = Motors(clock), IMU(readings), Writer()
        args = types.SimpleNamespace(trial=trial, turn=6, speed=20,
                                     duration=0.12, max_heading=15)
        return clock, motors, imu, writer, args

    def test_direction_and_deadline(self):
        for trial, expected in [("forward", (20, 0)), ("left", (20, -6)),
                                ("right", (20, 6)), ("stop", (0, 0))]:
            with self.subTest(trial=trial):
                clock, motors, imu, writer, args = self.setup_trial(trial)
                run_trial(motors, imu, args, writer, clock, clock.sleep)
                self.assertTrue(motors.commands)
                self.assertTrue(all(command[1:] == expected for command in motors.commands))
                self.assertTrue(all(command[0] < args.duration for command in motors.commands))
                self.assertTrue(any(row["phase"] == "stopped" for row in writer.rows))
                self.assertGreaterEqual(motors.stops[-1], args.duration)

    def test_missing_sensor_stops_after_motion(self):
        clock, motors, imu, writer, args = self.setup_trial(readings=[0, 0, None])
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            run_trial(motors, imu, args, writer, clock, clock.sleep)
        self.assertEqual(len(motors.commands), 1)
        self.assertEqual(writer.rows[-1]["phase"], "aborted")
        self.assertGreaterEqual(len(motors.stops), 3)

    def test_heading_limit_prevents_command(self):
        clock, motors, imu, writer, args = self.setup_trial(readings=[0, 18])
        with self.assertRaisesRegex(RuntimeError, "Heading limit"):
            run_trial(motors, imu, args, writer, clock, clock.sleep)
        self.assertFalse(motors.commands)
        self.assertEqual(writer.rows[-1]["phase"], "aborted")

    def test_interruption_stops_after_motion(self):
        clock, motors, imu, writer, args = self.setup_trial(readings=[0, 0, KeyboardInterrupt()])
        with self.assertRaises(KeyboardInterrupt):
            run_trial(motors, imu, args, writer, clock, clock.sleep)
        self.assertEqual(len(motors.commands), 1)
        self.assertEqual(writer.rows[-1]["phase"], "aborted")
        self.assertGreaterEqual(len(motors.stops), 3)

    def test_delayed_loop_cannot_resume_motion(self):
        clock, motors, imu, writer, args = self.setup_trial()
        args.duration = 0.5
        with self.assertRaisesRegex(RuntimeError, "Control loop delayed"):
            run_trial(motors, imu, args, writer, clock, lambda _: clock.sleep(0.15))
        self.assertEqual(len(motors.commands), 1)

    def test_slow_sensor_read_cannot_start_motion(self):
        clock, motors, imu, writer, args = self.setup_trial()

        def slow_heading():
            clock.sleep(0.11)
            return 0.0

        imu.heading = slow_heading
        with self.assertRaisesRegex(RuntimeError, "IMU read exceeded"):
            run_trial(motors, imu, args, writer, clock, clock.sleep)
        self.assertFalse(motors.commands)
        self.assertGreaterEqual(len(motors.stops), 3)

    def test_logging_failure_still_stops(self):
        clock, motors, imu, writer, args = self.setup_trial()

        def fail_write(row):
            if row["phase"] == "command":
                raise OSError("disk unavailable")

        writer.writerow = fail_write
        with self.assertRaisesRegex(OSError, "disk unavailable"):
            run_trial(motors, imu, args, writer, clock, clock.sleep)
        self.assertEqual(len(motors.commands), 1)
        self.assertGreaterEqual(len(motors.stops), 3)


if __name__ == "__main__":
    unittest.main()
