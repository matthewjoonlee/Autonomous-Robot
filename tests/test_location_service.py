import csv
import tempfile
import unittest
from pathlib import Path

from build_location_reference import frame_distances
from location_service import PHASES, PositionFuser, ZoneTracker, zone_index


class ZoneIndexTest(unittest.TestCase):
    def test_boundaries(self):
        cases = {30: "approach", 25.3: "approach", 25.1: "read_sign", 10.2: "read_sign",
                 10.0: "pause", 6.1: "pause", 5.9: "entering", 1.6: "entering",
                 1.4: "turn", -1.4: "turn", -1.6: "exiting", -6.0: "exiting", -6.2: "continue"}
        for distance, phase in cases.items():
            self.assertEqual(PHASES[zone_index(distance)], phase, distance)


class ZoneTrackerTest(unittest.TestCase):
    def feed(self, tracker, values, spread=0.0):
        return [change for change in (tracker.update(v, spread) for v in values) if change]

    def test_starts_in_measured_zone(self):
        self.assertEqual(self.feed(ZoneTracker(), [18, 18, 18]), ["read_sign"])

    def test_steps_through_every_phase_once_in_order(self):
        tracker = ZoneTracker()
        values = [36, 36] + [36 - 0.5 * i for i in range(110)]
        changes = self.feed(tracker, values)
        self.assertEqual(changes, list(PHASES))

    def test_never_skips_or_goes_back(self):
        tracker = ZoneTracker()
        changes = self.feed(tracker, [30, 30, 30, 0, 0, 0, 0, 0, 0, 0, 0, 30, 30, 30])
        self.assertEqual(changes, ["approach", "read_sign", "pause", "entering", "turn"])

    def test_ignores_uncertain_estimates(self):
        tracker = ZoneTracker()
        self.feed(tracker, [30, 30])
        self.assertEqual(self.feed(tracker, [0, 0, 0], spread=20), [])
        self.assertEqual(tracker.phase, "approach")


class FrameDistancesTest(unittest.TestCase):
    def write(self, folder, speeds, marks):
        with (folder / "frames.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "time_s", "side_ok", "heading_rel_deg", "speed", "turn"])
            for i, s in enumerate(speeds):
                w.writerow([i, i / 10, 1, 0, s, 0])
        with (folder / "marks.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "frame", "distance_in", "text"])
            for frame, d in marks:
                w.writerow([frame / 10, frame, d, d])

    def test_interpolates_by_driving_frames_and_keeps_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            # parked 0-1, drive 2-5 (4 frames), stopped 6-7 (judging pause), drive 8-11
            speeds = [0, 0, 22, 22, 22, 22, 0, 0, 22, 22, 22, 22, 0]
            self.write(folder, speeds, [(1, 36), (12, 28)])
            d = frame_distances(folder)
            self.assertEqual(d[0], 36)            # parked before the first mark
            self.assertEqual(d[6], d[7])          # no movement while stopped
            self.assertAlmostEqual(d[6], 32.0)    # halfway: 4 of 8 driving frames
            self.assertEqual(d[12], 28)


if __name__ == "__main__":
    unittest.main()


class ZoneOffsetTest(unittest.TestCase):
    def test_front_reference_reaches_each_zone_earlier_by_offset(self):
        center = ZoneTracker()
        front = ZoneTracker(offset=5.0)
        values = [36, 36] + [36 - 0.25 * i for i in range(220)]
        seen_center, seen_front = {}, {}
        for i, v in enumerate(values):
            for tracker, seen in ((center, seen_center), (front, seen_front)):
                change = tracker.update(v, 0.0)
                if change:
                    seen.setdefault(change, values[i])
        # The front reaches the turn zone when the center is still ~5 in before it.
        self.assertAlmostEqual(seen_front["turn"] - seen_center["turn"], 5.0, delta=0.6)


class BackwardJumpTest(unittest.TestCase):
    def test_single_far_misreads_are_ignored(self):
        tracker = ZoneTracker()
        values = [14, 14, 13, 12, 40, 11.5, 40, 40, 11, 10.5, 10, 9.5, 40, 9, 8.5, 8]
        changes = [c for c in (tracker.update(v, 0.0) for v in values) if c]
        self.assertEqual(changes, ["read_sign", "pause"])
        self.assertLess(tracker.smoothed, 10)

    def test_robot_really_moved_back_is_accepted_after_agreement(self):
        tracker = ZoneTracker()
        for v in [14, 14, 13]:
            tracker.update(v, 0.0)
        for v in [30, 30, 30]:
            tracker.update(v, 0.0)
        self.assertGreater(tracker.smoothed, 25)


class PositionFuserTest(unittest.TestCase):
    def drive(self, fuser, readings, speed=4.0, dt=0.1):
        t = 0.0
        for reading in readings:
            t += dt
            fuser.observe(reading if not callable(reading) else reading(t), t)
        return fuser.position(t), t

    def test_tracks_driving_and_ignores_misreads(self):
        fuser = PositionFuser()
        truth = lambda t: 40 - 4.0 * t
        readings = [truth] * 30 + [lambda t: 45.0] * 3 + [truth] * 20   # 3 misreads
        x, t = self.drive(fuser, readings)
        self.assertAlmostEqual(x, 40 - 4.0 * t, delta=1.0)

    def test_bad_start_is_corrected_by_agreeing_readings(self):
        fuser = PositionFuser()
        x, t = self.drive(fuser, [10.0] * 5)          # starts believing +10
        x, t2 = self.drive(fuser, [35.0] * 8)        # cameras consistently say +35
        self.assertGreater(fuser.position(t2), 25)

    def test_close_in_uses_driving_distance_only(self):
        fuser = PositionFuser(camera_until=20)
        for _ in range(5):
            fuser.observe(30.0, 0.0)                 # starts at +30
        before = fuser.position(4.0)                 # 30 - 4 x 4 = 14, inside 20 in
        for _ in range(8):
            fuser.observe(40.0, 4.0)                 # misreads near the intersection: ignored
        self.assertAlmostEqual(fuser.position(4.0), before)
        self.assertAlmostEqual(before, 14.0, delta=0.01)

    def test_close_first_readings_do_not_start_it(self):
        fuser = PositionFuser()
        for _ in range(5):
            fuser.observe(10.0, 0.5)                 # stuck close value at the start
        self.assertIsNone(fuser.position(0.5))
        for _ in range(5):
            fuser.observe(10.0, 3.5)                 # still only close: keep waiting
        self.assertIsNone(fuser.position(3.5))
        for _ in range(5):
            fuser.observe(40.0, 4.0)                 # far readings start it
        self.assertAlmostEqual(fuser.position(4.0), 40.0, delta=0.01)

    def test_sign_blocks_camera_readings(self):
        fuser = PositionFuser(speed=4.0, start=30.0)
        fuser.observe_sign(30.0, 0.0)
        for k in range(10):   # false far readings while the sign is in view
            fuser.observe(45.0, 0.1 * k)
        self.assertAlmostEqual(fuser.position(0.9), 30.0 - 3.6, delta=0.1)

    def test_sign_starts_and_steers(self):
        fuser = PositionFuser(speed=4.0)
        fuser.observe_sign(36.0, 1.0)
        self.assertAlmostEqual(fuser.position(1.0), 36.0)
        fuser.observe_sign(30.0, 1.0)
        self.assertAlmostEqual(fuser.position(1.0), 33.0)

    def test_known_start(self):
        fuser = PositionFuser(speed=4.0, start=38.0)
        self.assertAlmostEqual(fuser.position(1.0), 34.0)


class HandPushedTest(FrameDistancesTest):
    def test_hand_pushed_pass_interpolates_by_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.write(folder, [0] * 11, [(0, 30), (10, 20)])   # no motor commands at all
            d = frame_distances(folder)
            self.assertAlmostEqual(d[5], 25.0)
            self.assertEqual(d[0], 30)
            self.assertEqual(d[10], 20)
