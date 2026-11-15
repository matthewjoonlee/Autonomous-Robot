import csv
import tempfile
import unittest
from pathlib import Path

from build_location_reference import frame_distances
from location_service import PHASES, ZoneTracker, zone_index


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
