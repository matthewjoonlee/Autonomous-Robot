"""quiz3_app intersection run start: search along the lane until the intersection or a sign is seen."""
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import quiz3_app  # noqa: E402
from quiz3_app import App, sign_front_in  # noqa: E402


class Fake:
    def __init__(self):
        self.out = []

    def latest(self):
        return self.out.pop(0) if self.out else None

    def reset(self):
        pass

    def submit(self, *a):
        pass


def make_app(**overrides):
    defaults = dict(no_lane=True, lane_offset_in=0.0, camera_zones=False, drive_speed=4.0,
                    measure_speed=False, start_front_in=None, exit_before_in=1.0,
                    continue_after_in=3.0, zone_lead=0.8, pause_s=1.5, finish_s=0.0, lost_s=2.0,
                    unfamiliar_s=10.0, max_run_s=120.0, max_heading=30.0, search_s=60.0,
                    sign_trigger_in=45.0, no_sign_trigger=False, speed=22, max_turn=15, zone_offset=7.09)
    defaults.update(overrides)
    args = types.SimpleNamespace(**defaults)
    said = []
    announcer = types.SimpleNamespace(say=said.append)
    app = App(args, None, None, None, None, None, None, announcer, Fake(), None, Fake())
    app.mode, app.state = "intersection", "ready"
    app.said = said
    return app


def sign(frame_id, label, height):
    box = (0, 0, height, height)
    return {"frame_id": frame_id, "label": label, "box": box, "raw": {"label": label, "box": box}}


def location(frame_id, front, offset=7.09):
    return {"frame_id": frame_id, "familiar": True, "raw_distance_in": front + offset,
            "zone": None, "distance_in": front + offset, "annotations": []}


class SearchTest(unittest.TestCase):
    def test_sign_size_to_distance(self):
        self.assertAlmostEqual(sign_front_in((0, 0, 60, 60)), 28.1, delta=0.5)
        self.assertAlmostEqual(sign_front_in((0, 0, 100, 100)), 12.5, delta=0.5)
        self.assertIsNone(sign_front_in(None))

    def test_close_misreads_keep_it_searching_without_stopping(self):
        app = make_app()
        app.start_run(0.0)
        app.frame_id = 0
        for k in range(150):   # 15 s of the far-start false ~4 in reading
            now = 0.1 * k
            app.drive_time = now
            app.worker.out.append(location(app.run_frame0 + k, 4.0))
            app.update_run(now)
        self.assertEqual(app.state, "driving")
        self.assertTrue(app.searching())
        self.assertEqual(app.said, [])

    def test_sign_starts_the_zones(self):
        app = make_app()
        app.start_run(0.0)
        for k, height in enumerate((50, 110, 50)):   # one oversized box is outvoted
            app.sign_result = sign(app.run_frame0 + k, "STRAIGHT", height)
            app.update_run(0.1 * k)
            self.assertEqual(app.searching(), k < 2)
        self.assertAlmostEqual(app.odo_front, sign_front_in((0, 0, 50, 50)), delta=0.5)
        self.assertEqual(app.said, [])        # 35.9 in: still on the approach

    def test_far_sign_is_ignored(self):
        app = make_app()
        app.start_run(0.0)
        for k in range(5):
            app.sign_result = sign(app.run_frame0 + k, "LEFT", 30)
            app.update_run(0.1 * k)
        self.assertTrue(app.searching())

    def test_intersection_readings_start_the_zones(self):
        app = make_app()
        app.start_run(0.0)
        for k in range(5):
            app.worker.out.append(location(app.run_frame0 + k, 30.0))
            app.update_run(0.1 * k)
        self.assertFalse(app.searching())
        self.assertAlmostEqual(app.odo_front, 30.0, delta=0.5)

    def test_sign_in_view_overrides_false_far_readings(self):
        app = make_app()
        app.start_run(0.0)
        for k in range(30):   # sign says the robot closes from ~36 to ~19 in; camera says 36
            app.drive_time = 0.1 * k
            height = int(50 + k)
            app.sign_result = sign(app.run_frame0 + k, "STRAIGHT", height)
            app.worker.out.append(location(app.run_frame0 + k, 36.0))
            app.update_run(0.1 * k)
        self.assertAlmostEqual(app.odo_front, sign_front_in((0, 0, 79, 79)), delta=2.0)

    def test_stale_steady_box_is_not_used(self):
        app = make_app(start_front_in=20.0)
        app.start_run(0.0)
        for k in range(5):    # sign gone from this frame (raw None), steady box still held
            app.sign_result = {"frame_id": app.run_frame0 + k, "label": "LEFT",
                               "box": (0, 0, 50, 50), "raw": None}
            app.update_run(0.0)
        self.assertAlmostEqual(app.odo_front, 20.0, delta=0.01)

    def test_search_gives_up(self):
        app = make_app(search_s=5.0)
        app.start_run(0.0)
        app.worker.out.append(location(app.run_frame0, 4.0))
        app.update_run(1.0)
        app.worker.out.append(location(app.run_frame0 + 1, 4.0))
        app.update_run(6.0)
        self.assertEqual(app.state, "ready")


if __name__ == "__main__":
    unittest.main()
