"""Recording, scorecard, tuning overrides, and replay of recorded runs."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from quiz3_autonomous import Settings, control_loop, save_results, tuned
import replay_run
from run_metrics import main as metrics_main, score
from run_recording import FrameRecorder
from test_quiz3_autonomous import CenteringTests
from track_vision import DEFAULT_SESSION, read_image


def row(time, status="tracked", error=0.0, turn=0.0, speed=25, state="driving"):
    return {"elapsed_s": time, "status": status, "confidence": 0.9 if status == "tracked" else 0.0,
            "near_error_px": error if status == "tracked" else None, "turn": turn, "speed": speed,
            "state": state, "imu_delta_deg": 0.5}


class RecorderTests(unittest.TestCase):
    def test_saves_every_nth_frame_as_jpeg(self):
        frame = read_image(DEFAULT_SESSION / "001_centered_straight.jpg")
        recorder = FrameRecorder(every=2)
        for index in range(5):
            recorder.add(index, frame)
        with tempfile.TemporaryDirectory() as folder:
            info = recorder.save(Path(folder) / "frames")
            names = sorted(path.name for path in (Path(folder) / "frames").iterdir())
            decoded = cv2.imread(str(Path(folder) / "frames" / names[0]))
        self.assertEqual(names, ["frame_00000.jpg", "frame_00002.jpg", "frame_00004.jpg"])
        self.assertEqual(decoded.shape, frame.shape)
        self.assertEqual((info["recorded_frames"], info["dropped_frames"]), (3, 0))

    def test_byte_budget_drops_instead_of_growing(self):
        recorder = FrameRecorder(max_bytes=1)
        recorder.add(0, read_image(DEFAULT_SESSION / "001_centered_straight.jpg"))
        recorder.close()
        self.assertEqual((len(recorder.encoded), recorder.dropped_budget), (0, 1))


class ScoreTests(unittest.TestCase):
    def test_lost_episodes_centering_and_weaving(self):
        rows = [row(0.0, error=30, turn=2), row(0.1, error=20, turn=-2), row(0.2, "line_lost", state="coasting"),
                row(0.3, "line_lost", state="waiting", speed=0), row(0.4, error=5, turn=2),
                row(0.5, error=-4, turn=0.5)]
        metrics = score(rows)
        self.assertEqual(metrics["lost_episodes"], 1)
        self.assertAlmostEqual(metrics["longest_lost_s"], 0.2)
        self.assertAlmostEqual(metrics["waiting_s"], 0.1)
        self.assertEqual(metrics["peak_error_px"], 30)
        self.assertEqual(metrics["centered_after_s"], 0.4)
        self.assertEqual(metrics["turn_reversals"], 2)
        self.assertAlmostEqual(metrics["tracked_pct"], 66.7)

    def test_never_centered_and_empty(self):
        self.assertIsNone(score([row(0.0, error=30), row(0.1, error=25)])["centered_after_s"])
        self.assertEqual(score([]), {"frames": 0})


class TuningTests(unittest.TestCase):
    def test_overrides_apply_and_are_typed(self):
        settings = tuned(Settings(), ["position_gain=30", "resume_frames=3"])
        self.assertEqual((settings.position_gain, settings.resume_frames), (30.0, 3))

    def test_safety_limits_and_bad_values_rejected(self):
        for overrides in (["speed=40"], ["abort_heading=45"], ["lost_limit=60"], ["position_gain"],
                          ["turn_rate=fast"], ["coast_full=2"], ["smoothing=0"], ["position_gain=-1"],
                          ["max_target_heading=12"]):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    tuned(Settings(), overrides)


class ReplayTests(unittest.TestCase):
    def record_run(self, folder):
        """Run the real motor-free loop with a real recorder and save it like the robot does."""
        good = read_image(DEFAULT_SESSION / "001_centered_straight.jpg")
        left = read_image(DEFAULT_SESSION / "002_left_offset_straight.jpg")
        bad = np.zeros_like(good)
        sequence = [good, left, left, bad, bad, left, left, left, good, good]
        camera, imu, _, reference, clock, rows, _ = CenteringTests().setup_loop(sequence, [0] * len(sequence))
        recorder = FrameRecorder()
        control_loop(camera, imu, None, reference, 0, 0.04 * len(sequence) - 0.02, Settings(),
                     rows, recorder, clock, clock.sleep)
        run = Path(folder) / "run_test"
        run.mkdir()
        cv2.imwrite(str(run / "reference_raw.png"), good)
        outcome = {"settings": Settings().__dict__, "trial": "left", "reason": "duration_complete"}
        save_results(run, rows, recorder, outcome)
        return run, rows

    def test_replay_reproduces_recorded_decisions(self):
        with tempfile.TemporaryDirectory() as folder:
            run, rows = self.record_run(folder)
            saved = json.loads((run / "outcome.json").read_text())
            self.assertEqual(saved["recording"]["recorded_frames"], len(rows))
            self.assertTrue((run / "metrics.json").exists())
            _, _, replayed, _, coverage = replay_run.replay(run)
        self.assertEqual(coverage, 1.0)
        self.assertEqual([r["state"] for r in replayed], [r["state"] for r in rows])
        self.assertIn("coasting", [r["state"] for r in rows])
        for recorded, again in zip(rows, replayed):
            self.assertAlmostEqual(recorded["turn"], again["turn"], delta=0.5)
        self.assertGreater(max(r["turn"] for r in rows), 1)  # Left offset steered.

    def test_replay_report_and_mined_candidates(self):
        with tempfile.TemporaryDirectory() as folder:
            run, _ = self.record_run(folder)
            root = Path(folder).resolve()
            labels, output = root / "labels.csv", root / "out"
            with patch.object(replay_run, "LABELS", labels), patch.object(replay_run, "OUTPUT", output), \
                    patch.object(replay_run, "ROOT", root), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(replay_run.main([str(run), "--mine", "--set", "turn_rate=20"]), 0)
                with labels.open(newline="") as source:
                    first = list(csv.DictReader(source))
                replay_run.main([str(run), "--mine"])  # Already-known frames are not re-added.
                with labels.open(newline="") as source:
                    second = list(csv.DictReader(source))
            self.assertTrue((output / "run_test" / "events.jpg").exists())
            self.assertTrue((output / "run_test" / "candidates.jpg").exists())
        # Black frames 3-4 lost the lane; the synthetic sequence's instant
        # centered/offset switches (frames 1, 7, 8) look like detector jumps.
        kinds = {label["image"].split("_")[-1][:5]: label["source"] for label in first}
        self.assertEqual(kinds, {"00001": "interpolated (jump)", "00003": "interpolated (lost)",
                                 "00004": "interpolated (lost)", "00007": "interpolated (jump)",
                                 "00008": "interpolated (jump)"})
        self.assertTrue(all(label["status"] == "candidate" for label in first))
        self.assertEqual(len(second), len(first))

    def test_interpolated_label_between_tracked_neighbors(self):
        rows = [{"elapsed_s": 0.0, "state": "driving"}, {"elapsed_s": 0.04, "state": "coasting"},
                {"elapsed_s": 0.08, "state": "driving"}]
        detections = [(0, {"fit": (0.0, 0.50)}), (1, {"fit": None}), (2, {"fit": (0.0, 0.60)})]
        label = replay_run.interpolated(1, rows, detections, [True, False, True])
        self.assertAlmostEqual(label[0], 0.55)
        rows[2]["elapsed_s"] = 1.0  # Neighbor too far away: no label.
        self.assertIsNone(replay_run.interpolated(1, rows, detections, [True, False, True]))

    def test_metrics_table_lists_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            run, _ = self.record_run(folder)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(metrics_main([str(run.parent)]), 0)
        self.assertIn("run_test", output.getvalue())
        self.assertIn("tracked_pct", output.getvalue())


if __name__ == "__main__":
    unittest.main()
