"""Every confirmed frame in regression/labels.csv must keep passing.

The set grows from real runs (replay_run.py --mine, then human review), so a
detector change can only ship if it still handles every failure seen before.
Blank near_x/far_x means the lane cannot be measured and must not be tracked.
"""

import csv
from pathlib import Path
import unittest

from track_vision import FAR_Y, NEAR_Y, detect_lane, read_image

ROOT = Path(__file__).resolve().parents[1]
NEAR_TOLERANCE_PX, FAR_TOLERANCE_PX = 10, 15


class RegressionFrameTests(unittest.TestCase):
    def test_confirmed_frames(self):
        with (ROOT / "regression/labels.csv").open(newline="") as source:
            labels = [label for label in csv.DictReader(source) if label["status"] == "confirmed"]
        self.assertGreaterEqual(len(labels), 10)
        for label in labels:
            with self.subTest(image=label["image"], note=label["note"]):
                frame = read_image(ROOT / label["image"])
                detection = detect_lane(frame)
                if not label["near_x"]:
                    self.assertNotEqual(detection["status"], "tracked")
                    continue
                self.assertEqual(detection["status"], "tracked")
                slope, intercept = detection["fit"]
                width = frame.shape[1]
                self.assertLess(abs(slope * NEAR_Y + intercept - float(label["near_x"])) * width,
                                NEAR_TOLERANCE_PX)
                self.assertLess(abs(slope * FAR_Y + intercept - float(label["far_x"])) * width,
                                FAR_TOLERANCE_PX)


if __name__ == "__main__":
    unittest.main()
