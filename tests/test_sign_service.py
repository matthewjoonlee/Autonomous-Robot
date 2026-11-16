import unittest

import cv2
import numpy as np

from sign_service import SignTracker, arrow_direction, detect_sign

ARROW = np.array([[0, 30], [40, 0], [40, 18], [100, 18], [100, 42], [40, 42], [40, 60]])  # points left


def sign_image(symbol, size=150, at=(240, 110)):
    """Gray carpet-like scene with one course-style sign: white panel, black border, symbol."""
    img = np.full((360, 640, 3), (140, 140, 135), np.uint8)
    img += np.random.default_rng(0).integers(0, 12, img.shape, dtype=np.uint8)
    x, y = at
    w, h = size, int(size * 0.78)
    cv2.rectangle(img, (x, y), (x + w, y + h), (245, 245, 245), -1)
    cv2.rectangle(img, (x + 6, y + 6), (x + w - 6, y + h - 6), (25, 25, 25), 3)
    cx, cy, s = x + w // 2, y + h // 2, w / 160
    if symbol == "PAUSE":
        r = 0.32 * w
        pts = np.array([[cx + r * np.cos(a), cy + 0.8 * r * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 7)[:-1]])
        cv2.fillPoly(img, [pts.astype(np.int32)], (20, 20, 20))
        cv2.fillPoly(img, [((pts - [cx, cy]) * 0.85 + [cx, cy]).astype(np.int32)], (30, 30, 200))
    else:
        pts = ARROW.astype(float) * s
        if symbol == "RIGHT":
            pts[:, 0] = 100 * s - pts[:, 0]
        if symbol == "STRAIGHT":
            pts = pts[:, ::-1]
        pts -= pts.mean(axis=0)
        cv2.fillPoly(img, [(pts + [cx, cy]).astype(np.int32)], (15, 15, 15))
    return img, (x, y, w, h)


class ArrowDirectionTest(unittest.TestCase):
    def mask(self, pts, w, h):
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [pts.astype(np.int32)], 1)
        return m

    def test_directions(self):
        left = self.mask(ARROW, 101, 61)
        self.assertEqual(arrow_direction(left)[0], "LEFT")
        self.assertEqual(arrow_direction(left[:, ::-1])[0], "RIGHT")
        self.assertEqual(arrow_direction(left.T.copy())[0], "STRAIGHT")
        self.assertIsNone(arrow_direction(left.T[::-1].copy())[0])   # pointing down: not a course sign

    def test_plain_bar_is_not_an_arrow(self):
        self.assertIsNone(arrow_direction(np.ones((20, 80), np.uint8))[0])


class DetectSignTest(unittest.TestCase):
    def test_each_sign_label_and_box(self):
        for symbol in ("LEFT", "RIGHT", "STRAIGHT", "PAUSE"):
            img, (x, y, w, h) = sign_image(symbol)
            found = detect_sign(img)
            self.assertIsNotNone(found, symbol)
            self.assertEqual(found["label"], symbol)
            bx, by, bw, bh = found["box"]
            self.assertLess(abs(bx - x) + abs(by - y), 0.25 * w, symbol)     # box near the panel
            self.assertGreater(bw * bh, 0.5 * w * h, symbol)                 # encloses most of it

    def test_empty_scene(self):
        img, _ = sign_image("LEFT")
        img[:] = (140, 140, 135)
        self.assertIsNone(detect_sign(img))


class SignTrackerTest(unittest.TestCase):
    def test_needs_agreement_then_announces_once(self):
        tracker = SignTracker()
        left = {"label": "LEFT", "box": (1, 2, 3, 4)}
        changes = [tracker.update(d) for d in [left, None, left, left, left, left]]
        self.assertEqual([c for c in changes if c], ["LEFT"])
        self.assertEqual(tracker.label, "LEFT")

    def test_same_sign_again_after_removal_is_announced_again(self):
        tracker = SignTracker()
        left = {"label": "LEFT", "box": (1, 2, 3, 4)}
        changes = [tracker.update(d) for d in [left] * 4 + [None] * 12 + [left] * 4]
        self.assertEqual([c for c in changes if c], ["LEFT", "LEFT"])


if __name__ == "__main__":
    unittest.main()


class SignHoldTest(unittest.TestCase):
    def test_short_miss_keeps_the_label(self):
        tracker = SignTracker()
        left = {"label": "LEFT", "box": (1, 2, 3, 4)}
        changes = [tracker.update(d) for d in [left] * 4 + [None] * 4 + [left] * 4]
        self.assertEqual([c for c in changes if c], ["LEFT"])
