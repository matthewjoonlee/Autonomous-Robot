"""Sign detection service: which sign is in front of the robot, and where?

Rules-based OpenCV (Week 11, approach A). Course signs are a white panel with a
thin black printed border and one symbol: a black arrow (LEFT / RIGHT / STRAIGHT)
or a red hexagon (PAUSE).

1. Find the SYMBOL first, with strict shape tests (looking for a white panel first
   accepted pale carpet with dark chair legs inside):
   - PAUSE: a solid red blob shaped like a hexagon (convex, about 6 corners).
   - arrows: a dark blob with an arrow's outline - about 7 corners, a notch where
     the shaft meets the head (solidity 0.5-0.85), about 2:1 proportions, and one
     clearly wider end (the head). The head's end gives LEFT / RIGHT / STRAIGHT.
2. The symbol must be surrounded by white (the panel); chair legs, wheels and
   shadows are not.
3. The box grows from the symbol out to the panel's white area (bounded by the
   printed border), so it encloses the sign.
4. SignTracker turns per-frame results into a steady label: shown after 3 of the
   last 5 frames agree, held ~1 s when the sign briefly drops out.

Returns a box, label and drawing instructions; the controller decides what to
display and say. No motor commands.
"""

from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

LABELS = ("LEFT", "RIGHT", "STRAIGHT", "PAUSE")
SPOKEN = {"LEFT": "Left", "RIGHT": "Right", "STRAIGHT": "Straight", "PAUSE": "Pause"}
MIN_SYMBOL_AREA = 120   # pixels in the 640x360 front frame (a sign ~1.5 m away)
MIN_ARROW_AREA = 350    # arrows: larger, so chair wheels on bright carpet (~160 px) are not arrows
HEAD_RATIO = 1.35       # arrowhead height / shaft height needed to call a direction
# The sign's white panel can look tinted: in the Nov 15 lab, with the white balance
# locked at startup, it was light blue (saturation ~82), and the old limit of 80
# rejected nearly every real arrow. Brightness relative to the symbol does the work.
WHITE_MAX_SAT = 140
WHITE_SPREAD = 25       # max brightness spread (10th-90th percentile) of a panel's white
WHITE_SPREAD_PAUSE = 45


def _shape(contour):
    """Area, bounding box, solidity and corner count of a contour."""
    import cv2
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    hull = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / hull if hull else 0.0
    corners = len(cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True))
    return area, (x, y, w, h), solidity, corners


def arrow_direction(mask):
    """LEFT / RIGHT / STRAIGHT from a filled arrow mask, or None; plus the head ratio.

    The head is a triangle narrowing to its tip, so compare the TALLEST column (or
    widest row) of each half - the head is ~2x the shaft; averages hide it. Measure
    the lopsidedness both ways and use the stronger: the up arrow's wide head makes
    it nearly as wide as tall, so its proportions alone do not show its direction.
    """
    h, w = mask.shape
    columns, rows = mask.sum(axis=0).astype(float), mask.sum(axis=1).astype(float)
    left, right = np.percentile(columns[:w // 2], 90), np.percentile(columns[w - w // 2:], 90)
    top, bottom = np.percentile(rows[:h // 2], 90), np.percentile(rows[h - h // 2:], 90)
    across = max(left, right) / max(1.0, min(left, right))
    down = max(top, bottom) / max(1.0, min(top, bottom))
    if across >= down and across >= HEAD_RATIO:
        return ("LEFT" if left > right else "RIGHT"), across
    if down > across and down >= HEAD_RATIO and top > bottom:
        return "STRAIGHT", down
    return None, max(across, down)


def _dark_rim(hsv, contour, box):
    """Fraction of a thin band just outside the red blob that is dark (the hexagon's outline)."""
    import cv2
    x, y, w, h = box
    pad = max(3, int(0.12 * max(w, h)))
    height, width = hsv.shape[:2]
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(width, x + w + pad), min(height, y + h + pad)
    blob = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.drawContours(blob, [contour - [x0, y0]], -1, 1, -1)
    size = max(3, int(0.08 * max(w, h)))
    band = cv2.dilate(blob, np.ones((size, size), np.uint8)).astype(bool) & ~blob.astype(bool)
    val = hsv[y0:y1, x0:x1, 2]
    white = float(np.percentile(val, 90))
    return float((val[band] < 0.6 * white).mean()) if band.any() else 0.0


def _surrounded_by_white(hsv, box, symbol_value):
    """Fraction of a ring around the symbol that is panel-white."""
    x, y, w, h = box
    # Size the ring from the SHORT side: the up arrow fills most of the panel's
    # height, so a ring based on its length reached past the panel onto the carpet.
    pad = max(3, int(0.25 * min(w, h)))
    height, width = hsv.shape[:2]
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(width, x + w + pad), min(height, y + h + pad)
    region = hsv[y0:y1, x0:x1]
    ring = np.ones(region.shape[:2], bool)
    ring[max(0, y - y0 - 2):y - y0 + h + 2, max(0, x - x0 - 2):x - x0 + w + 2] = False
    edge = 4   # px: a symbol this close to the frame edge is cut off, not a whole sign
    if ring.sum() < 20 or x <= edge or y <= edge or x + w >= width - edge or y + h >= height - edge:
        return 0.0, None   # symbol cut off by the frame edge: not the whole sign
    sat, val = region[..., 1][ring], region[..., 2][ring]
    level = float(np.percentile(val, 70))
    # A dark arrow must be well below the panel's white; a red hexagon is already
    # bright (~165 vs white ~200), so for red symbols only low saturation counts.
    floor = 0.75 * level if symbol_value is None else max(0.75 * level, symbol_value + 60)
    white = (sat < WHITE_MAX_SAT) & (val > max(90.0, floor))
    if white.sum() > 10:
        # A sign panel is an EVEN white; bright room background around people and
        # chairs is not. Nov 15 lab: real sign 9-13 levels (10th-90th percentile),
        # false hits 34-92.
        low, high = np.percentile(val[white], [10, 90])
        # The red hexagon with its dark outline is distinctive on its own; the dim
        # Nov 12 room's real Pause sign had noisier white, so it gets more room.
        if high - low > (WHITE_SPREAD_PAUSE if symbol_value is None else WHITE_SPREAD):
            return 0.0, level
    return float(white.mean()), level


def _panel_box(hsv, box, level):
    """Grow from the symbol to the panel's white area; returns (x, y, w, h)."""
    import cv2
    x, y, w, h = box
    height, width = hsv.shape[:2]
    gx, gy = int(1.0 * w) + 4, int(1.0 * h) + 4
    x0, y0, x1, y1 = max(0, x - gx), max(0, y - gy), min(width, x + w + gx), min(height, y + h + gy)
    region = hsv[y0:y1, x0:x1]
    white = ((region[..., 1] < WHITE_MAX_SAT) & (region[..., 2] > 0.75 * level)).astype(np.uint8)
    white[y - y0:y - y0 + h, x - x0:x - x0 + w] = 1          # include the symbol itself
    count, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=4)
    symbol_label = labels[y - y0 + h // 2, x - x0 + w // 2]
    if symbol_label == 0:
        return box
    px, py, pw, ph, _ = stats[symbol_label]
    # Never let the panel exceed what a sign around this symbol can be.
    pw, ph = min(pw, int(2.2 * w) + 8), min(ph, int(2.6 * h) + 8)
    m = max(2, int(0.06 * max(pw, ph)))                      # include the printed border
    bx, by = max(0, x0 + px - m), max(0, y0 + py - m)
    return (bx, by, min(width - bx, pw + 2 * m), min(height - by, ph + 2 * m))


def _outer_contours(mask):
    """Outer outlines of every blob, including blobs inside another blob's hole.

    The arrow sits inside the sign's printed black border; RETR_EXTERNAL would
    return only the border and never the arrow.
    """
    import cv2
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    return [c for c, h in zip(contours, hierarchy[0]) if h[3] == -1]


def _symbols(frame, hsv):
    """Candidate (label, symbol box, score, details) from red and dark blobs."""
    import cv2
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    found = []
    red = (((hue < 10) | (hue > 165)) & (sat > 90) & (val > 50)).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    for contour in _outer_contours(red):
        area, box, solidity, corners = _shape(contour)
        aspect = box[2] / box[3]
        # Blur and sunlight soften the real hexagon (solidity 0.84, rim 0.22 seen).
        # The course hexagon is wider than tall (~1.1-1.4); a tall reddish cabinet at
        # the frame edge passed with aspect 0.8 (Nov 15 lab).
        if (area >= MIN_SYMBOL_AREA and solidity > 0.80 and 0.95 <= aspect <= 1.7 and 5 <= corners <= 10
                and _dark_rim(hsv, contour, box) >= 0.15):   # course hexagon has a dark outline
            found.append(("PAUSE", box, solidity, f"hexagon solidity {solidity:.2f}, {corners} corners",
                          None))
    bright = float(np.percentile(val, 90))
    dark = (val < min(100.0, 0.5 * bright)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    for contour in _outer_contours(dark):
        area, box, solidity, corners = _shape(contour)
        x, y, w, h = box
        long_side, short_side = max(w, h), min(w, h)
        # Course arrows: solidity ~0.78-0.86 (Nov 15 tests); chair/table shapes that
        # slipped through were 0.52-0.67.
        if (area < MIN_ARROW_AREA or not 0.68 <= solidity <= 0.90 or not 5 <= corners <= 10
                or not 1.1 <= long_side / short_side <= 3.2):
            continue
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [contour - [x, y]], -1, 1, -1)
        label, ratio = arrow_direction(mask)
        if label and ratio <= 4.0:   # real heads are ~2x the shaft
            found.append((label, box, min(1.0, (ratio - 1) / 2),
                          f"arrow head ratio {ratio:.2f}, solidity {solidity:.2f}, {corners} corners",
                          float(np.median(val[y:y + h, x:x + w][mask > 0]))))
    return found


def detect_sign(frame):
    """Best sign in a BGR frame: {"label", "box": (x, y, w, h), "score", "details"} or None."""
    import cv2
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    best = None
    for label, box, shape_score, details, symbol_value in _symbols(frame, hsv):
        white, level = _surrounded_by_white(hsv, box, symbol_value)
        if white < (0.55 if label == "PAUSE" else 0.60):
            continue
        size = min(1.0, box[2] * box[3] / 2500)
        score = round((0.4 + 0.6 * shape_score) * white * (0.5 + 0.5 * size), 3)
        if best is None or score > best["score"]:
            best = {"label": label, "box": tuple(int(v) for v in _panel_box(hsv, box, level)),
                    "symbol_box": tuple(int(v) for v in box), "score": score,
                    "details": f"{details}, white ring {white:.2f}"}
    return best


@dataclass
class SignTracker:
    """Steady label from noisy per-frame detections.

    A label is shown once `agree` of the last `window` frames say so. It stays
    shown (with its last box) until a different label becomes steady or no sign
    is seen for `hold` frames. `changed` is set on the frame a new label appears,
    including the same sign shown again after it was taken away, so every sign
    presentation is announced once.
    """
    window: int = 5
    agree: int = 3
    hold: int = 5     # frames (~0.5 s) before a vanished sign's label clears
    recent: deque = field(default_factory=lambda: deque(maxlen=5))
    label: str = None
    box: tuple = None
    missing: int = 0

    def update(self, detection):
        self.recent.append(detection["label"] if detection else None)
        if detection and detection["label"] == self.label:
            self.box = detection["box"]
        votes = Counter(x for x in self.recent if x)
        changed = None
        if votes:
            top, count = votes.most_common(1)[0]
            if count >= self.agree and top != self.label:
                self.label, changed, self.missing = top, top, 0
                if detection and detection["label"] == top:
                    self.box = detection["box"]
        if detection is None:
            self.missing += 1
            if self.missing >= self.hold:
                self.label = self.box = None
        else:
            self.missing = 0
        return changed


class SignService:
    """Frames in, SignResult out (the worker process wraps this)."""

    def __init__(self):
        self.tracker = SignTracker()

    def reset(self):
        self.tracker = SignTracker()

    def process(self, frame_id, timestamp, front, side=None):
        detection = detect_sign(front)
        changed = self.tracker.update(detection)
        label, box = self.tracker.label, self.tracker.box
        annotations = []
        if detection:  # raw per-frame candidate, thin
            x, y, w, h = detection["box"]
            annotations.append({"kind": "rect", "camera": "front", "xy": (x, y), "xy2": (x + w, y + h),
                                "color": (0, 200, 255), "thickness": 1})
        if label and box and self.tracker.missing <= 2:  # steady result, bold; gone within ~0.2 s
            x, y, w, h = box
            annotations.append({"kind": "rect", "camera": "front", "xy": (x - 2, y - 2),
                                "xy2": (x + w + 2, y + h + 2), "color": (0, 255, 0), "thickness": 3})
            annotations.append({"kind": "text", "camera": "front", "xy": (x, max(18, y - 8)),
                                "text": label, "color": (0, 255, 0), "scale": 0.8})
        return {"frame_id": frame_id, "timestamp": timestamp, "label": label, "box": box,
                "changed": changed, "raw": detection, "annotations": annotations}


def _worker_main(jobs, results, control):
    import queue
    from location_service import _put_latest
    service = SignService()
    while True:
        try:
            while True:
                message = control.get_nowait()
                if message == "stop":
                    return
                if message == "reset":
                    service.reset()
        except queue.Empty:
            pass
        try:
            frame_id, timestamp, front = jobs.get(timeout=0.1)
        except queue.Empty:
            continue
        try:
            _put_latest(results, service.process(frame_id, timestamp, front))
        except Exception as exc:
            _put_latest(results, {"frame_id": frame_id, "timestamp": timestamp, "error": repr(exc)})


class SignWorker:
    """SignService in its own process (same latest-frame pattern as LocationWorker).

    Results can be skipped under load, so the controller should announce when the
    steady label differs from what it last announced, not only on `changed`.
    """

    def __init__(self):
        import multiprocessing as mp
        context = mp.get_context("spawn")
        self.jobs = context.Queue(maxsize=1)
        self.results = context.Queue(maxsize=1)
        self.control = context.Queue()
        self.process = context.Process(target=_worker_main, daemon=True,
                                       args=(self.jobs, self.results, self.control))
        self.process.start()

    def submit(self, frame_id, timestamp, front):
        from location_service import _put_latest
        _put_latest(self.jobs, (frame_id, timestamp, front))

    def latest(self):
        import queue
        try:
            return self.results.get_nowait()
        except queue.Empty:
            return None

    def reset(self):
        self.control.put("reset")

    def close(self):
        self.control.put("stop")
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
        for q in (self.jobs, self.results, self.control):
            q.cancel_join_thread()
