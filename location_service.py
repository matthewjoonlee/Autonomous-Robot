"""Intersection location service: where is the robot relative to the intersection?

Week 10 design: this module reports evidence only. It never classifies signs,
decides to pause or turn, or commands motors; the controller does that.

How it works
- Front camera (full view, 640x360): how much grass lies beside the road in 12 row
  bands. The cross road shows as rows with no grass between nearer and farther
  grass; the gap moves down the frame as the robot approaches (+36 to ~+16 in).
- Side camera (right side, 640x480): grass and yellow-dash coverage in 8 vertical
  strips of the ground band. The corner curve, the cross road, its center dashes and
  the far corner pass the camera in order (+12 to -12 in).
- Distance: the robot center's distance from the cross-road center line, estimated
  as the median distance of the most similar frames in recordings with known
  positions (location_reference.npz, built by build_location_reference.py).
  Leave-one-run-out on the Nov 12 four-way runs: median error 0.3-0.5 in.
- Zones: ZoneTracker smooths the estimates and steps through the phases in order,
  never backwards and never skipping one, so every phase is reported once.

Zone boundaries (robot center, inches; + before the cross-road center line):
  approach > 24 > read_sign > 12 > pause > 6 > entering > 1.5 > turn > -1.5 > exiting > -6 > continue
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REFERENCE = Path(__file__).resolve().parent / "location_reference.npz"

PHASES = ("approach", "read_sign", "pause", "entering", "turn", "exiting", "continue")
LABELS = {
    "approach": "Continue travel",
    "read_sign": "Read the sign",
    "pause": "Pause zone",
    "entering": "Entering intersection",
    "turn": "Turn zone",
    "exiting": "Exiting intersection",
    "continue": "Continue travel",
}
BOUNDARIES = (24.0, 12.0, 6.0, 1.5, -1.5, -6.0)  # between consecutive PHASES

FRONT_BANDS = 12
SIDE_STRIPS = 8


def zone_index(distance):
    """Index into PHASES for a robot-center distance in inches."""
    return sum(distance < boundary for boundary in BOUNDARIES)


# ---------------------------------------------------------------- features

def front_features(frame):
    """Grass fraction beside the road (outer quarters) in 12 row bands, y 40%-100%."""
    import cv2
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = frame.shape[:2]
    hue, sat = hsv[..., 0], hsv[..., 1]
    green = (hue >= 30) & (hue < 75) & (sat > 40)
    values = []
    for k in range(FRONT_BANDS):
        rows = slice(int(height * (0.40 + 0.05 * k)), int(height * (0.45 + 0.05 * k)))
        values.append((green[rows, :width // 4].mean() + green[rows, 3 * width // 4:].mean()) / 2)
    return values


def side_masks(frame):
    """Grass and yellow-dash masks for the dim side camera.

    Grass is judged by color balance (green clearly above red and blue), not by fixed
    brightness, because the ground beside the robot is often in shade (brightness
    ~10-30 of 255 in the Nov 12 recordings).
    """
    import cv2
    blue, green, red = (channel.astype(np.int16) for channel in cv2.split(frame))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grass = (green - np.maximum(red, blue) > 6) & (green > 15)
    yellow = (hsv[..., 0] >= 15) & (hsv[..., 0] <= 40) & (hsv[..., 1] >= 80) & (hsv[..., 2] >= 35)
    return grass, yellow


def side_band(height):
    return slice(int(height * 0.55), int(height * 0.97))


def side_features(frame):
    """Grass fraction and (scaled, capped) yellow fraction in 8 strips of the ground band."""
    grass, yellow = side_masks(frame)
    height, width = frame.shape[:2]
    band, step = side_band(height), width // SIDE_STRIPS
    strips = [slice(j * step, (j + 1) * step) for j in range(SIDE_STRIPS)]
    return ([float(grass[band, s].mean()) for s in strips]
            + [min(1.0, float(yellow[band, s].mean()) * 20) for s in strips])


def features(front, side):
    """Feature vector for one frame pair (side may be None: side half set to NaN)."""
    side_part = side_features(side) if side is not None else [np.nan] * (2 * SIDE_STRIPS)
    return np.array(front_features(front) + side_part, dtype=np.float32)


# ---------------------------------------------------------------- distance model

class LocationModel:
    """Nearest-neighbor distance estimate against labeled reference frames."""

    def __init__(self, path=REFERENCE, neighbors=5):
        data = np.load(path)
        self.features = data["features"].astype(np.float32)
        self.distances = data["distances"].astype(np.float32)
        self.neighbors = neighbors

    def estimate(self, vector):
        """(distance in inches, spread in inches) - spread is the neighbors' disagreement."""
        usable = ~np.isnan(vector)
        cost = np.abs(self.features[:, usable] - vector[usable]).sum(axis=1)
        nearest = self.distances[np.argsort(cost)[:self.neighbors]]
        low, high = np.percentile(nearest, [25, 75])
        return float(np.median(nearest)), float(high - low)


# ---------------------------------------------------------------- zone tracking

@dataclass
class ZoneTracker:
    """Smooth distance estimates into an ordered sequence of phases.

    The robot drives forward through the intersection, so the zone only moves
    forward. It advances one phase at a time (each phase is reported once, even if
    the robot moves quickly) after `confirm` consecutive smoothed estimates are past
    the boundary by `margin` inches. Estimates whose neighbors disagree by more than
    `max_spread` inches are ignored.
    """
    confirm: int = 2
    margin: float = 0.3
    window: int = 2
    max_spread: float = 6.0
    zone: int = -1               # -1 until the first confident estimates
    recent: list = field(default_factory=list)
    streak: int = 0

    def update(self, distance, spread):
        """Feed one estimate; returns the new phase name if the zone changed, else None."""
        if distance is None or spread > self.max_spread:
            return None
        self.recent = (self.recent + [distance])[-self.window:]
        smoothed = float(np.median(self.recent))
        if self.zone < 0:
            if len(self.recent) < self.window:
                return None
            self.zone = zone_index(smoothed)
            return PHASES[self.zone]
        if self.zone + 1 < len(PHASES) and smoothed < BOUNDARIES[self.zone] - self.margin:
            self.streak += 1
            if self.streak >= self.confirm:
                self.zone += 1
                self.streak = 0
                return PHASES[self.zone]
        else:
            self.streak = 0
        return None

    @property
    def smoothed(self):
        return float(np.median(self.recent)) if self.recent else None

    @property
    def phase(self):
        return PHASES[self.zone] if self.zone >= 0 else None


# ---------------------------------------------------------------- result + annotations

def result(frame_id, timestamp, tracker, distance, spread):
    """LocationResult as a plain dict (crosses process boundaries cheaply)."""
    phase = tracker.phase
    smoothed = tracker.smoothed
    text = "LOCATION: searching" if phase is None else f"LOCATION: {LABELS[phase]}"
    if smoothed is not None:
        text += f"  ({smoothed:+.0f} in)"
    annotations = [
        {"kind": "text", "camera": "front", "xy": (8, 22), "text": text, "color": (0, 255, 255)},
        # Where the location evidence comes from, so judges can see what is measured.
        {"kind": "rect", "camera": "front", "xy": (0, 0.40), "xy2": (1.0, 1.0),
         "relative": True, "color": (255, 200, 0), "label": "front: cross-road gap"},
        {"kind": "rect", "camera": "side", "xy": (0, 0.55), "xy2": (1.0, 0.97),
         "relative": True, "color": (255, 200, 0), "label": "side: corner / cross road"},
    ]
    return {
        "frame_id": frame_id, "timestamp": timestamp, "zone": phase,
        "label": LABELS.get(phase, "Searching"),
        "distance_in": smoothed, "raw_distance_in": distance, "spread_in": spread,
        "sign_read_ready": phase == "read_sign",
        "pause_point_ready": phase == "pause",
        "turn_point_ready": phase == "turn",
        "annotations": annotations,
    }


class LocationService:
    """In-process service: frames in, LocationResult out. The worker wraps this."""

    def __init__(self, path=REFERENCE, **tracker_settings):
        self.model = LocationModel(path)
        self.tracker = ZoneTracker(**tracker_settings)

    def reset(self):
        self.tracker = ZoneTracker(confirm=self.tracker.confirm, margin=self.tracker.margin,
                                   window=self.tracker.window, max_spread=self.tracker.max_spread)

    def process(self, frame_id, timestamp, front, side):
        distance, spread = self.model.estimate(features(front, side))
        change = self.tracker.update(distance, spread)
        out = result(frame_id, timestamp, self.tracker, distance, spread)
        out["changed"] = change
        return out


# ---------------------------------------------------------------- worker process

def _put_latest(q, item):
    """Latest-frame pattern: replace an unread item instead of queueing stale work."""
    import queue
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


def _worker_main(jobs, results, control, path):
    import queue
    service = LocationService(path)
    while True:
        try:
            while True:  # control messages first, so a reset is never lost
                message = control.get_nowait()
                if message == "stop":
                    return
                if message == "reset":
                    service.reset()
        except queue.Empty:
            pass
        try:
            job = jobs.get(timeout=0.1)
        except queue.Empty:
            continue
        frame_id, timestamp, front, side = job
        try:
            _put_latest(results, service.process(frame_id, timestamp, front, side))
        except Exception as exc:  # report, keep serving
            _put_latest(results, {"frame_id": frame_id, "timestamp": timestamp, "error": repr(exc)})


class LocationWorker:
    """Runs LocationService in a separate Python process (Week 10 architecture).

    submit() hands over the newest frames without waiting; latest() returns the
    newest result, or None if nothing new. The worker never touches motors.
    """

    def __init__(self, path=REFERENCE):
        import multiprocessing as mp
        context = mp.get_context("spawn")  # do not inherit camera/motor handles
        self.jobs = context.Queue(maxsize=1)
        self.results = context.Queue(maxsize=1)
        self.control = context.Queue()
        self.process = context.Process(target=_worker_main, daemon=True,
                                       args=(self.jobs, self.results, self.control, str(path)))
        self.process.start()

    def submit(self, frame_id, timestamp, front, side):
        _put_latest(self.jobs, (frame_id, timestamp, front, side))

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
        # An unread frame left in a queue would otherwise block Python from exiting.
        for q in (self.jobs, self.results, self.control):
            q.cancel_join_thread()
