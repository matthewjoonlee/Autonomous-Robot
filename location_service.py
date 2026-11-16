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

Zone boundaries (inches; + before the cross-road center line; see BOUNDARIES):
  approach > 25.2 > read_sign > 10.1 > pause > 6 > entering > 1.5 > turn > -1.5 > exiting > -6.1 > continue
Zones are judged at the robot center unless ZoneTracker.offset moves the point
forward (quiz3_app --zone-ref front).
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
# Between consecutive PHASES, inches from the cross-road center line (+ before it).
# Nov 15: scaled from the Week 10 zone diagram onto the course mat (59 in road
# length; cross-road center ~29.25 in from the bottom edge).
BOUNDARIES = (25.2, 10.1, 6.0, 1.5, -1.5, -6.1)

FRONT_BANDS = 12
FAMILIAR_FACTOR = 2.0   # views up to this x the typical cross-run match cost are trusted
SIDE_STRIPS = 8


def zone_index(distance):
    """Index into PHASES for a robot-center distance in inches."""
    return sum(distance < boundary for boundary in BOUNDARIES)


# ---------------------------------------------------------------- features

def front_sides(frame):
    """(left, right) grass fractions beside the road (outer quarters) in 12 row bands."""
    import cv2
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = frame.shape[:2]
    hue, sat = hsv[..., 0], hsv[..., 1]
    green = (hue >= 30) & (hue < 75) & (sat > 40)
    left, right = [], []
    for k in range(FRONT_BANDS):
        rows = slice(int(height * (0.40 + 0.05 * k)), int(height * (0.45 + 0.05 * k)))
        left.append(float(green[rows, :width // 4].mean()))
        right.append(float(green[rows, 3 * width // 4:].mean()))
    return left, right


def cross_road_ahead(left, right):
    """True when the front camera sees the cross road still ahead: a row band with no
    grass beside the road, grass nearer to the robot, and grass beyond it.

    Nov 12 runs: seen only with the robot center more than 32 in from the cross-road
    center line, never closer. Colors of the front camera held up in a new room.
    """
    grass = [min(a, b) for a, b in zip(left, right)]
    for k in range(len(grass) - 1):
        if grass[k] < 0.08 and max(grass[k + 1:]) > 0.35 and max(grass[:k] or [0.0]) > 0.15:
            return True
    return False


def front_features(frame):
    """Grass beside the road per row band, on whichever side has LESS grass.

    The cross road shows as a band with no grass beside the road. Taking the side
    with less grass finds that opening whether the mat branches left, right, or
    both, so the four-way recordings also describe T and corner mats. (Averaging
    both sides, as before, halved the signal on one-sided mats.)
    """
    left, right = front_sides(frame)
    return [min(a, b) for a, b in zip(left, right)]


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
    """Nearest-neighbor distance estimate against labeled reference frames.

    `cost` measures how unlike the reference a frame is (mean feature difference to
    its nearest neighbors). Frames from a room or distance the reference never saw
    match badly; their estimates are guesses, so the service ignores them.
    Nov 15: reference runs vs each other 1.06 median / 2.64 at 99%; a brighter lab
    with a washed-out side camera 3.06 median - it announced wrong zones.
    """

    def __init__(self, path=REFERENCE, neighbors=5):
        data = np.load(path)
        self.features = data["features"].astype(np.float32)
        self.distances = data["distances"].astype(np.float32)
        self.neighbors = neighbors
        self.max_cost = self._familiar_limit(data["run"] if "run" in data else None)

    def _familiar_limit(self, runs):
        """Typical-run 99th percentile of frame cost against the OTHER runs.

        Per run, then the median across runs: a run from a very different room
        (Nov 15 lab vs Nov 12 room) matches the others badly and would otherwise
        loosen the limit for everyone (2.6 -> 4.8).
        """
        if runs is None or len(set(runs.tolist())) < 2:
            return 2.6
        per_run = {}
        for i in range(0, len(self.features), 3):
            other = self.features[runs != runs[i]]
            cost = np.abs(other - self.features[i]).sum(axis=1)
            per_run.setdefault(int(runs[i]), []).append(np.sort(cost)[:self.neighbors].mean())
        typical = float(np.median([np.percentile(c, 99) for c in per_run.values()]))
        # x2: in a calibrated room, correct estimates scored 2.5-4.5 against the
        # 2.4 limit and were thrown away, delaying every zone (Nov 15 lab test run).
        return max(2.0, FAMILIAR_FACTOR * typical)

    def estimate(self, vector):
        """(distance in inches, spread in inches, cost) - spread is the neighbors' disagreement."""
        usable = ~np.isnan(vector)
        cost = np.abs(self.features[:, usable] - vector[usable]).sum(axis=1)
        order = np.argsort(cost)[:self.neighbors]
        nearest = self.distances[order]
        low, high = np.percentile(nearest, [25, 75])
        return float(np.median(nearest)), float(high - low), float(cost[order].mean())


# ---------------------------------------------------------------- zone tracking

@dataclass
class ZoneTracker:
    """Smooth distance estimates into an ordered sequence of phases.

    The robot drives forward through the intersection, so the zone only moves
    forward. It advances one phase at a time (each phase is reported once, even if
    the robot moves quickly) after `confirm` consecutive smoothed estimates are past
    the boundary by `margin` inches (negative: before it, to announce on the line). Estimates whose neighbors disagree by more than
    `max_spread` inches are ignored.

    `offset`: how far the point judged for zones sits ahead of the robot center
    (inches). 0 = robot center; half the robot length = its front edge. Estimates
    are of the center (the recordings marked the center), so zones use
    distance - offset.
    """
    confirm: int = 2
    # Inches past a boundary before reporting it; NEGATIVE = report this far before the
    # estimated line, to cancel the delay of smoothing + confirmation (~2 frames, ~0.8 in).
    # Held-out on 6 runs (33 zone changes): margin +0.8 announced a median 1.7 in past
    # the line; -0.5 announces a median 0.4 in past it (10%-90%: 0.6 before to 1.1 past).
    margin: float = -0.5
    window: int = 2
    max_spread: float = 6.0
    offset: float = 0.0
    # The robot only drives forward, so an estimate this many inches FARTHER than the
    # recent ones is a misread (lab, Nov 15: the next intersection down the track made
    # frames at +10 read as +40, stalling Entering by 4 in). Accepted only if
    # `backward_confirm` such readings in a row agree (the robot really was moved back).
    max_backward: float = 6.0
    backward_confirm: int = 3
    backward: list = field(default_factory=list)
    zone: int = -1               # -1 until the first confident estimates
    recent: list = field(default_factory=list)
    streak: int = 0

    def update(self, distance, spread):
        """Feed one estimate; returns the new phase name if the zone changed, else None."""
        if distance is None or spread > self.max_spread:
            return None
        if self.recent and distance > max(self.recent) + self.max_backward:
            self.backward = (self.backward + [distance])[-self.backward_confirm:]
            consistent = (len(self.backward) == self.backward_confirm
                          and max(self.backward) - min(self.backward) < self.max_backward)
            if not consistent:
                return None
            self.recent = []          # the robot really moved back: start smoothing over
        self.backward = []
        self.recent = (self.recent + [distance])[-self.window:]
        smoothed = float(np.median(self.recent)) - self.offset
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

def result(frame_id, timestamp, tracker, distance, spread, familiar=True, cost=None):
    """LocationResult as a plain dict (crosses process boundaries cheaply)."""
    phase = tracker.phase
    smoothed = tracker.smoothed
    text = "LOCATION: searching" if phase is None else f"LOCATION: {LABELS[phase]}"
    if smoothed is not None:
        text += f"  ({smoothed:+.0f} in)"
    if not familiar:
        text += "  [unfamiliar view]"
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
        "familiar": familiar, "cost": cost,
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
        t = self.tracker
        self.tracker = ZoneTracker(confirm=t.confirm, margin=t.margin, window=t.window,
                                   max_spread=t.max_spread, offset=t.offset,
                                   max_backward=t.max_backward, backward_confirm=t.backward_confirm)

    def process(self, frame_id, timestamp, front, side):
        distance, spread, cost = self.model.estimate(features(front, side))
        familiar = cost <= self.model.max_cost
        # (A "cross road still ahead" veto was tried Nov 15 and removed: on tracks
        # with several intersections the NEXT cross road is visible while the robot
        # is at this one, and the veto discarded good estimates.)
        change = self.tracker.update(distance if familiar else None, spread)
        out = result(frame_id, timestamp, self.tracker, distance, spread, familiar, round(cost, 2))
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


def _worker_main(jobs, results, control, path, offset=0.0):
    import queue
    service = LocationService(path, offset=offset)
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

    def __init__(self, path=REFERENCE, offset=0.0):
        import multiprocessing as mp
        context = mp.get_context("spawn")  # do not inherit camera/motor handles
        self.jobs = context.Queue(maxsize=1)
        self.results = context.Queue(maxsize=1)
        self.control = context.Queue()
        self.process = context.Process(target=_worker_main, daemon=True,
                                       args=(self.jobs, self.results, self.control, str(path), offset))
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


# ---------------------------------------------------------------- fused position

class PositionFuser:
    """Robot FRONT position (inches) from driving distance, corrected by camera readings.

    Nov 15 lab: far out (front > ~20 in) the camera estimates tracked the robot to
    about an inch, but near the intersection they misread (+40 for the next
    intersection) or stuck at calibration-stop values. Driving distance (speed x time
    the motors drive) is smooth but drifts when the robot is slowed or blocked (one
    run "drove" 9 in while barely moving). So:

    - the position always advances by speed x driving time (stops do not count);
    - while the position is beyond `camera_until`, each camera reading within `gate`
      inches nudges it by `gain`; readings outside the gate are ignored, unless
      `reinit_n` in a row agree with each other - then the position jumps to them
      (bad start, or the robot did not move as commanded);
    - with `measure_speed`, the speed is re-measured from the accepted camera readings
      (robust slope over the last few seconds); off by default because those
      readings gave 3.5-4.5 in/s run to run, while 4.0 matched the calibration
      passes (0.38-0.42 in per frame at 10 fps) and the best test run;
    - the first position is the median of the first `init_n` readings at or beyond
      `camera_until` (the robot starts on the approach; a first reading stuck at a
      close calibration-stop value would otherwise lock in, since close-in readings
      are not used for corrections). Close readings start it only after
      `init_timeout` seconds of driving without far ones (a genuinely close start).
    """

    def __init__(self, speed=4.0, gain=0.25, gate=5.0, reinit_n=6, camera_until=20.0,
                 init_n=5, speed_window=4.0, speed_limits=(2.5, 6.0), measure_speed=False,
                 init_timeout=3.0):
        self.speed, self.default_speed = speed, speed
        self.gain, self.gate, self.reinit_n = gain, gate, reinit_n
        self.camera_until, self.init_n = camera_until, init_n
        self.speed_window, self.speed_limits = speed_window, speed_limits
        self.measure_speed = measure_speed
        self.x = None          # front position at drive time self.t
        self.t = 0.0
        self.pending = []      # far readings before the first position
        self.pending_close = []
        self.init_timeout = init_timeout
        self.outliers = []
        self.accepted = []     # (drive time, reading) used to measure speed

    def position(self, drive_time):
        """Front position now (advances by speed x driving time since the last update)."""
        if self.x is None:
            return None
        self.x -= self.speed * max(0.0, drive_time - self.t)
        self.t = max(self.t, drive_time)
        return self.x

    def observe(self, reading, drive_time):
        """Feed one camera front reading (None = no usable reading)."""
        x = self.position(drive_time)
        if reading is None:
            return x
        if x is None:
            if reading >= self.camera_until:
                self.pending.append((drive_time, reading))
                source = self.pending
            else:
                self.pending_close.append((drive_time, reading))
                source = self.pending_close if drive_time >= self.init_timeout else []
            if len(source) >= self.init_n:
                times, values = zip(*source[-self.init_n:])
                self.x = float(np.median(values)) - self.speed * (drive_time - float(np.mean(times)))
                self.t = drive_time
                self.accepted = list(source[-self.init_n:])
                self.pending, self.pending_close = [], []
            return self.x
        if x < self.camera_until:
            return x
        if abs(reading - x) <= self.gate:
            self.x += self.gain * (reading - x)
            self.outliers = []
            self.accepted.append((drive_time, reading))
            if self.measure_speed:
                self._measure_speed(drive_time)
        else:
            self.outliers.append(reading)
            recent = self.outliers[-self.reinit_n:]
            # Reset only to a position FARTHER back (the robot moved less than its
            # driving time says: blocked, slow, bad start). Runs of CLOSER readings are
            # misreads: in the lobby, readings flipped between the true ~43 in and a
            # false ~11 in, and resetting to them announced Entering at the start.
            if (len(recent) == self.reinit_n and max(recent) - min(recent) < self.gate
                    and float(np.median(recent)) > x):
                self.x = float(np.median(recent))
                self.outliers, self.accepted = [], [(drive_time, self.x)]
        return self.x

    def _measure_speed(self, now):
        points = [(t, v) for t, v in self.accepted if t >= now - self.speed_window]
        times = np.array([t for t, _ in points])
        if len(points) < 10 or np.ptp(times) < 1.5:
            return
        values = np.array([v for _, v in points])
        # Theil-Sen: median of pairwise slopes, robust to the odd misread.
        i, j = np.triu_indices(len(points), 1)
        dt = times[j] - times[i]
        keep = dt > 0.3
        if keep.sum() < 10:
            return
        slope = -float(np.median((values[j] - values[i])[keep] / dt[keep]))
        low, high = self.speed_limits
        if low <= slope <= high:
            self.speed = slope
