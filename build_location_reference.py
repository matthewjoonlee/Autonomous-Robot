"""Build location_reference.npz from recordings with trustworthy distance marks.

    python build_location_reference.py                 # the default recordings below
    python build_location_reference.py --check         # also leave-one-run-out accuracy
    python build_location_reference.py --add-latest --check
        # new room: the default runs plus your newest stop-style recording

A recording made in the quiz room teaches the service that room's lighting; the
leave-one-run-out check then shows how well the other rooms' runs predicted it.

Each recording (record_quiz3.py) has front.mp4, side.mp4, frames.csv and marks.csv.
Frames get robot-center distances by interpolating between marks in proportion to
*driving* frames (motor command forward), so stops, like the judging pauses, keep
their distance. Frames before the first or after the last mark count only while
the robot is parked at it.
"""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

from location_service import BOUNDARIES, REFERENCE, features, zone_index

ROOT = Path(__file__).resolve().parent
# Nov 15 lab passes, marked with the robot's FRONT edge on measured zone lines
# (+7.1 in to the center). Lab-only: the Nov 12 runs (other room, center eyeballed)
# disagree with the lab marks by ~3 in at the first mark (the cross road sits at
# row 0.86 there vs below the frame in the lab), and mixing them put the early
# zones off. Lab-only held-out: median error 0.4-0.6 in, zone exact 93-99%.
# (four_way_stops_20261115_164039 is a false start: marks seconds apart.)
DEFAULT_RUNS = [
    "quiz3_recordings/four_way_stops_20261115_164159",
    "quiz3_recordings/four_way_stops_20261115_164702",
    "quiz3_recordings/four_way_stops_20261115_164828",
    # Nov 16, the sunny lobby (exam venue), front-edge tape marks. (lobby_stops_20261116_152119
    # is left out: the side camera recorded black frames.)
    "quiz3_recordings/lobby_stops_20261116_152506",
]
# Original-room runs, for reference or another room: build with --add <run>.
NOV12_RUNS = [
    "quiz3_recordings/four_way_20261112_165907",
    "quiz3_recordings/four_way_20261112_170234",
    "quiz3_recordings/four_way_stops_20261112_170622",
]


BEFORE_LIMIT = 15.0  # inches extrapolated before the first mark


def frame_distances(folder):
    """Robot-center distance (inches) for each frame, or None where unknown."""
    folder = Path(folder)
    rows = list(csv.DictReader((folder / "frames.csv").open()))
    marks = [(int(r["frame"]), float(r["distance_in"]))
             for r in csv.DictReader((folder / "marks.csv").open()) if r["distance_in"]]
    if len(marks) < 2:
        raise ValueError(f"{folder}: need at least two distance marks")
    driving = [0]
    for row in rows:
        driving.append(driving[-1] + (float(row.get("speed") or 0) > 0))
    distances = [None] * len(rows)
    if driving[-1] == 0:
        # Pushed by hand (no motor commands logged): place frames between marks by
        # frame count instead, and nothing outside the marks. Push steadily and mark
        # every few inches.
        for (f0, d0), (f1, d1) in zip(marks, marks[1:]):
            for i in range(f0, min(f1, len(rows) - 1) + 1):
                distances[i] = d0 + (d1 - d0) * (i - f0) / max(1, f1 - f0)
        return distances
    for (f0, d0), (f1, d1) in zip(marks, marks[1:]):
        span = driving[f1] - driving[f0]
        for i in range(f0, min(f1, len(rows) - 1) + 1):
            distances[i] = d0 + (d1 - d0) * ((driving[i] - driving[f0]) / span if span else 0.0)
    first, d_first = marks[0]
    # Before the first mark: parked frames are at it; frames driven toward it get
    # distances from the run's own speed (inches per driving frame between marks),
    # up to `BEFORE_LIMIT` inches back. Covers starts from the straight mat before.
    (f_a, d_a), (f_b, d_b) = marks[0], marks[-1]
    per_frame = (d_a - d_b) / max(1, driving[f_b] - driving[f_a])
    for i in range(first):
        back = (driving[first] - driving[i]) * per_frame
        if back <= BEFORE_LIMIT:
            distances[i] = d_first + back
    last, d_last = marks[-1]
    for i in range(last, len(rows)):
        if driving[i] == driving[last]:
            distances[i] = d_last
    return distances


def read_video(path):
    import cv2
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            return frames
        frames.append(frame)


def run_dataset(folder):
    folder = ROOT / folder
    distances = frame_distances(folder)
    fronts, sides = read_video(folder / "front.mp4"), read_video(folder / "side.mp4")
    X, Y = [], []
    for front, side, distance in zip(fronts, sides, distances):
        if distance is not None:
            X.append(features(front, side))
            Y.append(distance)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def check(datasets, names, neighbors=5):
    """Leave-one-run-out: each run estimated only from the others."""
    from location_service import ZoneTracker
    for test, name in enumerate(names):
        Xt, Yt = datasets[test]
        Xr = np.vstack([d[0] for i, d in enumerate(datasets) if i != test])
        Yr = np.hstack([d[1] for i, d in enumerate(datasets) if i != test])
        estimates = []
        tracker, zones = ZoneTracker(), []
        for vector in Xt:
            cost = np.abs(Xr - vector).sum(axis=1)
            near = Yr[np.argsort(cost)[:neighbors]]
            low, high = np.percentile(near, [25, 75])
            estimates.append(float(np.median(near)))
            tracker.update(estimates[-1], high - low)
            zones.append(tracker.zone)
        estimates, zones = np.array(estimates), np.array(zones)
        truth = np.array([zone_index(y) for y in Yt])
        error = np.abs(estimates - Yt)
        print(f"{Path(name).name}: distance error median {np.median(error):.1f} in, "
              f"90% {np.percentile(error, 90):.1f} in; tracked zone exact "
              f"{np.mean(zones == truth) * 100:.0f}%, within one {np.mean(abs(zones - truth) <= 1) * 100:.0f}%")
        # Where the tracker's zone changes vs where the true boundary is.
        for z in range(1, len(BOUNDARIES) + 1):
            hits = np.flatnonzero(zones == z)
            if len(hits):
                print(f"    entered zone {z} at true distance {Yt[hits[0]]:+5.1f} in "
                      f"(boundary {BOUNDARIES[z - 1]:+.1f})")


def newest_marked_recording():
    """Newest quiz3_recordings/* folder whose marks.csv has two or more distances."""
    folders = sorted((ROOT / "quiz3_recordings").glob("*/marks.csv"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    for marks in folders:
        rows = [r for r in csv.DictReader(marks.open()) if r.get("distance_in")]
        if len(rows) >= 2 and (marks.parent / "front.mp4").exists():
            return str(marks.parent.relative_to(ROOT))
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", default=None, help="recordings (default: the Nov 12 runs)")
    parser.add_argument("--add", nargs="+", default=[], metavar="RUN",
                        help="recordings to use in addition to the defaults")
    parser.add_argument("--add-latest", action="store_true",
                        help="also use the newest quiz3_recordings folder with distance marks")
    parser.add_argument("--output", type=Path, default=REFERENCE)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    runs = list(args.runs or DEFAULT_RUNS) + list(args.add)
    if args.add_latest:
        latest = newest_marked_recording()
        if latest is None:
            parser.error("no recording in quiz3_recordings/ has two or more distance marks")
        if latest not in runs:
            runs.append(latest)
        print(f"Adding newest recording: {latest}")
    args.runs = runs
    datasets = [run_dataset(run) for run in args.runs]
    X = np.vstack([d[0] for d in datasets])
    Y = np.hstack([d[1] for d in datasets])
    run_index = np.hstack([np.full(len(d[1]), i) for i, d in enumerate(datasets)])
    np.savez_compressed(args.output, features=X, distances=Y, run=run_index,
                        runs=np.array([str(r) for r in args.runs]))
    print(f"{len(Y)} labeled frames from {len(args.runs)} runs, {Y.max():+.0f} to {Y.min():+.0f} in "
          f"-> {args.output}")
    if args.check:
        check(datasets, args.runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
