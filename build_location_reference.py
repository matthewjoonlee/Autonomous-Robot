"""Build location_reference.npz from recordings with trustworthy distance marks.

    python build_location_reference.py                 # the default recordings below
    python build_location_reference.py --check         # also leave-one-run-out accuracy

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
# Nov 12 four-way runs whose marks were checked against labeled still photos.
DEFAULT_RUNS = [
    "quiz3_recordings/four_way_20261112_165907",
    "quiz3_recordings/four_way_20261112_170234",
    "quiz3_recordings/four_way_stops_20261112_170622",
]


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
    for (f0, d0), (f1, d1) in zip(marks, marks[1:]):
        span = driving[f1] - driving[f0]
        for i in range(f0, min(f1, len(rows) - 1) + 1):
            distances[i] = d0 + (d1 - d0) * ((driving[i] - driving[f0]) / span if span else 0.0)
    first, d_first = marks[0]
    for i in range(first):
        if driving[first] == driving[i]:
            distances[i] = d_first
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", default=DEFAULT_RUNS)
    parser.add_argument("--output", type=Path, default=REFERENCE)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
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
