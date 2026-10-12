"""Replay recorded runs through the current lane detector and controller.

    python replay_run.py centering_results/run_X [run ...] [--set KEY=VALUE ...] [--video] [--mine]

Recorded frames, IMU headings, and timestamps go through the same Controller
the robot uses, then recorded and replayed scorecards are compared. Replay is
open loop: the robot's recorded motion is fixed, so it shows what the current
code would measure and command for exactly what the camera saw, not where the
robot would have gone. Outputs go to replay_results/<run>/ (not committed):
replay.csv, events.jpg (lost-lane and jump frames), and optionally replay.mp4.

--mine proposes regression labels for frames the detector missed or jumped
on, interpolated from neighboring tracked frames. They are appended to
regression/labels.csv as `candidate`; review candidates.jpg and change the
status to `confirmed` (or `rejected`) to make them permanent tests.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

from quiz3_autonomous import TUNABLE, Controller, Settings, new_row, tuned
from run_metrics import COLUMNS, number, score
from track_vision import FAR_Y, NEAR_Y, annotate, detect_lane, read_image

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "replay_results"
LABELS = ROOT / "regression/labels.csv"
LABEL_FIELDS = ("image", "near_x", "far_x", "status", "source", "note")
JUMP_PX = 15        # Tracked frame this far from its neighbors' interpolation is suspect.
BRACKET_S = 0.3     # Neighbor search window for interpolated labels.


def load_run(run):
    with (run / "readings.csv").open(newline="") as log:
        rows = list(csv.DictReader(log))
    outcome = json.loads((run / "outcome.json").read_text()) if (run / "outcome.json").exists() else {}
    images = {}
    for folder, pattern in (("raw", "frame_*.png"), ("frames", "frame_*.jpg")):
        for path in (run / folder).glob(pattern):
            images[int(path.stem.split("_")[1])] = path
    return rows, outcome, images


def replay(run, overrides=()):
    """Return (settings, reference, replayed rows, per-row detections, coverage)."""
    rows, outcome, images = load_run(run)
    recorded = outcome.get("settings", {})
    # Current code's settings; only the recorded speed is kept, since it matches the recorded motion.
    settings = tuned(Settings(speed=recorded.get("speed", 25)), overrides)
    if (run / "reference_raw.png").exists():
        # Older runs captured a centered reference; re-detect it with the current detector.
        reference_frame = read_image(run / "reference_raw.png")
        reference = detect_lane(reference_frame)
        if reference["fit"] is None:
            raise RuntimeError(f"{run.name}: reference frame has no lane")
        reference["width"] = reference_frame.shape[1]
    else:
        # Newer runs use the calibrated center saved in reference.json.
        saved = json.loads((run / "reference.json").read_text())
        reference = {"fit": tuple(saved["fit"]), "width": saved["width"]}
    from lane_center import load as load_center
    controller = Controller(reference, 0.0, settings, load_center())
    replayed, detections = [], []
    for recorded_row in rows:
        frame_id = int(recorded_row["frame_id"])
        elapsed, heading = number(recorded_row["elapsed_s"]), number(recorded_row.get("imu_delta_deg"))
        if frame_id not in images or elapsed is None or heading is None:
            continue
        frame = read_image(images[frame_id])
        capture = elapsed - (number(recorded_row.get("processing_s")) or 0.0)
        detection, summary = controller.observe(frame, heading, capture)
        row = new_row(elapsed, frame_id, number(recorded_row.get("processing_s")) or 0.0, summary)
        replayed.append(row)
        detections.append((frame_id, detection))
        try:
            controller.decide(detection, summary, heading, elapsed, row)
        except RuntimeError:
            break
    coverage = len(replayed) / max(1, len(rows))
    return settings, reference, replayed, detections, coverage


def near_far(fit):
    return fit[0] * NEAR_Y + fit[1], fit[0] * FAR_Y + fit[1]


def interpolated(index, rows, detections, trusted):
    """Lane (near_x, far_x) at row `index` from trusted neighbors within BRACKET_S."""
    time = rows[index]["elapsed_s"]
    before = next((j for j in range(index - 1, -1, -1) if trusted[j]), None)
    after = next((j for j in range(index + 1, len(rows)) if trusted[j]), None)
    if before is None or after is None:
        return None
    t0, t1 = rows[before]["elapsed_s"], rows[after]["elapsed_s"]
    if time - t0 > BRACKET_S or t1 - time > BRACKET_S:
        return None
    a, b = near_far(detections[before][1]["fit"]), near_far(detections[after][1]["fit"])
    weight = (time - t0) / (t1 - t0)
    return tuple(x0 + weight * (x1 - x0) for x0, x1 in zip(a, b))


def events(rows, detections, settings):
    """(index, kind, interpolated label or None) for lost-lane and jump frames."""
    trusted = [row["state"] == "driving" for row in rows]
    found = []
    for index, row in enumerate(rows):
        label = interpolated(index, rows, detections, trusted)
        if row["state"] == "searching":
            found.append((index, "lost", label))
        elif trusted[index] and label is not None:
            near = near_far(detections[index][1]["fit"])[0]
            if abs(near - label[0]) * 640 > JUMP_PX:
                found.append((index, "jump", label))
    return found


def overlay(frame, detection, reference, row, label=None, kind=None):
    from track_vision import summarize
    canvas = annotate(frame, detection, reference, summarize(detection, reference, number(row.get("imu_delta_deg"))))
    height, width = canvas.shape[:2]
    if label is not None:
        points = [(int(x * width), int(y * height)) for x, y in zip(label, (NEAR_Y, FAR_Y))]
        cv2.line(canvas, points[0], points[1], (255, 0, 255), 2)
    text = (f"#{row['frame_id']} t={row['elapsed_s']:.2f}s {kind or row['state']} speed {row['speed']} "
            f"turn {row['turn']:+.1f} lateral {row['lateral_error']} lost {row['lost_s']}s")
    cv2.rectangle(canvas, (0, height - 24), (width, height), (0, 0, 0), -1)
    cv2.putText(canvas, text, (6, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def contact_sheet(tiles, path, columns=4, size=(320, 240)):
    if not tiles:
        return False
    tiles = [cv2.resize(tile, size) for tile in tiles]
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % columns)
    rows = [np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)]
    return cv2.imwrite(str(path), np.vstack(rows))


def read_labels():
    if not LABELS.exists():
        return []
    with LABELS.open(newline="") as source:
        return list(csv.DictReader(source))


def write_labels(labels):
    LABELS.parent.mkdir(exist_ok=True)
    with LABELS.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(labels)


def report(run, overrides, video=False, mine=False, limit=24):
    rows, outcome, images = load_run(run)
    settings, reference, replayed, detections, coverage = replay(run, overrides)
    recorded_score = score(rows, outcome.get("settings", {}).get("min_confidence", 0.5))
    replay_score = score(replayed, settings.min_confidence)
    out = OUTPUT / run.name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "replay.csv").open("w", newline="") as log:
        writer = csv.DictWriter(log, fieldnames=list(replayed[0]) if replayed else ["frame_id"])
        writer.writeheader()
        writer.writerows(replayed)

    print(f"\n{run.relative_to(ROOT) if run.is_relative_to(ROOT) else run}"
          f"  trial={outcome.get('trial')}  ended: {outcome.get('reason')}")
    every = (outcome.get("recording") or {}).get("record_every", 1)
    if coverage * every < 0.95:
        print(f"  Only {coverage:.0%} of frames were recorded; lost-lane timing in replay is approximate.")
    elif every > 1:
        print(f"  Every {every}th frame was recorded; replay steps between those frames.")
    changed = {key: getattr(settings, key) for key in TUNABLE + ("max_turn", "abort_heading")
               if outcome.get("settings", {}).get(key) not in (None, getattr(settings, key))}
    if changed:
        print(f"  Replay settings differ from the run: {changed}")
    print(f"  {'metric':18s} {'recorded':>10s} {'replay':>10s}")
    for key in COLUMNS:
        print(f"  {key:18s} {str(recorded_score.get(key)):>10s} {str(replay_score.get(key)):>10s}")
    recorded_by_id = {int(row["frame_id"]): row for row in rows}
    disagreements = sum(1 for row in replayed
                        if recorded_by_id[row["frame_id"]].get("state") != row["state"]
                        or abs((number(recorded_by_id[row["frame_id"]].get("physical_turn")) or 0)
                               - row["physical_turn"]) >= 1)
    print(f"  replayed {len(replayed)} frames; {disagreements} differ from the recorded state/turn")

    found = events(replayed, detections, settings)
    tiles = []
    for index, kind, label in found[:limit]:
        frame_id, detection = detections[index]
        tiles.append(overlay(read_image(images[frame_id]), detection, reference, replayed[index], label, kind))
    if contact_sheet(tiles, out / "events.jpg"):
        print(f"  {len(found)} lost/jump frames; first {len(tiles)} in {out / 'events.jpg'}")

    if video and replayed:
        writer = cv2.VideoWriter(str(out / "replay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 replay_score.get("fps") or 20, (640, 480))
        for (frame_id, detection), row in zip(detections, replayed):
            writer.write(cv2.resize(overlay(read_image(images[frame_id]), detection, reference, row), (640, 480)))
        writer.release()
        print(f"  video: {out / 'replay.mp4'}")

    if mine and not run.is_relative_to(ROOT):
        print("  --mine skipped: labels need runs inside this repository")
    elif mine:
        labels = read_labels()
        known = {label["image"] for label in labels}
        added, previews = 0, []
        for index, kind, label in found:
            frame_id, detection = detections[index]
            image = images[frame_id].relative_to(ROOT).as_posix()
            if label is None or image in known:
                continue
            labels.append({"image": image, "near_x": f"{label[0]:.4f}", "far_x": f"{label[1]:.4f}",
                           "status": "candidate", "source": f"interpolated ({kind})", "note": ""})
            known.add(image)
            added += 1
            if len(previews) < limit:
                previews.append(overlay(read_image(images[frame_id]), detection, reference,
                                        replayed[index], label, kind))
        if added:
            write_labels(labels)
            contact_sheet(previews, out / "candidates.jpg")
            print(f"  {added} candidate labels added to {LABELS.relative_to(ROOT)}; "
                  f"magenta = proposed lane in {out / 'candidates.jpg'}")
        else:
            print("  no new candidate labels")
    return replay_score


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="run folders, or a folder of runs")
    parser.add_argument("--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
                        help="tuning overrides to try against the recording")
    parser.add_argument("--video", action="store_true", help="write an annotated replay.mp4")
    parser.add_argument("--mine", action="store_true", help="propose regression labels from missed frames")
    args = parser.parse_args(argv)
    runs = []
    for path in args.runs:
        path = path.resolve()
        runs += [path] if (path / "readings.csv").exists() else sorted(
            run for run in path.glob("run_*") if (run / "readings.csv").exists())
    if not runs:
        parser.error("no run folders with readings.csv")
    failures = 0
    for run in runs:
        if not (run / "reference_raw.png").exists() and not (run / "reference.json").exists():
            print(f"\n{run.name}: no reference; skipped")
            continue
        try:
            report(run, args.set, args.video, args.mine)
        except (RuntimeError, ValueError) as exc:
            failures += 1
            print(f"\n{run.name}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
