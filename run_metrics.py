"""Scorecard for a centering run, so each change is judged by the same numbers.

`python run_metrics.py centering_results` prints one row per run.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

CENTERED_PX = 10  # |near-row image error| counted as centered.
COLUMNS = ("distance_cm", "tracked_pct", "lost_episodes", "longest_lost_s", "search_s", "lateral_rms_px", "rms_error_px",
           "peak_error_px", "final_error_px", "centered_after_s", "turn_jitter",
           "turn_reversals", "max_heading_deg")


def number(value):
    if value is None or value == "":
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def score(rows, min_confidence=0.5):
    """Metrics from telemetry rows (live records or readings.csv dicts)."""
    rows = [row for row in rows if number(row.get("elapsed_s")) is not None]
    if not rows:
        return {"frames": 0}
    times = [number(row["elapsed_s"]) for row in rows]
    trusted = [row.get("status") == "tracked" and (number(row.get("confidence")) or 0) >= min_confidence
               and number(row.get("near_error_px")) is not None for row in rows]
    errors = [(time, number(row["near_error_px"])) for time, row, ok in zip(times, rows, trusted) if ok]

    episodes, longest, lost_start = 0, 0.0, None
    for time, ok in zip(times, trusted):
        if not ok and lost_start is None:
            episodes += 1
            lost_start = time
        elif ok and lost_start is not None:
            longest = max(longest, time - lost_start)
            lost_start = None
    if lost_start is not None:
        longest = max(longest, times[-1] - lost_start)

    searching = sum(later - earlier for earlier, later, row in zip(times, times[1:], rows)
                    if row.get("state") in ("searching", "coasting"))
    distances = [number(row.get("distance_cm")) for row in rows if number(row.get("distance_cm")) is not None]
    # Lateral error has the IMU-measured rotation removed (newer runs only).
    lateral = [640 * number(row["lateral_error"]) for row, ok in zip(rows, trusted)
               if ok and number(row.get("lateral_error")) is not None]
    turn_key = "turn" if "turn" in rows[0] else "physical_turn"
    moving = [number(row.get(turn_key)) or 0.0 for row in rows if (number(row.get("speed")) or 0) > 0]
    steps = [later - earlier for earlier, later in zip(moving, moving[1:])]
    signs = [math.copysign(1, turn) for turn in moving if abs(turn) >= 1]
    headings = [abs(number(row.get("imu_delta_deg"))) for row in rows
                if number(row.get("imu_delta_deg")) is not None]

    centered_after = None
    if errors and abs(errors[-1][1]) <= CENTERED_PX:
        centered_after = errors[0][0]
        for time, error in errors:
            if abs(error) > CENTERED_PX:
                centered_after = None
            elif centered_after is None:
                centered_after = time
    final = [error for time, error in errors if time >= times[-1] - 1.0]

    def rounded(value, digits=2):
        return None if value is None else round(value, digits)

    return {
        "frames": len(rows),
        "distance_cm": rounded(distances[-1], 1) if distances else None,
        "duration_s": rounded(times[-1] - times[0]),
        "fps": rounded((len(rows) - 1) / (times[-1] - times[0]), 1) if times[-1] > times[0] else None,
        "tracked_pct": rounded(100 * sum(trusted) / len(rows), 1),
        "lost_episodes": episodes,
        "longest_lost_s": rounded(longest),
        "search_s": rounded(searching),
        "lateral_rms_px": rounded(math.sqrt(sum(x * x for x in lateral) / len(lateral)), 1) if lateral else None,
        "rms_error_px": rounded(math.sqrt(sum(e * e for _, e in errors) / len(errors)), 1) if errors else None,
        "peak_error_px": rounded(max(abs(e) for _, e in errors), 1) if errors else None,
        "final_error_px": rounded(sum(final) / len(final), 1) if final else None,
        "centered_after_s": rounded(centered_after),
        "turn_jitter": rounded(math.sqrt(sum(s * s for s in steps) / len(steps)), 3) if steps else None,
        "turn_reversals": sum(1 for a, b in zip(signs, signs[1:]) if a != b),
        "max_heading_deg": rounded(max(headings), 2) if headings else None,
    }


def describe(metrics):
    """One-line human summary printed after each run."""
    if not metrics.get("frames"):
        return "No telemetry recorded."
    distance = f"~{metrics['distance_cm'] / 30.48:.1f} ft (estimated) | " if metrics.get("distance_cm") else ""
    return (f"{distance}{metrics['frames']} frames at {metrics['fps']} fps | lane tracked {metrics['tracked_pct']}% | "
            f"lost {metrics['lost_episodes']}x (longest {metrics['longest_lost_s']}s) | "
            f"lateral RMS {metrics['lateral_rms_px']} px | image error RMS {metrics['rms_error_px']} px, "
            f"peak {metrics['peak_error_px']} px, "
            f"final {metrics['final_error_px']} px | turn jitter {metrics['turn_jitter']}, "
            f"reversals {metrics['turn_reversals']}")


def run_metrics(run):
    """Saved metrics.json, or computed from readings.csv for older runs."""
    saved = run / "metrics.json"
    if saved.exists():
        return json.loads(saved.read_text())
    outcome = json.loads((run / "outcome.json").read_text()) if (run / "outcome.json").exists() else {}
    with (run / "readings.csv").open(newline="") as log:
        rows = list(csv.DictReader(log))
    return score(rows, outcome.get("settings", {}).get("min_confidence", 0.5))


def table(runs):
    header = ["run", "trial", "reason"] + list(COLUMNS)
    lines = [header]
    for run in runs:
        outcome = json.loads((run / "outcome.json").read_text()) if (run / "outcome.json").exists() else {}
        metrics = run_metrics(run)
        lines.append([run.name, str(outcome.get("trial", "")), str(outcome.get("reason", ""))[:24]]
                     + ["" if metrics.get(key) is None else str(metrics.get(key)) for key in COLUMNS])
    widths = [max(len(line[index]) for line in lines) for index in range(len(header))]
    return "\n".join("  ".join(cell.ljust(width) for cell, width in zip(line, widths)) for line in lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="run folders or folders containing runs")
    args = parser.parse_args(argv)
    runs = []
    for path in args.paths:
        runs += [path] if (path / "readings.csv").exists() else sorted(
            run for run in path.glob("run_*") if (run / "readings.csv").exists())
    if not runs:
        print("No runs with readings.csv found.", file=sys.stderr)
        return 1
    print(table(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
