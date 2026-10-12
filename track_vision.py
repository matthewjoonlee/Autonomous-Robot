"""Quiz 3 vision diagnostics for saved images or live front-camera frames.

No motor commands. Image errors are normalized coordinates, not distances or
steering commands. Rotation and lateral displacement both affect these errors.
"""

import argparse
import csv
from datetime import datetime
import itertools
import json
from pathlib import Path
import signal
import sys
import time

import cv2
import numpy as np

from calibrate_track import choose_front, read_heading
from test_imu import positive_number, signed_delta


DEFAULT_SESSION = Path(__file__).resolve().parent / "track_calibration/session_20261005_172428_111633"
ROI_TOP, ROI_BOTTOM = 0.56, 0.98
NEAR_Y, FAR_Y = 0.85, 0.65


def dark_track_support(hsv, x, y, w, h):
    """Fraction of a small surrounding ring that looks like dark track."""
    height, width = hsv.shape[:2]
    pad = max(3, round(width * 0.008))
    left, top = max(0, x - pad), max(0, y - pad)
    patch = hsv[top:min(height, y + h + pad), left:min(width, x + w + pad)]
    ring = np.ones(patch.shape[:2], dtype=bool)
    ring[y - top:y - top + h, x - left:x - left + w] = False
    surrounding = patch[ring]
    if not len(surrounding):
        return 0.0
    return float(np.mean((surrounding[:, 2] < 125) & (surrounding[:, 1] < 120)))


def path_track_support(hsv, slope, intercept, low_y, high_y):
    """Require dark pavement on both sides of the fitted path along its span."""
    height, width = hsv.shape[:2]
    ys = np.linspace(low_y, high_y, 32)
    samples = []
    for offset in (-0.025, 0.025):
        xs = slope * ys + intercept + offset
        if np.any((xs < 0) | (xs >= 1)):
            return 0.0
        pixels = hsv[(ys * height).astype(int), (xs * width).astype(int)]
        samples.extend(((pixels[:, 2] < 125) & (pixels[:, 1] < 120)).tolist())
    return float(np.mean(samples))


def detect_line(frame):
    """Find a straight dashed-line candidate; refuse weak or competing fits."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # The original hue range included textured green surfaces (around H=42).
    # Retain the lighter yellow dashes (around H=30..33) without that green tail.
    mask = cv2.inRange(hsv, np.array([18, 85, 130]), np.array([38, 255, 255]))
    # Rejoin tiny vertical splits in a dash (tile seams / thresholding under
    # blur), without bridging the ordinary gaps between separate dashes.
    kernel = np.ones((max(3, round(height / 160)), 1), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask[:int(height * ROI_TOP)] = 0
    mask[int(height * ROI_BOTTOM):] = 0
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    points = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        # Reject specks, horizontal cross markings, and implausibly large blobs.
        if (area >= max(10, width * height * 0.00003)
                and w < width * 0.12 and h < height * 0.12
                and w / max(h, 1) <= 5.0
                and area / (w * h) >= 0.45
                and dark_track_support(hsv, x, y, w, h) >= 0.65):
            points.append((centers[index][0] / width, centers[index][1] / height))
    points = np.array(points, dtype=float).reshape(-1, 2)
    result = {"status": "line_lost", "confidence": 0.0, "points": points,
              "fit": None, "inliers": np.zeros(len(points), dtype=bool)}
    if len(points) < 4:
        return result

    fits = []
    for first, second in itertools.combinations(points, 2):
        if abs(first[1] - second[1]) < 0.10:
            continue
        slope = (second[0] - first[0]) / (second[1] - first[1])
        intercept = first[0] - slope * first[1]
        inliers = np.abs(points[:, 0] - (slope * points[:, 1] + intercept)) < 0.018
        selected = points[inliers]
        if len(selected) < 4:
            continue
        span = float(np.ptp(selected[:, 1]))
        # Perspective and tile seams can shorten the accepted dash span while
        # moving. Allow a compact fit only with stronger dash-count support and
        # actual coverage of the row where the controller measures position.
        dense_near = (len(selected) >= 6 and span >= 0.12
                      and selected[:, 1].min() <= NEAR_Y <= selected[:, 1].max())
        if (span < 0.18 and not dense_near) or selected[:, 1].max() < 0.82:
            continue
        if path_track_support(hsv, slope, intercept,
                              selected[:, 1].min(), selected[:, 1].max()) < 0.80:
            continue
        fits.append((len(selected), span, inliers))
    if not fits:
        return result
    fits.sort(key=lambda fit: (fit[0], fit[1]), reverse=True)
    _, span, inliers = fits[0]
    slope, intercept = np.polyfit(points[inliers, 1], points[inliers, 0], 1)
    # A distinct second path with comparable support is ambiguous.
    for support, _, other in fits[1:]:
        if support >= max(4, int(inliers.sum()) - 1) and np.count_nonzero(other & inliers) <= 1:
            result["status"] = "ambiguous"
            return result
    residual = float(np.sqrt(np.mean(
        (points[inliers, 0] - (slope * points[inliers, 1] + intercept)) ** 2)))
    selected = points[inliers]
    dense_near = (len(selected) >= 6 and span >= 0.12
                  and selected[:, 1].min() <= NEAR_Y <= selected[:, 1].max())
    # Give the compact, well-supported near-field case adequate quality weight;
    # otherwise the controller would reject it despite accepting the fit here.
    coverage = min(1.0, span / (0.15 if dense_near else 0.30))
    confidence = min(1.0, inliers.sum() / 7) * coverage * max(0, 1 - residual / 0.018)
    if confidence < 0.45:
        return result
    result.update(status="tracked", confidence=float(confidence),
                  fit=(float(slope), float(intercept)), inliers=inliers)
    return result


def detect_lane(frame, prior=None, gate=0.10, tolerance=0.02):
    """Fit the centerline of the dark lane from per-row lane candidates.

    Blur barely moves the middle of a wide dark strip, so rows that fail are
    simply skipped instead of rejecting the frame. Yellow counts as lane so the
    dashes (smeared wide by rotation blur) do not split it. Each row offers
    candidate centers: the midpoint of every dark run, or, for a run cut off by
    the image border (rotated or displaced robot), the dash pixels inside it.
    Runs spanning most of the width (crossing bands) are ignored. With a prior
    fit, only candidates near the predicted center count.

    The lane is the straight line supported by the most rows: lines through
    pairs of candidates (one in the near half, one in the far half) are scored
    by how many candidates lie within `tolerance` of them. A wrong row near the
    robot, another robot down the track, a shoe, or a carpet patch supports few
    rows and is outvoted, instead of derailing a row-by-row chain.
    """
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, np.array([18, 85, 130]), np.array([38, 255, 255])) > 0
    # Dark, unsaturated pixels are lane, except clearly green ones: in dim rooms the
    # green mats fall under the brightness/saturation limits too. Bluish lighting
    # pushed a third of the (blue-black) lane above saturation 120, so very dark
    # blue-hued pixels also count.
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # Mats: hue 42-57, brightness 112-163 across rooms. The lane is far darker (55-76)
    # and can look greenish in some lighting, so only brighter green pixels are mat.
    green = (hue >= 30) & (hue < 75) & (saturation > 40) & (value >= 90)
    lane = ((value < 125) & (saturation < 135) & ~green) | yellow
    kernel = np.ones((1, max(3, round(width / 40))), dtype=np.uint8)
    lane = cv2.morphologyEx(lane.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    rows = range(int(height * ROI_BOTTOM) - 1, int(height * ROI_TOP), -4)
    candidates = []  # (x, y, run width) in pixels
    for y in rows:
        transitions = np.diff(np.concatenate(([0], lane[y].astype(np.int8), [0])))
        starts, ends = np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
        for a, b in zip(starts, ends):
            if not 0.05 * width < b - a < 0.7 * width:
                continue
            if 0 < a and b < width:
                x = (a + b) / 2
            else:
                dash = np.flatnonzero(yellow[y, a:b])
                if not 0 < len(dash) < 0.08 * width:
                    continue
                x = a + dash.mean()
            if prior is not None and abs(x - (prior[0] * y / height + prior[1]) * width) > gate * width:
                continue
            candidates.append((x, y, b - a))
    points = []
    if candidates:
        cand = np.array(candidates, dtype=float)
        middle = height * (ROI_TOP + ROI_BOTTOM) / 2
        near, far = cand[cand[:, 1] > middle], cand[cand[:, 1] <= middle]
        if len(near) and len(far):
            # Every near/far pair defines a line x = m*y + c; keep plausible leans.
            dy = far[None, :, 1] - near[:, None, 1]
            m = (far[None, :, 0] - near[:, None, 0]) / dy
            c = near[:, None, 0] - m * near[:, None, 1]
            plausible = np.abs(m) <= 1.5
            close = np.abs(m[..., None] * cand[:, 1] + c[..., None] - cand[:, 0]) < tolerance * width
            # Most supporting candidates wins; total run width breaks ties (the lane is wide).
            score = close.sum(-1) + 1e-4 * (close * cand[:, 2]).sum(-1)
            score[~plausible] = -1
            i, j = np.unravel_index(np.argmax(score), score.shape)
            if score[i, j] > 0:
                predicted = m[i, j] * cand[:, 1] + c[i, j]
                for y in sorted(set(cand[:, 1]), reverse=True):
                    in_row = cand[cand[:, 1] == y]
                    k = np.argmin(np.abs(in_row[:, 0] - (m[i, j] * y + c[i, j])))
                    if abs(in_row[k, 0] - (m[i, j] * y + c[i, j])) < tolerance * width:
                        points.append((in_row[k, 0] / width, y / height))
    points = np.array(points, dtype=float).reshape(-1, 2)
    result = {"status": "line_lost", "confidence": 0.0, "points": points,
              "fit": None, "inliers": np.zeros(len(points), dtype=bool)}
    if len(points) < 6:
        return result
    inliers = np.ones(len(points), dtype=bool)
    for _ in range(2):  # Refit twice so a cluster of outliers cannot hold the first fit.
        slope, intercept = np.polyfit(points[inliers, 1], points[inliers, 0], 1)
        residuals = np.abs(points[:, 0] - (slope * points[:, 1] + intercept))
        inliers = residuals < max(0.012, 3 * np.median(residuals[inliers]))
        if inliers.sum() < 6:
            return result
    selected = points[inliers]
    slope, intercept = np.polyfit(selected[:, 1], selected[:, 0], 1)
    residual = float(np.sqrt(np.mean((selected[:, 0] - (slope * selected[:, 1] + intercept)) ** 2)))
    span = float(np.ptp(selected[:, 1]))
    coverage = min(1.0, inliers.sum() / (0.6 * len(rows))) * min(1.0, span / 0.25)
    confidence = coverage * max(0.0, 1 - residual / 0.02)
    result["inliers"] = inliers
    if confidence < 0.3:
        return result
    result.update(status="tracked", confidence=float(confidence),
                  fit=(float(slope), float(intercept)))
    return result


DETECTORS = {"lane": detect_lane, "dashes": detect_line}


def read_image(path):
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return frame


def summarize(detection, reference, heading=None):
    row = {"status": detection["status"], "confidence": round(detection["confidence"], 3),
           "near_error_px": None, "near_error_normalized": None,
           "image_direction_error": None, "imu_delta_deg": heading,
           "diagnostic": "LINE LOST - no position estimate"}
    if detection["status"] == "ambiguous":
        row["diagnostic"] = "AMBIGUOUS - no position estimate"
    if detection["fit"] is None:
        return row
    slope, intercept = detection["fit"]
    ref_slope, ref_intercept = reference["fit"]
    near = (slope * NEAR_Y + intercept) - (ref_slope * NEAR_Y + ref_intercept)
    far = (slope * FAR_Y + intercept) - (ref_slope * FAR_Y + ref_intercept)
    row.update(near_error_normalized=round(near, 5),
               near_error_px=round(near * reference["width"], 1),
               image_direction_error=round(near - far, 5))
    if heading is not None and abs(heading) > 3:
        row["diagnostic"] = "ROTATED LEFT" if heading < 0 else "ROTATED RIGHT"
    elif abs(near) < 0.025:
        row["diagnostic"] = "NEAR REFERENCE"
    else:
        row["diagnostic"] = "LINE RIGHT OF REFERENCE" if near > 0 else "LINE LEFT OF REFERENCE"
    return row


def annotate(frame, detection, reference, summary):
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    cv2.rectangle(canvas, (0, int(height * ROI_TOP)),
                  (width - 1, int(height * ROI_BOTTOM)), (255, 0, 255), 1)
    for point, accepted in zip(detection["points"], detection["inliers"]):
        cv2.circle(canvas, (int(point[0] * width), int(point[1] * height)),
                   4, (0, 255, 0) if accepted else (0, 0, 255), 1)
    for fit, color in [(reference["fit"], (255, 255, 0)), (detection["fit"], (0, 255, 0))]:
        if fit is not None:
            ends = [(int((fit[0] * y + fit[1]) * width), int(y * height))
                    for y in (ROI_TOP, ROI_BOTTOM)]
            cv2.line(canvas, ends[0], ends[1], color, 2)
    text = [summary["diagnostic"], f"Confidence {summary['confidence']:.2f}",
            f"Image error px: {summary['near_error_px']}  IMU deg: {summary['imu_delta_deg']}",
            "Cyan: centered reference | Green: detected line | NO MOTOR COMMANDS"]
    cv2.rectangle(canvas, (0, 0), (width, 95), (0, 0, 0), -1)
    for index, line in enumerate(text):
        cv2.putText(canvas, line, (8, 20 + index * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.43, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def save_frame(path, frame):
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Cannot save frame: {path}")


def offline(session, reference, output, detect):
    with (session / "captures.csv").open(newline="") as source:
        captures = list(csv.DictReader(source))
    if not captures:
        raise RuntimeError("Session has no capture rows")
    rows = []
    for capture in captures:
        name = capture["image"]
        frame = read_image(session / name)
        raw_heading = capture.get("delta_after_deg", "")
        heading = float(raw_heading) if raw_heading else None
        detection = detect(frame)
        summary = summarize(detection, reference, heading)
        row = {"image": name, "pose": capture["pose"], **summary}
        rows.append(row)
        save_frame(output / name, annotate(frame, detection, reference, summary))
        print(json.dumps(row))
    with (output / "results.csv").open("x", newline="") as log:
        writer = csv.DictWriter(log, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def live(args, reference, output):
    from picamera2 import Picamera2
    from imu_controller import IMUDevice
    detect = DETECTORS[getattr(args, "detector", "lane")]
    camera = imu = None
    started = False
    try:
        cameras = Picamera2.global_camera_info()
        for index, info in enumerate(cameras):
            print(f"Camera {index}: {info}")
        camera = Picamera2(args.camera if args.camera is not None else choose_front(cameras))
        camera.configure(camera.create_video_configuration(
            main={"size": (640, 480), "format": "XRGB8888"}, queue=False))
        camera.start()
        started = True
        imu = IMUDevice()
        time.sleep(1.5)
        metadata = camera.capture_metadata()
        if "ColourGains" in metadata:
            camera.set_controls({"AwbEnable": False, "ColourGains": metadata["ColourGains"]})
        input("Align robot centered and straight, then press Enter to zero heading and begin: ")
        raw_output = output / "raw"
        raw_output.mkdir()
        reference_frame = cv2.cvtColor(camera.capture_array(), cv2.COLOR_BGRA2BGR)
        save_frame(output / "reference_raw.png", reference_frame)
        starting_detection = detect(reference_frame)
        if args.reference is None:
            # A fresh explicitly aligned reference avoids differences between
            # the old still photos and today's camera pose/exposure.
            reference = starting_detection
            if reference["status"] != "tracked":
                raise RuntimeError("Starting line not reliably visible. Use a continuous straight "
                                   "track with several nearby dashes visible and restart.")
            reference["width"] = reference_frame.shape[1]
        save_frame(output / "reference_annotated.jpg", annotate(
            reference_frame, starting_detection, reference,
            summarize(starting_detection, reference)))
        zero = read_heading(imu)
        if zero is None:
            raise RuntimeError("Heading unavailable; cannot set reference")
        print("Move/rotate by hand. No motor commands. Ctrl+C stops.")
        print("Saving unannotated PNGs in raw/ and diagnostic JPGs alongside results.csv.")
        start = time.monotonic()
        count = 0
        with (output / "results.csv").open("x", newline="") as log:
            writer = None
            while args.duration is None or time.monotonic() - start < args.duration:
                before = time.monotonic()
                frame = cv2.cvtColor(camera.capture_array(), cv2.COLOR_BGRA2BGR)
                heading = read_heading(imu)
                delta = signed_delta(heading, zero, imu.SIGN) if heading is not None else None
                detection = detect(frame)
                summary = summarize(detection, reference, delta)
                name = f"frame_{count:05d}.jpg"
                raw_name = f"raw/frame_{count:05d}.png"
                # Save before drawing overlays, preserving exact detector input.
                save_frame(output / raw_name, frame)
                row = {"image": name, "raw_image": raw_name,
                       "elapsed_s": round(time.monotonic() - start, 3),
                       "imu_status": "ok" if heading is not None else "unavailable", **summary}
                if writer is None:
                    writer = csv.DictWriter(log, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                log.flush()
                save_frame(output / name, annotate(frame, detection, reference, summary))
                print(json.dumps(row), flush=True)
                count += 1
                time.sleep(max(0, args.interval - (time.monotonic() - before)))
    finally:
        actions = ([imu.close] if imu is not None else [])
        if camera is not None:
            actions += ([camera.stop] if started else []) + [camera.close]
        for action in actions:
            try:
                action()
            except Exception as exc:
                print(f"Cleanup warning: {exc}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--reference", type=Path, help="centered, straight reference image")
    parser.add_argument("--live", action="store_true", help="use front camera and IMU on Pi")
    parser.add_argument("--camera", type=int)
    parser.add_argument("--interval", type=positive_number, default=1.0,
                        help="live seconds between saved diagnostic frames")
    parser.add_argument("--duration", type=positive_number)
    parser.add_argument("--detector", choices=sorted(DETECTORS), default="lane",
                        help="lane centerline (controller default) or individual dashes")
    parser.add_argument("--output", type=Path, default=Path("vision_results"))
    args = parser.parse_args()
    if args.camera is not None and args.camera < 0:
        parser.error("--camera must be nonnegative")
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        reference = None
        if not args.live or args.reference is not None:
            path = args.reference or args.session / "001_centered_straight.jpg"
            frame = read_image(path)
            reference = DETECTORS[args.detector](frame)
            if reference["status"] != "tracked":
                raise RuntimeError("Cannot reliably detect the centered reference line")
            reference["width"] = frame.shape[1]
        output = args.output / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        output.mkdir(parents=True, exist_ok=False)
        print(f"Results: {output}")
        if args.live:
            live(args, reference, output)
        else:
            offline(args.session, reference, output, DETECTORS[args.detector])
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Completed results are saved.")
    except Exception as exc:
        print(f"Vision test failed: {exc}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
    return 0


if __name__ == "__main__":
    sys.exit(main())
