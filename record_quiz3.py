"""Record both cameras continuously while the robot moves through an intersection.

    python record_quiz3.py --label four_way

No motor commands. Push the robot by hand, or drive it with the PS5 controller by
running robot_controller.py in a second terminal (this program only uses the cameras
and IMU, so the two can run together).

While recording, each time the robot's CENTER mark passes a tape-measure mark, type
that distance in inches and press Enter (e.g. 36, 24, 12, 0, -12; + before the
cross-road center line, - past it). Frames between marks get distances by
interpolation, so marks every 6-12 inches are enough. Enter alone drops an unlabeled
mark. Type q and Enter (or press Ctrl+C) to stop.

Output: quiz3_recordings/<label>_<time>/front.mp4, side.mp4 (full view, same frame
numbering), frames.csv (time, IMU heading, side camera ok) and marks.csv.
"""

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import select
import signal
import sys
import time

from capture_quiz3 import HUNG, open_side, read_heading, slug, with_timeout


def open_writer(cv2, path, fps, size):
    """MP4 video writer, falling back to MJPG AVI if this OpenCV lacks mp4v."""
    writer = cv2.VideoWriter(str(path.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if writer.isOpened():
        return writer
    writer = cv2.VideoWriter(str(path.with_suffix(".avi")), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open a video writer for {path}")
    return writer


class LineReader:
    """Non-blocking typed lines from stdin.

    Reads the file descriptor directly: mixing select() with Python's buffered
    sys.stdin let a readline() block the recording loop until the next line.
    """

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.buffer = b""
        self.closed = False
        self.pending = []

    def poll(self):
        while not self.closed and select.select([self.fd], [], [], 0)[0]:
            chunk = os.read(self.fd, 1024)
            if not chunk:
                self.closed = True
            self.buffer += chunk
        lines, self.pending = self.pending, []
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            lines.append(line.decode(errors="replace").strip())
        return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="run", help="e.g. four_way or right_turn")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--legacy-fov", action="store_true", help="old cropped 640x480 front view")
    parser.add_argument("--output", type=Path, default=Path("quiz3_recordings"))
    args = parser.parse_args()
    if args.fps <= 0 or args.max_seconds <= 0:
        parser.error("--fps and --max-seconds must be positive")

    front = side = imu = None
    started, writers = [], []
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    folder = None
    frames = marks = 0
    try:
        import cv2
        import numpy as np
        from picamera2 import Picamera2
        from calibrate_track import choose_front
        from camera_setup import (FRONT_SIZE, LEGACY_FRONT_SIZE, SIDE_SIZE, front_bgr,
                                  front_config, side_bgr, side_config)
        from capture_quiz3 import grab
        from capture_side_camera import choose_side

        infos = Picamera2.global_camera_info()
        front = Picamera2(choose_front(infos))
        front.configure(front_config(front, legacy=args.legacy_fov))
        front.start()
        started.append(front)
        try:
            side = with_timeout(lambda: open_side(Picamera2, side_config, choose_side), 10)
            started.append(side)
        except Exception as exc:
            print(f"WARNING: side camera unavailable ({exc}); recording front only.")
        try:
            from imu_controller import IMUDevice
            imu = IMUDevice()
        except Exception as exc:
            print(f"WARNING: IMU unavailable ({exc}); headings will be blank.")
        time.sleep(2)
        metadata = front.capture_metadata()
        if "ColourGains" in metadata:  # same white-balance lock as the driving programs
            front.set_controls({"AwbEnable": False, "ColourGains": metadata["ColourGains"]})
        reference = read_heading(imu)

        folder = args.output / f"{slug(args.label)}_{datetime.now():%Y%m%d_%H%M%S}"
        folder.mkdir(parents=True, exist_ok=False)
        front_size = LEGACY_FRONT_SIZE if args.legacy_fov else FRONT_SIZE
        front_writer = open_writer(cv2, folder / "front", args.fps, front_size)
        writers.append(front_writer)
        side_writer = open_writer(cv2, folder / "side", args.fps, SIDE_SIZE)
        writers.append(side_writer)
        blank_side = np.zeros((SIDE_SIZE[1], SIDE_SIZE[0], 3), dtype=np.uint8)

        print(f"Saving to {folder}. No motor commands are sent.")
        reader = LineReader()  # one reader for everything typed, so nothing is lost
        print("Put the robot at its start position, then press Enter to START recording> ",
              end="", flush=True)
        while True:
            typed = reader.poll()
            if typed:
                reader.pending = typed[1:]  # anything typed after Enter counts as marks
                break
            if reader.closed:
                raise EOFError
            time.sleep(0.05)
        print("RECORDING. Type the robot-center distance (inches) + Enter at each tape mark; "
              "q + Enter stops.")
        with (folder / "frames.csv").open("x", newline="") as frame_log, \
                (folder / "marks.csv").open("x", newline="") as mark_log:
            frame_csv = csv.writer(frame_log)
            frame_csv.writerow(["frame", "time_s", "side_ok", "heading_rel_deg"])
            mark_csv = csv.writer(mark_log)
            mark_csv.writerow(["time_s", "frame", "distance_in", "text"])
            start = time.monotonic()
            next_frame = start
            last_report = start
            stop = False
            while True:
                now = time.monotonic()
                elapsed = now - start
                if elapsed >= args.max_seconds:
                    print(f"Reached --max-seconds {args.max_seconds:g}; stopping.")
                    break
                for text in reader.poll():
                    if text.lower() == "q":
                        stop = True
                        break
                    try:
                        distance = float(text)
                    except ValueError:
                        distance = ""
                    mark_csv.writerow([f"{elapsed:.3f}", frames, distance, text])
                    mark_log.flush()
                    marks += 1
                    print(f"  mark {text or '(blank)'} at {elapsed:.1f} s, frame {frames}")
                if stop:
                    break
                if now < next_frame:
                    time.sleep(min(next_frame - now, 0.01))
                    continue
                next_frame = max(next_frame + 1 / args.fps, now)
                try:
                    frame = with_timeout(lambda: grab(front, front_bgr)[0], 3)
                except TimeoutError:
                    HUNG.append(front)
                    print("The FRONT camera stopped responding; stopping (recording so far is saved).")
                    break
                side_frame = None
                if side is not None:
                    try:
                        side_frame = with_timeout(lambda: grab(side, side_bgr)[0], 3)
                    except Exception as exc:
                        if isinstance(exc, TimeoutError):
                            HUNG.append(side)
                        print(f"The SIDE camera stopped responding ({exc}); continuing front only. "
                              "Check its USB cable.")
                        side = None
                heading = None
                if imu is not None:
                    try:
                        value = with_timeout(lambda: read_heading(imu, tries=1), 0.5)
                        if value is not None and reference is not None:
                            from test_imu import signed_delta
                            heading = round(signed_delta(value, reference, imu.SIGN), 3)
                    except TimeoutError:
                        HUNG.append(imu)
                        imu = None
                front_writer.write(frame)
                side_writer.write(side_frame if side_frame is not None else blank_side)
                frame_csv.writerow([frames, f"{time.monotonic() - start:.3f}",
                                    int(side_frame is not None), heading])
                frames += 1
                if now - last_report >= 5:
                    frame_log.flush()
                    last_report = now
                    print(f"  {elapsed:.0f} s, {frames} frames")
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
    except Exception as exc:
        print(f"Recording failed: {exc}", file=sys.stderr)
        return 1
    finally:
        for writer in writers:
            writer.release()
        for camera in started:
            if not any(camera is hung for hung in HUNG):
                try:
                    with_timeout(camera.stop, 5)
                    with_timeout(camera.close, 5)
                except Exception:
                    pass
        if imu is not None and not any(imu is hung for hung in HUNG):
            try:
                with_timeout(imu.close, 5)
            except Exception:
                pass
        signal.signal(signal.SIGTERM, previous)
    if folder is not None:
        print(f"Saved {frames} frames and {marks} marks in {folder}")
    return 0


if __name__ == "__main__":
    code = main()
    if HUNG:  # a stuck camera thread would keep Python from exiting
        sys.stdout.flush()
        os._exit(code)
    sys.exit(code)
