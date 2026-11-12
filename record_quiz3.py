"""Record both cameras continuously while the robot moves through an intersection.

Drive with the PS4/PS5 controller (recommended):
    python record_quiz3.py --label four_way --drive

  Left stick     drive normally (to get to the start; capped at --manual-max)
  hold R1 / R2   creep straight forward at the fixed --speed (left stick steers a little)
  Cross (X)      START recording; after that, mark the next tape distance from --marks
  Triangle       skip the next distance in --marks (if you missed that tape mark)
  Square         unlabeled mark
  Circle         stop recording (motors stop too)

Each time the robot's CENTER mark passes a tape-measure mark, press Cross. The
distances are taken in order from --marks (default 36 30 24 18 12 6 0 -6 -12 inches;
+ before the cross-road center line, - past it). Frames between marks get distances
by interpolation. Do not run robot_controller.py at the same time.

Push by hand instead (no motors):
    python record_quiz3.py --label four_way
  Press Enter to start; type each distance + Enter as the center passes the mark;
  q + Enter stops. (Add --pad to use the controller buttons for marks.)

Motors move only while you hold R1/R2 or the left stick, and stop when this program
ends or a camera stops responding. Output: quiz3_recordings/<label>_<time>/front.mp4,
side.mp4 (same frame numbering), frames.csv (time, heading, motor command), marks.csv.
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

# Button numbers as in ps5_controller.py (same for PS4 controllers on the Pi).
CROSS, CIRCLE, TRIANGLE, SQUARE, R1, R2 = 0, 1, 2, 3, 5, 7
DEFAULT_MARKS = "36,30,24,18,12,6,0,-6,-12"
MOTOR_INTERVAL = 0.04  # seconds between motor commands (auto-stop is 200 ms)


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


class Pad:
    """PS4/PS5 controller read directly with pygame (quiet, unlike PS5_Controller)."""

    def __init__(self, wait_s=30):
        import pygame
        self.pygame = pygame
        pygame.init()
        pygame.joystick.init()
        deadline = time.monotonic() + wait_s
        if pygame.joystick.get_count() == 0:
            print("Waiting for the controller to connect...")
        while pygame.joystick.get_count() == 0:
            if time.monotonic() > deadline:
                raise RuntimeError("No controller connected (press its PS button to pair).")
            pygame.event.pump()
            time.sleep(0.2)
            pygame.joystick.quit()
            pygame.joystick.init()
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.name = self.joystick.get_name()
        self.down = set()

    def poll(self):
        """(buttons newly pressed, buttons held, (left x, left y))."""
        self.pygame.event.pump()
        held = {i for i in range(self.joystick.get_numbuttons()) if self.joystick.get_button(i)}
        pressed, self.down = held - self.down, held
        return pressed, held, (self.joystick.get_axis(0), self.joystick.get_axis(1))

    def close(self):
        try:
            self.pygame.joystick.quit()
            self.pygame.quit()
        except Exception:
            pass


def drive_command(held, stick, speed, max_turn, manual_max, deadzone=0.1):
    """(speed, turn) for the Sabertooth, with robot_controller.py's sign conventions.

    R1/R2 held: creep forward at a fixed speed, stick x steers within max_turn.
    Otherwise the left stick drives, scaled to manual_max with a squared response:
    half stick gives a quarter of manual_max, so small pushes do not lurch.
    Nothing held: stop.
    """
    lx, ly = (0.0 if abs(value) < deadzone else value for value in stick)
    if R1 in held or R2 in held:
        return speed, -round(lx * max_turn)
    if lx or ly:
        return -round(ly * abs(ly) * manual_max), -round(lx * abs(lx) * manual_max)
    return 0, 0


def parse_marks(text):
    try:
        return [float(part) for part in text.replace(" ", ",").split(",") if part]
    except ValueError:
        raise argparse.ArgumentTypeError("--marks must be numbers separated by commas")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="run", help="e.g. four_way or right_turn")
    parser.add_argument("--drive", action="store_true", help="drive with the controller (motors on)")
    parser.add_argument("--pad", action="store_true", help="controller buttons for marks, no motors")
    parser.add_argument("--speed", type=int, default=20, help="R1 creep speed (lane centering uses 25)")
    parser.add_argument("--max-turn", type=int, default=15, help="steering while creeping")
    parser.add_argument("--manual-max", type=int, default=30, help="left-stick drive limit")
    parser.add_argument("--marks", type=parse_marks, default=parse_marks(DEFAULT_MARKS),
                        help=f"distances for Cross presses, in order (default {DEFAULT_MARKS})")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--legacy-fov", action="store_true", help="old cropped 640x480 front view")
    parser.add_argument("--output", type=Path, default=Path("quiz3_recordings"))
    args = parser.parse_args()
    if args.fps <= 0 or args.max_seconds <= 0:
        parser.error("--fps and --max-seconds must be positive")
    if not (0 <= args.speed <= 40 and 0 <= args.max_turn <= 40 and 0 <= args.manual_max <= 127):
        parser.error("--speed and --max-turn must be 0-40, --manual-max 0-127")

    front = side = imu = pad = saber = None
    started, writers = [], []
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    folder = None
    frames = marks = 0
    command = (0, 0)
    try:
        import cv2
        import numpy as np
        from picamera2 import Picamera2
        from calibrate_track import choose_front
        from camera_setup import (FRONT_SIZE, LEGACY_FRONT_SIZE, SIDE_SIZE, front_bgr,
                                  front_config, side_bgr, side_config)
        from capture_quiz3 import grab
        from capture_side_camera import choose_side

        if args.drive or args.pad:
            pad = Pad()
            print(f"Controller: {pad.name}")
        if args.drive:
            from sabertooth import Sabertooth
            saber = Sabertooth()
            if saber.ser is None or not saber.ser.is_open:
                raise RuntimeError("Motor serial port unavailable")
            saber.stop()
            saber.set_auto_stop(200)  # motors stop by themselves if commands stop arriving
            saber.set_ramping(8)

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

        folder = args.output / f"{slug(args.label)}_{datetime.now():%Y%m%d_%H%M%S}"
        folder.mkdir(parents=True, exist_ok=False)
        front_size = LEGACY_FRONT_SIZE if args.legacy_fov else FRONT_SIZE
        front_writer = open_writer(cv2, folder / "front", args.fps, front_size)
        writers.append(front_writer)
        side_writer = open_writer(cv2, folder / "side", args.fps, SIDE_SIZE)
        writers.append(side_writer)
        blank_side = np.zeros((SIDE_SIZE[1], SIDE_SIZE[0], 3), dtype=np.uint8)

        last_motor = 0.0

        def update_motors(held=frozenset(), stick=(0.0, 0.0)):
            """Send the controller's drive command at most every MOTOR_INTERVAL."""
            nonlocal command, last_motor
            if saber is None:
                return
            now = time.monotonic()
            if now - last_motor < MOTOR_INTERVAL:
                return
            last_motor = now
            command = drive_command(held, stick, args.speed, args.max_turn, args.manual_max)
            if command == (0, 0):
                saber.stop()
            else:
                saber.drive(*command)

        print(f"Saving to {folder}." + ("" if saber else " No motor commands are sent."))
        reader = LineReader()  # one reader for everything typed, so nothing is lost
        if pad:
            print("Get to the start position" + (" with the left stick" if saber else "")
                  + ", then press Cross (or Enter) to START recording.")
        else:
            print("Put the robot at its start position, then press Enter to START recording> ",
                  end="", flush=True)
        while True:
            typed = reader.poll()
            if typed:
                reader.pending = typed[1:]  # anything typed after Enter counts as marks
                break
            if reader.closed and pad is None:
                raise EOFError
            if pad:
                pressed, held, stick = pad.poll()
                if CROSS in pressed:
                    break
                if CIRCLE in pressed:
                    raise KeyboardInterrupt
                update_motors(held, stick)
            time.sleep(0.02)
        upcoming = list(args.marks)
        if pad:
            print("RECORDING. " + ("Hold R1 to creep forward. " if saber else "")
                  + f"Press Cross at each mark: {' '.join(f'{m:g}' for m in upcoming)}. "
                  "Triangle skips one, Circle stops.")
        else:
            print("RECORDING. Type the robot-center distance (inches) + Enter at each tape mark; "
                  "q + Enter stops.")
        with (folder / "frames.csv").open("x", newline="") as frame_log, \
                (folder / "marks.csv").open("x", newline="") as mark_log:
            frame_csv = csv.writer(frame_log)
            frame_csv.writerow(["frame", "time_s", "side_ok", "heading_rel_deg", "speed", "turn"])
            mark_csv = csv.writer(mark_log)
            mark_csv.writerow(["time_s", "frame", "distance_in", "text"])
            reference = read_heading(imu)  # heading 0 = direction at the start of recording
            start = time.monotonic()
            next_frame = start
            last_report = start
            stop = False

            def add_mark(elapsed, distance, text):
                nonlocal marks
                mark_csv.writerow([f"{elapsed:.3f}", frames, distance, text])
                mark_log.flush()
                marks += 1
                print(f"  mark {text or '(blank)'} at {elapsed:.1f} s, frame {frames}")

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
                    add_mark(elapsed, distance, text)
                if pad:
                    pressed, held, stick = pad.poll()
                    if CIRCLE in pressed:
                        stop = True
                    if CROSS in pressed:
                        if upcoming:
                            value = upcoming.pop(0)
                            add_mark(elapsed, value, f"{value:g}")
                        else:
                            add_mark(elapsed, "", "extra")
                    if TRIANGLE in pressed and upcoming:
                        print(f"  skipped {upcoming.pop(0):g}"
                              + (f"; next is {upcoming[0]:g}" if upcoming else ""))
                    if SQUARE in pressed:
                        add_mark(elapsed, "", "")
                    update_motors(set() if stop else held, stick)
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
                                    int(side_frame is not None), heading, *command])
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
        if saber is not None:  # motors first
            for action in (saber.stop, saber.close):
                try:
                    action()
                except Exception as exc:
                    print(f"Motor cleanup warning: {exc}", file=sys.stderr)
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
        if pad is not None:
            pad.close()
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
