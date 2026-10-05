"""Stationary Quiz 3 heading test. No motor controller is imported or used."""

import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import select
import signal
import sys
import time


def positive_number(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return number


def signed_delta(heading, reference, sign):
    value = ((heading - reference) * sign) % 360.0
    return value - 360.0 if value > 180.0 else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=positive_number, default=0.2,
                        help="seconds between readings (default: 0.2)")
    parser.add_argument("--duration", type=positive_number,
                        help="optional maximum test duration in seconds")
    parser.add_argument("--log", type=Path,
                        help="new CSV path (default: timestamped file in imu_logs)")
    args = parser.parse_args()
    log_path = args.log or Path("imu_logs") / datetime.now().strftime("imu_%Y%m%d_%H%M%S_%f.csv")

    # Import here so --help works even on a computer without Pi hardware libraries.
    try:
        from imu_controller import IMUDevice
    except ImportError as exc:
        print(f"IMU libraries unavailable: {exc}. Run this test in the Pi's Python environment.",
              file=sys.stderr)
        return 1

    imu = None
    samples = unavailable = 0
    reference = None
    label = "unlabeled"
    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    previous_handler = signal.signal(signal.SIGTERM, request_stop)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Never overwrite a previous calibration log.
        with log_path.open("x", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerow(["timestamp", "elapsed_s", "label", "heading_deg",
                             "delta_deg", "reference_deg", "status", "event"])
            imu = IMUDevice()
            print("No motor commands. Keep the robot stationary for initial readings.")
            print("Commands (press Enter): z = zero, c = centered, l = rotated left,")
            print("r = rotated right, s = stationary drift check, q = quit. Ctrl+C also exits.")
            print("Once readings settle, align straight and enter z. Hold still for 10 seconds,")
            print("then rotate left, return centered, rotate right, and return centered.")
            print(f"CSV log: {log_path}")
            start = time.monotonic()
            next_read = start
            stdin_open = True

            while not stop_requested:
                now = time.monotonic()
                if args.duration is not None and now - start >= args.duration:
                    break
                event = ""
                if stdin_open and select.select([sys.stdin], [], [], 0)[0]:
                    line = sys.stdin.readline()
                    if not line:
                        stdin_open = False
                    else:
                        command = line.strip().lower()
                        if command == "q":
                            break
                        if command in {"c", "l", "r", "s"}:
                            label = {"c": "centered", "l": "left", "r": "right",
                                     "s": "stationary"}[command]
                            event = "label_changed"
                        elif command == "z":
                            event = "zero_requested"
                        elif command:
                            print("Use z, c, l, r, s, or q, followed by Enter.")

                # Use one sensor read for both absolute and relative heading. The
                # local reference avoids reporting success if IMUDevice.zero()
                # silently fails during a missing reading.
                heading = imu.heading()
                if heading is not None and not math.isfinite(heading):
                    heading = None
                samples += 1
                if heading is None:
                    unavailable += 1
                if event == "zero_requested":
                    if heading is None:
                        event = "zero_failed"
                        print("Zero failed: heading unavailable; previous reference retained.")
                    else:
                        reference = heading
                        label = "centered"
                        event = "zero_set"
                        print(f"Reference set to {reference:.2f} degrees.")
                delta = (signed_delta(heading, reference, imu.SIGN)
                         if heading is not None and reference is not None else None)
                status = ("unavailable" if heading is None else
                          "not_zeroed" if reference is None else "ok")
                elapsed = time.monotonic() - start
                writer.writerow([datetime.now().astimezone().isoformat(), f"{elapsed:.3f}",
                                 label, "" if heading is None else f"{heading:.3f}",
                                 "" if delta is None else f"{delta:.3f}",
                                 "" if reference is None else f"{reference:.3f}", status, event])
                log_file.flush()
                heading_text = "unavailable" if heading is None else f"{heading:7.2f} deg"
                delta_text = "--" if delta is None else f"{delta:+7.2f} deg"
                print(f"{elapsed:6.1f}s  {label:10s} heading={heading_text}  delta={delta_text}  {status}",
                      flush=True)
                next_read += args.interval
                delay = next_read - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_read = time.monotonic()
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C.")
    except Exception as exc:
        print(f"IMU test failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if imu is not None:
            imu.close()
        signal.signal(signal.SIGTERM, previous_handler)
        print(f"Readings: {samples}; unavailable: {unavailable}. Log: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
