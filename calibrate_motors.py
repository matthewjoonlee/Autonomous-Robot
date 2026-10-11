"""One bounded Quiz 3 motor/IMU trial. Simulation is the default."""

import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import signal
import sys
import time

from test_imu import positive_number, signed_delta


INTERVAL = 0.04
MAX_READ_SECONDS = 0.10
FIELDS = ["timestamp", "elapsed_s", "trial", "phase", "speed_command",
          "turn_command", "heading_deg", "delta_deg", "status"]


def bounded_int(low, high):
    def parse(value):
        number = int(value)
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f"must be between {low} and {high}")
        return number
    return parse


def fresh_heading(imu, clock=time.monotonic):
    start = clock()
    heading = imu.heading()
    if clock() - start > MAX_READ_SECONDS:
        raise RuntimeError("IMU read exceeded 100 ms; trial aborted")
    if heading is None or not math.isfinite(heading):
        raise RuntimeError("IMU heading unavailable; trial aborted")
    return heading


def run_trial(saber, imu, args, writer, clock=time.monotonic, sleep=time.sleep):
    """Stop even when sensor reads, motor writes, logging, or timing fail."""
    start = clock()
    last_delta = 0.0
    turn = {"stop": 0, "forward": 0, "left": -args.turn, "right": args.turn}[args.trial]
    speed = 0 if args.trial == "stop" else args.speed

    def record(phase, heading, status, drive_speed=0, drive_turn=0):
        nonlocal last_delta
        last_delta = signed_delta(heading, reference, imu.SIGN) if heading is not None else None
        writer.writerow(dict(zip(FIELDS, [
            datetime.now().astimezone().isoformat(), round(clock() - start, 4), args.trial,
            phase, drive_speed, drive_turn, heading, last_delta, status,
        ])))

    reference = None
    try:
        saber.stop()
        reference = fresh_heading(imu, clock)
        record("baseline", reference, "ok")
        movement_start = clock()
        deadline = movement_start + args.duration
        previous_command = movement_start
        while clock() < deadline:
            heading = fresh_heading(imu, clock)
            now = clock()
            if now >= deadline:
                break
            if now - previous_command > MAX_READ_SECONDS:
                raise RuntimeError("Control loop delayed over 100 ms; trial aborted")
            delta = signed_delta(heading, reference, imu.SIGN)
            if abs(delta) >= args.max_heading:
                raise RuntimeError(f"Heading limit reached ({delta:+.2f} degrees); trial aborted")
            saber.drive(speed, turn)
            previous_command = clock()
            record("command", heading, "ok", speed, turn)
            sleep(max(0, min(INTERVAL, deadline - clock())))
        saber.stop()
        record("stopped", fresh_heading(imu, clock), "duration_complete")
        # Observe heading after the command stops; this does not establish linear
        # stopping distance or prove the physical wheels have stopped.
        settle_deadline = clock() + 0.5
        while clock() < settle_deadline:
            saber.stop()
            record("settling", fresh_heading(imu, clock), "ok")
            sleep(INTERVAL)
        print(f"Trial complete. Final relative heading: {last_delta:+.2f} degrees.")
        return last_delta
    except BaseException as exc:
        # Issue stop before attempting disk I/O, including interruption paths.
        saber.stop()
        writer.writerow(dict(zip(FIELDS, [
            datetime.now().astimezone().isoformat(), round(clock() - start, 4),
            args.trial, "aborted", 0, 0, None, last_delta, str(exc) or type(exc).__name__,
        ])))
        raise
    finally:
        saber.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", choices=["stop", "forward", "left", "right"], default="stop")
    parser.add_argument("--speed", type=bounded_int(1, 35), default=20,
                        help="forward command, not a measured physical speed (default: 20)")
    parser.add_argument("--turn", type=bounded_int(1, 15), default=6)
    parser.add_argument("--duration", type=positive_number, default=0.5,
                        help="command duration in seconds, maximum 2 (default: 0.5)")
    parser.add_argument("--max-heading", type=positive_number, default=15,
                        help="abort at this relative angle, maximum 30 degrees")
    parser.add_argument("--execute", action="store_true", help="enable physical hardware trial")
    parser.add_argument("--output", type=Path, default=Path("motor_calibration"))
    args = parser.parse_args()
    if args.duration > 2 or args.max_heading > 30:
        parser.error("duration must be <= 2 seconds and max-heading <= 30 degrees")
    turn = {"stop": 0, "forward": 0, "left": -args.turn, "right": args.turn}[args.trial]
    speed = 0 if args.trial == "stop" else args.speed
    print(f"Trial {args.trial}: speed={speed}, turn={turn}, duration={args.duration:.2f}s.")
    if not args.execute:
        print("Simulation preview only: no hardware imported and no motor commands sent.")
        print("Use --execute on the Pi for a physical trial; it will require typing RUN.")
        return 0

    saber = imu = None
    handlers = {}

    def terminate(_signal, _frame):
        raise KeyboardInterrupt

    for signum in (signal.SIGTERM, signal.SIGHUP):
        handlers[signum] = signal.signal(signum, terminate)
    try:
        from imu_controller import IMUDevice
        from sabertooth import Sabertooth

        imu = IMUDevice()
        fresh_heading(imu)
        saber = Sabertooth()
        if saber.ser is None or not saber.ser.is_open:
            raise RuntimeError("Motor serial port unavailable")
        saber.stop()
        saber.set_auto_stop(200)
        saber.set_ramping(8)  # Match robot_controller.py; persistent controller setting.
        print("Stop all other motor-control programs. Clear the travel area and have a physical stop ready.")
        print("A remote-shell disconnect is not a guaranteed stop. Ctrl+C interrupts this trial.")
        print("Verify stopping physically; the configured 200 ms timeout still needs hardware verification.")
        answer = input("Align straight. Type RUN and press Enter to execute this one trial: ")
        if answer.strip() != "RUN":
            print("Cancelled; no motion trial executed.")
            return 0
        session = args.output / datetime.now().strftime("trial_%Y%m%d_%H%M%S_%f")
        session.mkdir(parents=True, exist_ok=False)
        log_path = session / "readings.csv"
        print(f"Log: {log_path}")
        with log_path.open("x", newline="", buffering=1) as log:
            writer = csv.DictWriter(log, fieldnames=FIELDS)
            writer.writeheader()
            run_trial(saber, imu, args, writer)
            note = input("Robot moved/stopped correctly? Approximate distance and turn direction (Enter to skip): ")
            (session / "operator_note.txt").write_text(note + "\n")
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted; stop requested. Completed readings are saved.")
    except Exception as exc:
        print(f"Motor test failed: {exc}", file=sys.stderr)
        return 1
    finally:
        actions = ([saber.stop, saber.close] if saber is not None else [])
        actions += [imu.close] if imu is not None else []
        for action in actions:
            try:
                action()
            except Exception as exc:
                print(f"Cleanup warning: {exc}", file=sys.stderr)
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    return 0


if __name__ == "__main__":
    sys.exit(main())
